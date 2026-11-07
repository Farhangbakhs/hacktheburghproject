"""Laplacian-pyramid (motion) EVM, ported from saracen/magnification (MIT licence).

Source: https://github.com/saracen/magnification src/main.cpp (C++, itself based
on kgram007/Eulerian-Motion-Magnification), implementing the *motion* variant
of Wu et al. 2012:

  1. BGR -> Lab (float, input scaled to [0,1])
  2. Laplacian pyramid with `levels` band levels + Gaussian residual
  3. per band level: the same difference-of-two-first-order-lowpass IIR as our
     section 3 contract (r_hi, r_lo), amplified by
        alpha_l = min(alpha, (lambda_l / delta / 8 - 1) * exaggeration)
     with delta = lambda_c / 8 / (1 + alpha), lambda_l = diag/3 halving per
     level (spatial-wavelength cutoff: fine levels get less gain).
     The coarsest residual and finest band are zeroed (as in the original).
  4. reconstruct, attenuate a/b chroma by chrom_attenuation, add to the input
  5. Lab -> BGR

Differences from the original, all marked:
  * cutoffs given in Hz (converted to r = 1 - exp(-2 pi fc / fs)); the C++
    takes raw r values (its defaults 0.4 / 0.05 = 2.44 / 0.25 Hz at 30 fps).
  * output is returned as a BGR *delta* (magnified - input) so the dashboard
    can restrict it to the face skin mask before adding.
  * optional work_scale: process at a pyrDown'ed size and upsample the delta.

NON-CONTRACT: needs a multi-level full-resolution pyramid; the FPGA kernel
in PLAN.md section 3 filters a single 160x120 level. Laptop-only / flag to team.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from evm_common import FilterParams


@dataclass
class LaplacianConfig:
    levels: int = 5
    lambda_c: float = 20.0
    chrom_attenuation: float = 0.1
    exaggeration: float = 2.0
    work_scale: int = 0          # 0 = full res, 1 = half (pyrDown once), ...
    clip: float = 60.0           # max |delta| per channel in pixel levels


class LaplacianEVM:
    """Full-resolution processor: process(frame_bgr) -> float32 BGR delta."""

    full_res = True

    def __init__(self, params: FilterParams | None = None, config: LaplacianConfig | None = None):
        self.params = params or FilterParams()
        self.params.validate()
        self.cfg = config or LaplacianConfig()
        self.reset()

    @property
    def name(self) -> str:
        c = self.cfg
        return f"laplacian L{c.levels} lc{c.lambda_c:g} chrom{c.chrom_attenuation:g}"

    def set_params(self, params: FilterParams) -> None:
        params.validate()
        self.params = params

    def reset(self) -> None:
        self._lp1: list[np.ndarray] | None = None
        self._lp2: list[np.ndarray] | None = None
        self._last_delta: np.ndarray | None = None
        self.frames_since_init = 0

    def band_state(self) -> np.ndarray | None:
        return self._last_delta

    # -- core -------------------------------------------------------------------
    @staticmethod
    def _laplacian(img: np.ndarray, levels: int) -> list[np.ndarray]:
        pyr, cur = [], img
        for _ in range(levels):
            down = cv2.pyrDown(cur)
            up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
            pyr.append(cur - up)
            cur = down
        pyr.append(cur)  # Gaussian residual at index `levels`
        return pyr

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        cfg, p = self.cfg, self.params
        work = frame_bgr
        for _ in range(cfg.work_scale):
            work = cv2.pyrDown(work)
        lab = cv2.cvtColor(work.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        pyr = self._laplacian(lab, cfg.levels)

        if self._lp1 is None or self._lp1[0].shape != pyr[0].shape:
            self._lp1 = [b.copy() for b in pyr]
            self._lp2 = [b.copy() for b in pyr]
            self.frames_since_init = 1
            self._last_delta = np.zeros_like(frame_bgr, np.float32)
            return self._last_delta
        self.frames_since_init += 1

        r_hi, r_lo = np.float32(p.r_hi), np.float32(p.r_lo)
        h, w = lab.shape[:2]
        delta_s = cfg.lambda_c / 8.0 / (1.0 + p.alpha)
        lam = math.sqrt(w * w + h * h) / 3.0
        filtered: list[np.ndarray] = [None] * len(pyr)  # type: ignore[list-item]
        for level in range(cfg.levels, -1, -1):  # coarsest first, as in the C++
            if level == cfg.levels or level == 0:
                filtered[level] = np.zeros_like(pyr[level])
            else:
                self._lp1[level] += r_hi * (pyr[level] - self._lp1[level])
                self._lp2[level] += r_lo * (pyr[level] - self._lp2[level])
                cur_alpha = (lam / delta_s / 8.0 - 1.0) * cfg.exaggeration
                filtered[level] = (self._lp1[level] - self._lp2[level]) * np.float32(
                    max(0.0, min(p.alpha, cur_alpha)))
            lam /= 2.0

        motion = filtered[cfg.levels]
        for level in range(cfg.levels - 1, -1, -1):
            up = cv2.pyrUp(motion, dstsize=(filtered[level].shape[1], filtered[level].shape[0]))
            motion = up + filtered[level]
        motion[..., 1:] *= np.float32(cfg.chrom_attenuation)

        out = cv2.cvtColor(lab + motion, cv2.COLOR_Lab2BGR) * 255.0
        delta = out - work.astype(np.float32)
        for _ in range(cfg.work_scale):
            delta = cv2.pyrUp(delta)
        if delta.shape[:2] != frame_bgr.shape[:2]:
            delta = cv2.resize(delta, (frame_bgr.shape[1], frame_bgr.shape[0]))
        np.clip(delta, -cfg.clip, cfg.clip, out=delta)
        self._last_delta = delta
        return delta


def make_laplacian(params: FilterParams | None = None, **kw) -> LaplacianEVM:
    return LaplacianEVM(params, LaplacianConfig(**kw))


def main(argv: list[str] | None = None) -> int:
    import sys

    from dashboard import run_dashboard
    from evm_float import FloatEVM

    return run_dashboard(FloatEVM, ["--preset", "laplacian", "--alpha", "20", "--fc-lo", "0.25",
                                    "--fc-hi", "2.44", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
