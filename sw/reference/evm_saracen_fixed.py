"""Section 3 v2 (PROPOSED): saracen-style Laplacian EVM split between laptop and FPGA.

Bit-exact fixed-point model of the FPGA kernel + the laptop pre/post stages.
Algorithm = saracen/magnification (see cpp/saracen_evm.cpp) with our tuned face
settings, partitioned so the FPGA does exactly what it did in v1 (a per-value
difference-of-two-first-order-lowpass IIR + gain), only on different data.

Laptop pre (per frame):
    work  = pyrDown(frame_bgr)                     640x480 -> 320x240
    lab   = BGR2Lab(work / 255)                    float32, L in [0, 100]
    Lap   = Laplacian pyramid of L, `levels` bands (L0 = 320x240 ... L3 = 40x30) + residual
    send  = L1 .. L(levels-1)                      saracen zeroes L0 and the residual,
                                                   so they never leave the laptop
    word  = int16 Q8.8 = clip(rint(band * 256), -32767, 32767), low 16 bits of a
            32-bit LE word (two's complement), high 16 bits = 0
    order = level 1 rows (row-major), then level 2, then level 3

FPGA kernel (bit-exact, per value, 64-bit intermediates, arithmetic shifts):
    init (first frame or INIT):  L_hi = L_lo = v << 8
    x     = v << 8                                  Q8.8 -> Q16.16
    L_hi += (r_hi * (x - L_hi)) >> 16               r_hi, r_lo uint16 Q0.16 (as v1)
    L_lo += (r_lo * (x - L_lo)) >> 16
    band  = L_hi - L_lo                             Q16.16
    out   = clip((alpha[level] * band) >> 16, -32767, 32767)   alpha uint16 Q8.8 -> Q8.8
    level = index of the segment the value falls in (count registers)
    output word: int16 Q8.8 in the low 16 bits, high 16 bits = 0

Laptop post:
    filtered bands = out / 256 (L units); L0 and residual = 0
    motion = Laplacian reconstruction; lab.L += motion   (a, b unchanged: chroma
             attenuation 0 - see HANDOFF_JACK.md)
    delta  = Lab2BGR(lab) * 255 - work, masked to face skin, upsampled to 640x480

Per-level gains are saracen's spatial-wavelength clamp, computed on the laptop:
    delta_s = lambda_c / 8 / (1 + alpha);  lam_l = diag(work) / 3 / 2^(levels - l)
    alpha_l = clamp(min(alpha, (lam_l / delta_s / 8 - 1) * exaggeration), 0, 255.99)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from evm_common import FilterParams, coeff_q016

MAX_HW_LEVELS = 4
Q = 256          # Q8.8 scale
I16 = 32767


@dataclass
class SaracenHW:
    """Laptop-side configuration (the FPGA only sees the derived registers)."""
    levels: int = 4              # Laplacian bands at the work size (FPGA gets levels-1 of them)
    work_scale: int = 1          # pyrDown steps before processing (640x480 -> 320x240)
    lambda_c: float = 40.0
    exaggeration: float = 2.0
    settle_s: float = 0.0        # 0 = auto (as SaracenCppEVM)
    ramp_s: float = 0.5

    def validate(self) -> None:
        if not 2 <= self.levels <= MAX_HW_LEVELS + 1:
            raise ValueError(f"levels must be 2..{MAX_HW_LEVELS + 1}")


@dataclass
class Registers:
    r_hi: int
    r_lo: int
    alpha: list[int]                       # Q8.8 per FPGA level (len MAX_HW_LEVELS, unused = 0)
    count: list[int]                       # words per FPGA level (len MAX_HW_LEVELS, unused = 0)
    shapes: list[tuple[int, int]] = field(default_factory=list)  # (h, w) per FPGA level

    @property
    def total(self) -> int:
        return int(sum(self.count))

    def as_dict(self) -> dict:
        d = {"r_hi": self.r_hi, "r_lo": self.r_lo}
        for i in range(MAX_HW_LEVELS):
            d[f"alpha_{i}"] = self.alpha[i]
            d[f"count_{i}"] = self.count[i]
        d["total_words"] = self.total
        return d


# -- laptop helpers ------------------------------------------------------------------------
def work_frame(frame_bgr: np.ndarray, cfg: SaracenHW) -> np.ndarray:
    w = frame_bgr
    for _ in range(cfg.work_scale):
        w = cv2.pyrDown(w)
    return w


def laplacian_L(lab: np.ndarray, levels: int) -> list[np.ndarray]:
    """Laplacian pyramid of the L channel: [L0 .. L(levels-1), residual]."""
    cur = np.ascontiguousarray(lab[..., 0])
    pyr = []
    for _ in range(levels):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        pyr.append(cur - up)
        cur = down
    pyr.append(cur)
    return pyr


def level_alphas(alpha: float, cfg: SaracenHW, work_hw: tuple[int, int]) -> list[float]:
    """saracen's per-level gain for FPGA levels 1..levels-1 (index 0 = Laplacian level 1)."""
    h, w = work_hw
    delta_s = cfg.lambda_c / 8.0 / (1.0 + alpha)
    lam_top = math.sqrt(w * w + h * h) / 3.0          # at level == levels (residual)
    out = []
    for lvl in range(1, cfg.levels):
        lam = lam_top / (2 ** (cfg.levels - lvl))
        a = min(alpha, (lam / delta_s / 8.0 - 1.0) * cfg.exaggeration)
        out.append(float(np.clip(a, 0.0, 65535 / Q)))
    return out


def registers_for(params: FilterParams, cfg: SaracenHW, work_hw: tuple[int, int]) -> Registers:
    cfg.validate()
    h, w = work_hw
    shapes, hh, ww = [], h, w
    for lvl in range(cfg.levels):
        if lvl >= 1:
            shapes.append((hh, ww))
        hh, ww = (hh + 1) // 2, (ww + 1) // 2
    alphas = [int(round(a * Q)) for a in level_alphas(params.alpha, cfg, work_hw)]
    counts = [sh * sw for sh, sw in shapes]
    pad = MAX_HW_LEVELS - len(shapes)
    return Registers(coeff_q016(params.fc_hi, params.fs), coeff_q016(params.fc_lo, params.fs),
                     alphas + [0] * pad, counts + [0] * pad, shapes)


def laptop_pre(frame_bgr: np.ndarray, cfg: SaracenHW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (work uint8, lab float32, words int16 [total]) for one full-res frame."""
    work = work_frame(frame_bgr, cfg)
    lab = cv2.cvtColor(work.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    pyr = laplacian_L(lab, cfg.levels)
    send = [pyr[lvl] for lvl in range(1, cfg.levels)]
    words = np.concatenate([np.clip(np.rint(b * Q), -I16, I16).astype(np.int16).ravel() for b in send])
    return work, lab, words


def laptop_post(work: np.ndarray, lab: np.ndarray, out_words: np.ndarray, regs: Registers,
                cfg: SaracenHW) -> np.ndarray:
    """Kernel output words -> BGR delta (float32) at the work size."""
    filt, off = [], 0
    for (sh, sw), n in zip(regs.shapes, regs.count):
        filt.append(out_words[off:off + n].astype(np.float32).reshape(sh, sw) / Q)
        off += n
    # residual and L0 are zero (saracen); rebuild from the coarsest FPGA level
    top_h, top_w = (regs.shapes[-1][0] + 1) // 2, (regs.shapes[-1][1] + 1) // 2
    motion = np.zeros((top_h, top_w), np.float32)
    for f in reversed(filt):
        motion = cv2.pyrUp(motion, dstsize=(f.shape[1], f.shape[0])) + f
    motion = cv2.pyrUp(motion, dstsize=(lab.shape[1], lab.shape[0]))  # + L0 (= 0)
    lab2 = lab.copy()
    lab2[..., 0] += motion
    out = cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR) * 255.0
    return out - work.astype(np.float32)


# -- FPGA kernel model (bit-exact) ---------------------------------------------------------
class SaracenKernel:
    """process(words int16 [total], init) -> int16 [total]. State int32 Q16.16."""

    def __init__(self, regs: Registers):
        self.set_registers(regs)
        self._l_hi: np.ndarray | None = None
        self._l_lo: np.ndarray | None = None

    def set_registers(self, regs: Registers) -> None:
        for v in (regs.r_hi, regs.r_lo, *regs.alpha, *regs.count):
            if not 0 <= int(v) <= 0xFFFF:
                raise ValueError(f"register value {v} not uint16")
        self.regs = regs
        self._alpha_vec = np.repeat(np.array(regs.alpha, np.int64), np.array(regs.count, np.int64))

    def reset(self) -> None:
        self._l_hi = None

    def state(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._l_hi is None:
            return None
        return self._l_hi.astype(np.int32), self._l_lo.astype(np.int32)

    def process(self, words: np.ndarray, init: bool = False) -> np.ndarray:
        if words.dtype != np.int16 or words.shape != (self.regs.total,):
            raise ValueError(f"expected int16[{self.regs.total}], got {words.dtype}{words.shape}")
        x = words.astype(np.int64) << 8
        if init or self._l_hi is None:
            self._l_hi, self._l_lo = x.copy(), x.copy()
        self._l_hi += (np.int64(self.regs.r_hi) * (x - self._l_hi)) >> 16
        self._l_lo += (np.int64(self.regs.r_lo) * (x - self._l_lo)) >> 16
        band = self._l_hi - self._l_lo
        return np.clip((self._alpha_vec * band) >> 16, -I16, I16).astype(np.int16)


def pack_words(v: np.ndarray) -> bytes:
    """int16 values -> 32-bit LE words, value in low 16 bits, high 16 bits zero."""
    buf = np.zeros((v.size, 2), "<i2")
    buf[:, 0] = v
    return buf.tobytes()


def unpack_words(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, "<i2").reshape(-1, 2)
    if buf[:, 1].any():
        raise ValueError("high 16 bits of each word must be zero")
    return buf[:, 0].copy()


# -- dashboard processor (laptop pre -> kernel model -> laptop post) ------------------------
class SaracenFixedEVM:
    """Drop-in dashboard processor running the v2 split with the bit-exact kernel model
    (stand-in for the FPGA; Person 3's PynqRemote would replace SaracenKernel)."""

    full_res = True
    applies_mask = True

    def __init__(self, params: FilterParams | None = None, cfg: SaracenHW | None = None):
        self.params = params or FilterParams()
        self.cfg = cfg or SaracenHW()
        self.cfg.validate()
        self._kernel: SaracenKernel | None = None
        self._work_hw: tuple[int, int] | None = None
        self._mask_full: np.ndarray | None = None
        self._last_delta: np.ndarray | None = None
        self.frames_since_init = 0
        self._init = True

    @property
    def name(self) -> str:
        return f"saracen v2 fixed L{self.cfg.levels} lc{self.cfg.lambda_c:g}"

    @property
    def registers(self) -> Registers | None:
        return self._kernel.regs if self._kernel else None

    def set_params(self, params: FilterParams) -> None:
        params.validate()
        self.params = params
        if self._kernel is not None and self._work_hw is not None:
            self._kernel.set_registers(registers_for(params, self.cfg, self._work_hw))

    def set_mask_full(self, mask: np.ndarray | None) -> None:
        self._mask_full = mask

    def reset(self) -> None:
        self._init = True
        self.frames_since_init = 0

    def band_state(self) -> np.ndarray | None:
        return self._last_delta

    def warmup_gain(self) -> float:
        fs = self.params.fs
        r_lo = self.params.r_lo
        settle = self.cfg.settle_s * fs if self.cfg.settle_s > 0 else min(1.5 / max(r_lo, 1e-3), 3 * fs)
        return float(np.clip((self.frames_since_init - settle) / max(1.0, self.cfg.ramp_s * fs), 0, 1))

    @property
    def gate(self) -> float:
        return self.warmup_gain()

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        work, lab, words = laptop_pre(frame_bgr, self.cfg)
        hw = work.shape[:2]
        if self._kernel is None or self._work_hw != hw:
            self._work_hw = hw
            self._kernel = SaracenKernel(registers_for(self.params, self.cfg, hw))
            self._init = True
        out = self._kernel.process(words, init=self._init)
        if self._init:
            self._init, self.frames_since_init = False, 0
        self.frames_since_init += 1
        delta = laptop_post(work, lab, out, self._kernel.regs, self.cfg)
        g = np.float32(self.warmup_gain())
        if self._mask_full is not None:
            m = cv2.resize(self._mask_full, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) * g
            delta *= m[..., None]
        else:
            delta *= g
        if delta.shape[:2] != frame_bgr.shape[:2]:
            delta = cv2.resize(delta, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._last_delta = delta
        return delta


def make_saracen_fixed(params: FilterParams | None = None, **kw) -> SaracenFixedEVM:
    return SaracenFixedEVM(params, SaracenHW(**kw))


def main(argv: list[str] | None = None) -> int:
    import sys

    from dashboard import run_dashboard
    from evm_float import FloatEVM

    return run_dashboard(FloatEVM, ["--preset", "saracen_fixed", "--alpha", "20", "--fc-lo", "0.25",
                                    "--fc-hi", "2.44", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
