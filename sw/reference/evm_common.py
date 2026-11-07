"""Shared helpers for the EVM reference implementations (docs/PLAN.md section 3).

Used by both evm_float.py and (later) evm_fixed.py so the two differ only in
the temporal filter arithmetic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Section 3 defaults.
FS_DEFAULT = 30.0
FC_HI_DEFAULT = 2.0
FC_LO_DEFAULT = 0.83
ALPHA_DEFAULT = 50.0
SMALL_W, SMALL_H = 160, 120
OUT_CLIP = 127


def lowpass_coeff(fc: float, fs: float = FS_DEFAULT) -> float:
    """First-order IIR low-pass coefficient: r = 1 - exp(-2*pi*fc/fs)."""
    if fc <= 0 or fs <= 0:
        raise ValueError(f"fc and fs must be positive (fc={fc}, fs={fs})")
    return 1.0 - math.exp(-2.0 * math.pi * fc / fs)


def coeff_q016(fc: float, fs: float = FS_DEFAULT) -> int:
    """Coefficient as unsigned Q0.16, round(r * 65536), clamped to uint16."""
    return min(65535, int(round(lowpass_coeff(fc, fs) * 65536)))


def alpha_q88(alpha: float) -> int:
    """Gain as unsigned Q8.8, round(alpha * 256), clamped to uint16."""
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    return min(65535, int(round(alpha * 256)))


@dataclass
class FilterParams:
    """Run-time tunables (map 1:1 onto the AXI-Lite control registers)."""

    fc_lo: float = FC_LO_DEFAULT
    fc_hi: float = FC_HI_DEFAULT
    alpha: float = ALPHA_DEFAULT
    fs: float = FS_DEFAULT

    def validate(self) -> None:
        if not (0 < self.fc_lo < self.fc_hi < self.fs / 2):
            raise ValueError(
                f"need 0 < fc_lo < fc_hi < fs/2, got {self.fc_lo}, {self.fc_hi}, fs={self.fs}"
            )
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

    @property
    def r_hi(self) -> float:
        return lowpass_coeff(self.fc_hi, self.fs)

    @property
    def r_lo(self) -> float:
        return lowpass_coeff(self.fc_lo, self.fs)


def pyramid_levels_for(width: int, target_w: int = SMALL_W) -> int:
    """Number of pyrDown steps that brings `width` closest to `target_w`."""
    levels, w = 0, width
    while w // 2 >= target_w * 0.75:
        w = (w + 1) // 2
        levels += 1
    return levels


def downscale(frame: np.ndarray, levels: int) -> np.ndarray:
    """Gaussian pyramid downscale (cv2.pyrDown applied `levels` times)."""
    out = frame
    for _ in range(levels):
        out = cv2.pyrDown(out)
    return out


def upscale(small: np.ndarray, levels: int, shape_hw: tuple[int, int]) -> np.ndarray:
    """Inverse of downscale: pyrUp `levels` times, then snap to exact size.

    Works on any dtype cv2.pyrUp supports (uint8, int16, float32).
    """
    out = small
    for _ in range(levels):
        out = cv2.pyrUp(out)
    h, w = shape_hw
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def add_signal(frame: np.ndarray, signal_full: np.ndarray) -> np.ndarray:
    """frame (uint8) + amplified signal (any numeric) -> clipped uint8."""
    # cv2.add saturates to [0, 255] and rounds; ~5x faster than the NumPy float path
    return cv2.add(frame, signal_full.astype(np.float32, copy=False), dtype=cv2.CV_8U)
