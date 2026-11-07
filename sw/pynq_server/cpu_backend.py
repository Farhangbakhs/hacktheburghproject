"""CPU backend: NumPy implementation of the fixed-point filter in PLAN.md section 3.

This is the server's stand-in until Jack's DMA backend exists, and runs on both
the laptop and the PYNQ ARM. It should agree bit-for-bit with Person 2's
sw/reference/evm_fixed.py; check against test_vectors/ once they exist.

Backend interface (Jack's DMA backend implements the same):
    process(words_in, r_hi, r_lo, alpha, init) -> words_out
      words_in : (H, W, 4) uint8   bytes B, G, R, 0
      words_out: (H, W, 4) int8    signal B, G, R, 0
    State is re-initialised on init=True or when the frame size changes.
"""
from __future__ import annotations

import numpy as np


class CpuBackend:
    name = "cpu"

    def __init__(self) -> None:
        self.l_hi = None  # int64 working copies of the int32 Q16.16 state
        self.l_lo = None

    def reset(self) -> None:
        self.l_hi = None
        self.l_lo = None

    def process(self, words_in: np.ndarray, r_hi: int, r_lo: int, alpha: int,
                init: bool) -> np.ndarray:
        p = words_in[:, :, :3].astype(np.int64)
        x = p << 16

        if init or self.l_hi is None or self.l_hi.shape != x.shape:
            self.l_hi = x.copy()
            self.l_lo = x.copy()
        else:
            # NumPy >> on signed ints is arithmetic (floor), matching the contract.
            self.l_hi += (r_hi * (x - self.l_hi)) >> 16
            self.l_lo += (r_lo * (x - self.l_lo)) >> 16

        band = self.l_hi - self.l_lo
        out = np.clip((alpha * band) >> 24, -127, 127).astype(np.int8)

        h, w, _ = words_in.shape
        words_out = np.zeros((h, w, 4), dtype=np.int8)
        words_out[:, :, :3] = out
        return words_out
