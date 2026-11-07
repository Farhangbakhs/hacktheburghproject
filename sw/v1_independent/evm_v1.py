"""Independent implementation of PLAN.md section 3 (v1 colour EVM filter).

Written from the contract text only (not from sw/reference/). Cross-checked against
Person 2's golden vectors in test_vectors/v1_contract/ by verify_vectors.py.

Per pixel, per channel (BGR):
    x     = p << 16
    L_hi += (r_hi * (x - L_hi)) >> 16          (arithmetic shift, 64-bit intermediates)
    L_lo += (r_lo * (x - L_lo)) >> 16
    band  = L_hi - L_lo                         (Q16.16)
    out   = clip((alpha * band) >> 24, -127, 127)   (alpha Q8.8)
INIT: L_hi = L_lo = p << 16.
"""
from __future__ import annotations

import math
import numpy as np

FS, FC_HI, FC_LO, ALPHA = 30.0, 2.0, 0.83, 50.0


def coeff_q016(fc: float, fs: float = FS) -> int:
    """round((1 - exp(-2*pi*fc/fs)) * 65536) as uint16."""
    return min(65535, int(round((1.0 - math.exp(-2.0 * math.pi * fc / fs)) * 65536)))


def alpha_q88(alpha: float) -> int:
    return min(65535, int(round(alpha * 256)))


class FixedV1:
    """Bit-exact fixed-point model. process() takes HxWx3 uint8, returns HxWx3 int8."""

    def __init__(self, r_hi=None, r_lo=None, alpha=None):
        self.r_hi = coeff_q016(FC_HI) if r_hi is None else int(r_hi)
        self.r_lo = coeff_q016(FC_LO) if r_lo is None else int(r_lo)
        self.alpha = alpha_q88(ALPHA) if alpha is None else int(alpha)
        self.l_hi = self.l_lo = None

    def reset(self):
        self.l_hi = self.l_lo = None

    def process(self, frame: np.ndarray, init: bool = False) -> np.ndarray:
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
        x = frame.astype(np.int64) << 16
        if init or self.l_hi is None or self.l_hi.shape != x.shape:
            self.l_hi = x.copy()
            self.l_lo = x.copy()
        # numpy >> on signed int64 is an arithmetic (floor) shift
        self.l_hi += (self.r_hi * (x - self.l_hi)) >> 16
        self.l_lo += (self.r_lo * (x - self.l_lo)) >> 16
        band = self.l_hi - self.l_lo
        out = np.clip((self.alpha * band) >> 24, -127, 127)
        return out.astype(np.int8)

    def state_int32(self):
        """(L_hi, L_lo) as int32; raises if the contract's int32 state would overflow."""
        for a in (self.l_hi, self.l_lo):
            if a.min() < -(2**31) or a.max() > 2**31 - 1:
                raise OverflowError("state exceeds int32")
        return self.l_hi.astype(np.int32), self.l_lo.astype(np.int32)


class FloatV1:
    """Same maths in float64 (no quantisation). Output not rounded."""

    def __init__(self, fc_hi=FC_HI, fc_lo=FC_LO, alpha=ALPHA, fs=FS):
        self.r_hi = 1 - math.exp(-2 * math.pi * fc_hi / fs)
        self.r_lo = 1 - math.exp(-2 * math.pi * fc_lo / fs)
        self.alpha = alpha
        self.l_hi = self.l_lo = None

    def reset(self):
        self.l_hi = self.l_lo = None

    def process(self, frame: np.ndarray, init: bool = False) -> np.ndarray:
        x = frame.astype(np.float64)
        if init or self.l_hi is None or self.l_hi.shape != x.shape:
            self.l_hi, self.l_lo = x.copy(), x.copy()
        self.l_hi += self.r_hi * (x - self.l_hi)
        self.l_lo += self.r_lo * (x - self.l_lo)
        return np.clip(self.alpha * (self.l_hi - self.l_lo), -127, 127)


# ---- DMA word layout helpers (one pixel per 32-bit LE word: B, G, R, 0) ----
def frames_to_words(frames: np.ndarray) -> np.ndarray:
    """(N,H,W,3) uint8/int8 -> (N,H,W) uint32 words (byte3 = 0)."""
    n, h, w, _ = frames.shape
    buf = np.zeros((n, h, w, 4), np.uint8)
    buf[..., :3] = frames.view(np.uint8)
    return buf.view("<u4").reshape(n, h, w)


def words_to_frames(words: np.ndarray, signed: bool = False) -> np.ndarray:
    n, h, w = words.shape
    b = np.ascontiguousarray(words.astype("<u4")).view(np.uint8).reshape(n, h, w, 4)[..., :3]
    return b.view(np.int8) if signed else b.copy()


def load_vectors(d, inp="input_frames.bin"):
    import json
    from pathlib import Path
    d = Path(d)
    p = json.loads((d / "params.json").read_text())
    n, h, w = p["N"], p["h"], p["w"]
    fin = words_to_frames(np.fromfile(d / inp, "<u4").reshape(n, h, w))
    fout = words_to_frames(np.fromfile(d / "expected_output.bin", "<u4").reshape(n, h, w), signed=True)
    st = np.fromfile(d / "expected_state.bin", "<i4")
    st = st.reshape(p["state_frames"], 2, h, w, 3)
    return p, fin, fout, st
