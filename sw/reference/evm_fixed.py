"""Bit-exact fixed-point model of the EVM kernel (docs/PLAN.md section 3).

This is the golden reference the HLS kernel must match bit-for-bit.

Per pixel, per channel (B, G, R), 64-bit intermediates, arithmetic (floor) shifts:

    init (first frame or INIT flag):   L_hi = L_lo = p << 16
    x     = p << 16
    L_hi += (r_hi * (x - L_hi)) >> 16
    L_lo += (r_lo * (x - L_lo)) >> 16
    band  = L_hi - L_lo                               # Q16.16
    out   = clip((alpha * band) >> 24, -127, 127)     # alpha Q8.8 -> int8

    r_hi, r_lo : uint16 Q0.16 = round(r * 65536), r = 1 - exp(-2*pi*fc/fs)
    alpha      : uint16 Q8.8  = round(alpha * 256)
    L_hi, L_lo : int32 Q16.16 (they stay in [0, 255<<16]; proof in tests)

Notes for HLS (C++):
  * `>>` on a negative signed value must be an ARITHMETIC shift (floor), e.g.
    (-1) >> 16 == -1. ap_int / ap_fixed and int64_t on gcc/MSVC behave this way.
  * products need 64 bits: r * (x - L) up to 65535 * 2^24, alpha * band up to
    65535 * 2^24.
  * On an INIT frame the update still runs but changes nothing (x == L), so
    the output of an INIT frame is all zeros.

DMA word layout (one pixel per 32-bit little-endian word):
  input : byte0 = B, byte1 = G, byte2 = R, byte3 = 0           (uint8)
  output: byte0 = B, byte1 = G, byte2 = R, byte3 = 0           (int8, two's complement)
"""
from __future__ import annotations

import numpy as np

from evm_common import OUT_CLIP, FilterParams, alpha_q88, coeff_q016


class FixedEVM:
    """Bit-exact section 3 kernel model on small BGR frames.

    process_int8(small_bgr uint8 HxWx3, init=False) -> int8 HxWx3   (contract output)
    process(small_bgr)                              -> float32 copy (dashboard interface)
    """

    name = "fixed"

    def __init__(self, params: FilterParams | None = None):
        self.params = params or FilterParams()
        self.params.validate()
        self._regs_from_params(self.params)
        self._l_hi: np.ndarray | None = None
        self._l_lo: np.ndarray | None = None
        self._init_pending = True
        self.frames_since_init = 0

    # -- registers ----------------------------------------------------------------
    def _regs_from_params(self, p: FilterParams) -> None:
        self.r_hi = coeff_q016(p.fc_hi, p.fs)
        self.r_lo = coeff_q016(p.fc_lo, p.fs)
        self.alpha = alpha_q88(p.alpha)

    def set_registers(self, r_hi: int, r_lo: int, alpha: int) -> None:
        """Set raw register values (uint16 each), exactly as the PYNQ server would."""
        for name, v in (("r_hi", r_hi), ("r_lo", r_lo), ("alpha", alpha)):
            if not 0 <= int(v) <= 0xFFFF:
                raise ValueError(f"{name}={v} not a uint16")
        self.r_hi, self.r_lo, self.alpha = int(r_hi), int(r_lo), int(alpha)

    def set_params(self, params: FilterParams) -> None:
        params.validate()
        self.params = params
        self._regs_from_params(params)

    @property
    def registers(self) -> dict:
        return {"r_hi": self.r_hi, "r_lo": self.r_lo, "alpha": self.alpha}

    # -- state ------------------------------------------------------------------------
    def reset(self) -> None:
        """INIT flag: re-initialise state from the next frame."""
        self._init_pending = True

    def state(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(L_hi, L_lo) as int32 Q16.16 arrays HxWx3, or None before the first frame."""
        if self._l_hi is None:
            return None
        return self._l_hi.astype(np.int32), self._l_lo.astype(np.int32)

    def band_state(self) -> np.ndarray | None:
        """Band in pixel units (float32), for the dashboard trace."""
        if self._l_hi is None:
            return None
        return ((self._l_hi - self._l_lo) / 65536.0).astype(np.float32)

    # -- kernel -------------------------------------------------------------------------
    def process_int8(self, small_bgr: np.ndarray, init: bool = False) -> np.ndarray:
        if small_bgr.dtype != np.uint8 or small_bgr.ndim != 3 or small_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8, got {small_bgr.shape} {small_bgr.dtype}")
        x = small_bgr.astype(np.int64) << 16
        if init or self._init_pending or self._l_hi is None or self._l_hi.shape != x.shape:
            self._l_hi = x.copy()
            self._l_lo = x.copy()
            self._init_pending = False
            self.frames_since_init = 0
        # numpy >> on signed int64 is an arithmetic shift (floor), as required.
        self._l_hi += (np.int64(self.r_hi) * (x - self._l_hi)) >> 16
        self._l_lo += (np.int64(self.r_lo) * (x - self._l_lo)) >> 16
        self.frames_since_init += 1
        band = self._l_hi - self._l_lo
        out = (np.int64(self.alpha) * band) >> 24
        return np.clip(out, -OUT_CLIP, OUT_CLIP).astype(np.int8)

    def process(self, small_bgr: np.ndarray) -> np.ndarray:
        return self.process_int8(small_bgr).astype(np.float32)


# -- DMA word packing (section 3 stream format) --------------------------------------
def pack_input(frame_bgr: np.ndarray) -> bytes:
    """HxWx3 uint8 BGR -> H*W little-endian 32-bit words: B, G, R, 0."""
    h, w, _ = frame_bgr.shape
    buf = np.zeros((h, w, 4), np.uint8)
    buf[..., :3] = frame_bgr
    return buf.tobytes()


def unpack_input(data: bytes, h: int, w: int) -> np.ndarray:
    buf = np.frombuffer(data, np.uint8).reshape(h, w, 4)
    if buf[..., 3].any():
        raise ValueError("input byte3 must be 0")
    return buf[..., :3].copy()


def pack_output(signal_int8: np.ndarray) -> bytes:
    """HxWx3 int8 -> H*W words: int8 B, G, R, then a 0 byte."""
    h, w, _ = signal_int8.shape
    buf = np.zeros((h, w, 4), np.int8)
    buf[..., :3] = signal_int8
    return buf.tobytes()


def unpack_output(data: bytes, h: int, w: int) -> np.ndarray:
    buf = np.frombuffer(data, np.int8).reshape(h, w, 4)
    return buf[..., :3].copy()


def main(argv: list[str] | None = None) -> int:
    """Live dashboard on the bit-exact model (should look like the float one)."""
    import sys

    from dashboard import run_dashboard

    return run_dashboard(FixedEVM, sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
