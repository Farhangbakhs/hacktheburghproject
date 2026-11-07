"""Float reference of the EVM temporal bandpass + amplify (docs/PLAN.md section 3).

Same maths as the fixed-point contract, in float:
    L_hi += r_hi * (x - L_hi)
    L_lo += r_lo * (x - L_lo)
    band  = L_hi - L_lo
    out   = clip(alpha * band, -127, 127)

Run live:  python evm_float.py            (see --help; keys listed in dashboard.py)
"""
from __future__ import annotations

import numpy as np

from evm_common import OUT_CLIP, FilterParams


class FloatEVM:
    """Per-pixel, per-channel IIR bandpass + gain on small BGR frames.

    process(small_bgr uint8 HxWx3) -> float32 HxWx3 amplified signal in
    [-127, 127] (pixel units). State re-initialises on the first frame,
    after reset(), or when the frame shape changes.
    """

    name = "float"

    def __init__(self, params: FilterParams | None = None):
        self.params = params or FilterParams()
        self.params.validate()
        self._l_hi: np.ndarray | None = None
        self._l_lo: np.ndarray | None = None
        self.frames_since_init = 0

    def set_params(self, params: FilterParams) -> None:
        """Change cutoffs/gain live without resetting filter state."""
        params.validate()
        self.params = params

    def reset(self) -> None:
        """Equivalent of the INIT flag: re-initialise from the next frame."""
        self._l_hi = self._l_lo = None

    def process(self, small_bgr: np.ndarray) -> np.ndarray:
        if small_bgr.dtype != np.uint8 or small_bgr.ndim != 3 or small_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8, got {small_bgr.shape} {small_bgr.dtype}")
        x = small_bgr.astype(np.float32)
        if self._l_hi is None or self._l_hi.shape != x.shape:
            self._l_hi = x.copy()
            self._l_lo = x.copy()
            self.frames_since_init = 0
        p = self.params
        self._l_hi += np.float32(p.r_hi) * (x - self._l_hi)
        self._l_lo += np.float32(p.r_lo) * (x - self._l_lo)
        self.frames_since_init += 1
        band = self._l_hi - self._l_lo
        return np.clip(np.float32(p.alpha) * band, -OUT_CLIP, OUT_CLIP)

    def process_int8(self, small_bgr: np.ndarray) -> np.ndarray:
        """Same output floored to int8 (the contract's arithmetic >> floors).

        Float rounding means this can still differ from evm_fixed.py by a few
        LSB; only evm_fixed.py is bit-exact.
        """
        return np.floor(self.process(small_bgr)).astype(np.int8)

    def band_state(self) -> np.ndarray | None:
        """Un-amplified bandpass output (debug/visualisation)."""
        if self._l_hi is None:
            return None
        return self._l_hi - self._l_lo


def main(argv: list[str] | None = None) -> int:
    from dashboard import run_dashboard

    return run_dashboard(FloatEVM, argv)


if __name__ == "__main__":
    raise SystemExit(main())
