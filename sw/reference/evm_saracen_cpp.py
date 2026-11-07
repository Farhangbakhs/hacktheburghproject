"""ctypes wrapper for cpp/build/saracen_evm.dll (saracen/magnification C++ EVM).

Laplacian-pyramid motion EVM in Lab colour, unchanged from the C++ repo
(see cpp/saracen_evm.cpp). Exposed with the dashboard processor interface;
`full_res = True`, so the dashboard passes full frames and masks the returned
delta to the face skin.

Cutoffs: the C++ takes raw IIR coefficients r (its defaults 0.4 / 0.05). We
convert FilterParams Hz -> r = 1 - exp(-2 pi fc / fs), so the dashboard's
sliders keep meaning Hz. saracen's defaults correspond to 0.25 - 2.44 Hz at 30 fps.

Build first:  cpp\\build.cmd   (needs opencv_world4140.dll in the OpenCV SDK folder)
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from evm_common import FilterParams

BUILD = Path(__file__).parent / "cpp" / "build"
OPENCV_BIN = Path(os.environ.get("OPENCV_DIR", Path(os.environ.get("LOCALAPPDATA", "")) / "theTell" / "opencv" / "build")) / "x64" / "vc16" / "bin"

_lib = None


def load_library() -> ctypes.CDLL:
    """Load saracen_evm.dll (and opencv_world4140.dll next to it or in the SDK)."""
    global _lib
    if _lib is not None:
        return _lib
    dll = BUILD / "saracen_evm.dll"
    if not dll.exists():
        raise FileNotFoundError(f"{dll} not built - run cpp\\build.cmd")
    for d in (BUILD, OPENCV_BIN):
        if d.exists():
            os.add_dll_directory(str(d))
    lib = ctypes.CDLL(str(dll))
    d, i, vp = ctypes.c_double, ctypes.c_int, ctypes.c_void_p
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    lib.evm_create.argtypes = [i, d, d, d, d, d, d]
    lib.evm_create.restype = vp
    lib.evm_set.argtypes = [vp, d, d, d, d, d, d]
    lib.evm_reset.argtypes = [vp]
    lib.evm_destroy.argtypes = [vp]
    lib.evm_process.argtypes = [vp, u8p, i, i, u8p]
    lib.evm_process.restype = i
    _lib = lib
    return lib


@dataclass
class SaracenConfig:
    """Defaults tuned for faces on a 640x480 webcam (measured on recordings/brightroom.npz):
    half-res + lambda_c 40 -> ~15 ms/frame (30 fps possible), 64% less skin grain than
    saracen's full-res lambda_c 20, keeping ~70% of the magnified motion.
    saracen's own defaults: levels 5, lambda_c 20, work_scale 0."""
    levels: int = 4
    lambda_c: float = 40.0
    chrom_attenuation: float = 0.1
    exaggeration: float = 2.0
    work_scale: int = 1          # pyrDown the frame this many times before the DLL (speed + less grain)
    settle_s: float = 0.0        # 0 = auto: ~1.5 time constants of the slow low-pass
    ramp_s: float = 0.5          # fade-in after settling, so the start-up transient is never shown


class SaracenCppEVM:
    full_res = True
    applies_mask = True   # masks at the (smaller) working size; dashboard skips its full-res mask

    def __init__(self, params: FilterParams | None = None, config: SaracenConfig | None = None):
        self.lib = load_library()
        self.params = params or FilterParams()
        self.cfg = config or SaracenConfig()
        c = self.cfg
        self._h = self.lib.evm_create(c.levels, self.params.alpha, c.lambda_c, self.params.r_hi,
                                      self.params.r_lo, c.chrom_attenuation, c.exaggeration)
        self._out: np.ndarray | None = None
        self._last_delta: np.ndarray | None = None
        self.frames_since_init = 0
        self._mask_full: np.ndarray | None = None

    def set_mask_full(self, mask: np.ndarray | None) -> None:
        """Feathered skin mask (float32 HxW, full frame size) or None for no masking."""
        self._mask_full = mask

    @property
    def name(self) -> str:
        return f"saracen C++ L{self.cfg.levels} chrom{self.cfg.chrom_attenuation:g}"

    def set_params(self, params: FilterParams) -> None:
        params.validate()
        self.params = params
        c = self.cfg
        self.lib.evm_set(self._h, params.alpha, c.lambda_c, params.r_hi, params.r_lo,
                         c.chrom_attenuation, c.exaggeration)

    def reset(self) -> None:
        self.lib.evm_reset(self._h)
        self.frames_since_init = 0

    def band_state(self) -> np.ndarray | None:
        return self._last_delta

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Magnified uint8 BGR frame, exactly what the C++ program displays."""
        f = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        if self._out is None or self._out.shape != f.shape:
            self._out = np.empty_like(f)
        u8p = ctypes.POINTER(ctypes.c_ubyte)
        self.frames_since_init = self.lib.evm_process(
            self._h, f.ctypes.data_as(u8p), f.shape[1], f.shape[0], self._out.ctypes.data_as(u8p))
        return self._out

    def warmup_gain(self) -> float:
        """0 while the IIR filter settles after (re)start, then ramps to 1."""
        fs = self.params.fs
        settle = self.cfg.settle_s * fs if self.cfg.settle_s > 0 else min(1.5 / max(self.params.r_lo, 1e-3), 3 * fs)
        ramp = max(1.0, self.cfg.ramp_s * fs)
        return float(np.clip((self.frames_since_init - settle) / ramp, 0.0, 1.0))

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """float32 delta (magnified - input) so the caller can mask it to the face."""
        work = frame_bgr
        for _ in range(self.cfg.work_scale):
            work = cv2.pyrDown(work)
        out = self.process_frame(work)
        delta = cv2.subtract(out, work, dtype=cv2.CV_32F)
        wh = (work.shape[1], work.shape[0])
        g = np.float32(self.warmup_gain())
        if self._mask_full is not None:
            m = cv2.resize(self._mask_full, wh, interpolation=cv2.INTER_AREA) * g
            delta = cv2.multiply(delta, cv2.merge([m, m, m]))
        elif g != 1:
            delta *= g
        if delta.shape[:2] != frame_bgr.shape[:2]:
            delta = cv2.resize(delta, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._last_delta = delta
        return delta

    @property
    def gate(self) -> float:
        return self.warmup_gain()

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and getattr(self, "lib", None) is not None:
            self.lib.evm_destroy(h)
            self._h = None


def make_saracen(params: FilterParams | None = None, **kw) -> SaracenCppEVM:
    return SaracenCppEVM(params, SaracenConfig(**kw))


def main(argv: list[str] | None = None) -> int:
    import sys

    from dashboard import run_dashboard
    from evm_float import FloatEVM

    # saracen defaults: alpha 20, r 0.4/0.05 (= 2.44 / 0.25 Hz at 30 fps)
    return run_dashboard(FloatEVM, ["--preset", "saracen_cpp", "--alpha", "20", "--fc-lo", "0.25",
                                    "--fc-hi", "2.44", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
