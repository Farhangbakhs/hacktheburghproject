"""Tests for the Laplacian (motion) EVM port of saracen/magnification."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import FilterParams  # noqa: E402
from evm_laplacian import LaplacianEVM, make_laplacian  # noqa: E402


def textured(h=240, w=320, shift=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 128 + 60 * np.sin((xx - shift) / 6.0) * np.cos(yy / 9.0)
    return np.clip(np.stack([v, v * 0.9, v * 0.8], -1), 0, 255).astype(np.uint8)


def test_constant_gives_zero_and_shape():
    evm = make_laplacian(FilterParams(alpha=20))
    f = textured()
    for _ in range(10):
        d = evm.process(f)
        assert d.shape == f.shape and d.dtype == np.float32
    assert np.abs(d).max() < 1.0


def test_subpixel_motion_is_amplified():
    p = FilterParams(fc_lo=0.5, fc_hi=3.0, alpha=20)
    evm = LaplacianEVM(p)
    still = textured()
    amp_out = []
    for i in range(150):
        f = textured(shift=0.3 * math.sin(2 * math.pi * 1.2 * i / 30))
        d = evm.process(f)
        if i > 60:
            amp_out.append(np.abs(d).mean())
    in_change = np.abs(textured(shift=0.3).astype(float) - still).mean()
    assert np.mean(amp_out) > 2 * in_change


def test_reset_and_speed():
    evm = make_laplacian(FilterParams(alpha=20))
    f = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    evm.process(f)
    t0 = time.perf_counter()
    for _ in range(5):
        evm.process(f)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"laplacian 640x480: {ms:.1f} ms/frame")
    evm.reset()
    assert evm.frames_since_init == 0
    assert ms < 200
