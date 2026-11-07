"""Tests for the embedded saracen/magnification C++ DLL (skipped if not built)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import FilterParams  # noqa: E402

try:
    from evm_saracen_cpp import load_library, make_saracen
    load_library()
except Exception as e:  # pragma: no cover
    pytest.skip(f"saracen_evm.dll not available: {e}", allow_module_level=True)


def textured(h=240, w=320, shift=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 128 + 60 * np.sin((xx - shift) / 6.0) * np.cos(yy / 9.0)
    return np.clip(np.stack([v, v * 0.9, v * 0.8], -1), 0, 255).astype(np.uint8)


def test_static_scene_unchanged():
    evm = make_saracen(FilterParams(alpha=20))
    f = textured()
    for _ in range(10):
        d = evm.process(f)
    assert d.shape == f.shape and np.abs(d).mean() < 1.0


def test_subpixel_motion_amplified():
    evm = make_saracen(FilterParams(fc_lo=0.5, fc_hi=3.0, alpha=20))
    amp = []
    for i in range(150):
        d = evm.process(textured(shift=0.3 * math.sin(2 * math.pi * 1.2 * i / 30)))
        if i > 60:
            amp.append(np.abs(d).mean())
    change = np.abs(textured(shift=0.3).astype(float) - textured()).mean()
    assert np.mean(amp) > 2 * change


def test_reset_and_speed():
    evm = make_saracen(FilterParams(alpha=20))
    f = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    evm.process(f)
    t0 = time.perf_counter()
    for _ in range(10):
        evm.process(f)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"saracen C++ 640x480: {ms:.1f} ms/frame")
    evm.reset()
    assert evm.frames_since_init == 0 and ms < 200
