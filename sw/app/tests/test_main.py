"""End-to-end app test without a camera or window."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import App, downscale, face_mask, magnify  # noqa: E402


class SyntheticSource:
    """640x480 noisy scene pulsing at `bpm` (no face, so centre ROI).

    Like blood, the pulse is a colour change (strongest in green), not pure
    brightness, which the CHROM estimator deliberately cancels.
    """
    PULSE_BGR = np.array([0.3, 1.0, 0.5], np.float32)
    auto_exposure = None

    def __init__(self, bpm=72.0, n=400, seed=1):
        self.hz, self.n, self.i = bpm / 60.0, n, 0
        self.rng = np.random.default_rng(seed)
        self.base = self.rng.integers(60, 190, size=(480, 640, 3)).astype(np.float32)

    def read(self):
        if self.i >= self.n:
            return None
        pulse = 1.0 * math.sin(2 * math.pi * self.hz * self.i / 30.0)
        self.i += 1
        frame = self.base + pulse * self.PULSE_BGR + self.rng.normal(0, 1.0, size=self.base.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def set_auto_exposure(self, on):
        pass

    def release(self):
        pass


@pytest.mark.parametrize("shape", [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)])
def test_downscale_exact_size(shape):
    assert downscale(np.zeros(shape, np.uint8), 160, 120).shape == (120, 160, 3)


def test_magnify_adds_signal():
    frame = np.full((480, 640, 3), 100, np.uint8)
    sig = np.full((120, 160, 3), -20, np.int8)
    assert np.all(magnify(frame, sig) == 80)


@pytest.mark.parametrize("backend,bpm", [("local", 72), ("mock", 72), ("local", 140)])
def test_app_headless_bpm(backend, bpm):
    app = App(SyntheticSource(bpm=bpm), backend, "127.0.0.1", (160, 120),
              headless=True, realtime=False)
    app.run()
    assert app.frames == 400
    assert app.backend == backend and app.errors == 0
    assert app.bpm.reliable and abs(app.bpm.bpm - bpm) < 3, app.bpm.bpm


def test_face_mask_fades_off_face():
    m = face_mask((480, 640, 3), (220, 140, 200, 200))
    assert m.shape == (480, 640) and m.dtype == np.float32
    assert m[250, 320] > 0.95          # face centre
    assert m[10, 10] < 0.01            # background corner
    frame = np.full((480, 640, 3), 100, np.uint8)
    sig = np.full((120, 160, 3), 20, np.int8)
    out = magnify(frame, sig, mask=m)
    assert out[250, 320, 1] >= 118 and out[10, 10, 1] == 100


def test_pynq_unreachable_falls_back_to_local():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    closed_port = s.getsockname()[1]
    s.close()  # nothing listens here now (5005 may be a real server you're running)
    app = App(SyntheticSource(n=5), "pynq", "127.0.0.1", (160, 120), headless=True,
              realtime=False, port=closed_port)
    assert app.backend == "local"
    app.run()
    assert app.frames == 5
