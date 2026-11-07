"""Tests for the offline MIT EVM replica."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_eval import synthetic_clip  # noqa: E402
from evm_mit import (  # noqa: E402
    MITParams,
    bgr_to_yiq,
    estimate_pulse_hz,
    gaussian_stack,
    ideal_bandpass,
    magnify,
    yiq_to_bgr,
)


def test_yiq_roundtrip():
    f = np.random.default_rng(0).integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
    assert np.abs(yiq_to_bgr(bgr_to_yiq(f)).astype(int) - f).max() <= 1


def test_ideal_bandpass_keeps_only_band():
    fs, n = 30.0, 300
    t = np.arange(n) / fs
    x = (np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 3.0 * t))[:, None, None, None].astype(np.float32)
    y = ideal_bandpass(x, 0.8, 1.2, fs)[:, 0, 0, 0]
    assert np.allclose(y, np.sin(2 * np.pi * 1.0 * t), atol=1e-3)


def test_pulse_found_and_amplified():
    frames, truth = synthetic_clip(n=300, h=240, w=320, pulse_hz=1.2, pulse_amp=0.5, noise_std=2.0)
    frames = np.stack(frames)
    stack = gaussian_stack(bgr_to_yiq(frames), 3)
    f0, snr = estimate_pulse_hz(stack, 30.0, truth["skin_mask"])
    assert f0 == pytest.approx(1.2, abs=0.05) and snr > 5
    mag, _ = magnify(frames, MITParams(alpha=50, level=3, fl=f0 - 0.1, fh=f0 + 0.1))
    m = truth["skin_mask"]
    g_in = frames[..., 1][:, m].mean(axis=1)
    g_out = mag[..., 1][:, m].mean(axis=1).astype(float)
    assert g_out.std() > 10 * g_in.std()
    spec = np.abs(np.fft.rfft(g_out - g_out.mean()))
    assert np.fft.rfftfreq(len(g_out), 1 / 30)[spec.argmax()] == pytest.approx(1.2, abs=0.05)
