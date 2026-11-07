"""Tests for the pulse-locked, face-only live EVM."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import FilterParams, downscale  # noqa: E402
from evm_eval import synthetic_clip  # noqa: E402
from evm_locked import PulseLockedEVM, contract_gain, pos_signal, spectrum_peak  # noqa: E402

FS = 30.0


def run(frames, mask=None, alpha=50, **kw):
    from evm_locked import LockConfig
    evm = PulseLockedEVM(FilterParams(alpha=alpha), LockConfig(**kw))
    outs = []
    ms = downscale(mask.astype(np.float32), 1) if mask is not None else None
    for f in frames:
        evm.set_mask(ms)
        outs.append(evm.process(downscale(f, 1)))
    return evm, outs, ms


@pytest.fixture(scope="module")
def pulse_clip():
    return synthetic_clip(n=450, h=240, w=320, pulse_hz=1.3, pulse_amp=0.5, noise_std=2.0)


@pytest.mark.parametrize("mix", [0.0, 0.6, 1.0])
def test_locks_and_renders_pulse_on_skin_only(pulse_clip, mix):
    frames, truth = pulse_clip
    evm, outs, ms = run(frames, truth["skin_mask"], mix=mix)
    assert evm.locked and evm.f0 == pytest.approx(1.3, abs=0.05)
    skin = ms > 0.9
    g = np.array([o[..., 1][skin].mean() for o in outs[-150:]])
    spec = np.abs(np.fft.rfft(g - g.mean()))
    assert np.fft.rfftfreq(len(g), 1 / FS)[spec.argmax()] == pytest.approx(1.3, abs=0.1)
    # rendered green swing ~ alpha x true green pulse (0.5 px), within a factor of 3
    amp = g.std() * np.sqrt(2)
    assert 50 * 0.5 / 3 < amp < 50 * 0.5 * 3, amp
    # nothing drawn off the skin mask
    assert np.abs(outs[-1][ms < 1e-3]).max() < 1e-3


def test_no_pulse_no_lock_no_output():
    frames, truth = synthetic_clip(n=300, h=240, w=320, pulse_amp=0.0, noise_std=2.0)
    evm, outs, _ = run(frames, truth["skin_mask"])
    assert not evm.locked
    assert np.abs(outs[-1]).max() < 1.0


def test_constant_input_zero_and_reset():
    evm = PulseLockedEVM()
    f = np.full((60, 80, 3), 100, np.uint8)
    for _ in range(200):
        assert np.all(evm.process(f) == 0)
    evm.reset()
    assert evm.f0 is None and not evm.locked


def test_pos_recovers_pulse_under_luminance_noise():
    rng = np.random.default_rng(1)
    t = np.arange(450) / FS
    lum = rng.normal(0, 1.0, len(t))  # common-mode (brightness) noise, larger than the pulse
    pulse = 0.3 * np.sin(2 * np.pi * 1.2 * t)
    base = np.array([60.0, 80.0, 120.0])  # BGR
    pbv_bgr = np.array([0.53, 0.77, 0.33])
    c = base * (1 + 0.01 * lum[:, None]) + base * pbv_bgr * pulse[:, None] / 100
    f, snr = spectrum_peak(pos_signal(c, FS), FS, (0.7, 3.0))
    assert f == pytest.approx(1.2, abs=0.05) and snr > 3


def test_contract_gain_peak():
    assert 0.35 < contract_gain(1.3, FilterParams()) < 0.45
