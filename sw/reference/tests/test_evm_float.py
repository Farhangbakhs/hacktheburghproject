"""Tests for the float EVM reference, shared helpers and dashboard (plan step 1)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import (  # noqa: E402
    FilterParams,
    add_signal,
    alpha_q88,
    coeff_q016,
    downscale,
    lowpass_coeff,
    pyramid_levels_for,
    upscale,
)
from evm_float import FloatEVM  # noqa: E402

FS = 30.0


def analytic_gain(f: float, p: FilterParams) -> float:
    """|H_hi - H_lo| at frequency f for H(z) = r / (1 - (1-r) z^-1)."""
    z = np.exp(1j * 2 * math.pi * f / p.fs)

    def h(r):
        return r / (1 - (1 - r) / z)

    return abs(h(p.r_hi) - h(p.r_lo))


def run_sine(f: float, p: FilterParams, n: int = 900, amp: float = 2.0) -> float:
    """Feed a uniform 4x4 frame oscillating at f; return steady-state output gain."""
    evm = FloatEVM(FilterParams(p.fc_lo, p.fc_hi, 1.0, p.fs))
    outs = []
    for i in range(n):
        v = 128 + amp * math.sin(2 * math.pi * f * i / p.fs)
        frame = np.full((4, 4, 3), v, np.float32)
        # bypass uint8 quantisation for a clean response measurement
        evm._l_hi = frame.copy() if evm._l_hi is None else evm._l_hi
        evm._l_lo = frame.copy() if evm._l_lo is None else evm._l_lo
        evm._l_hi += p.r_hi * (frame - evm._l_hi)
        evm._l_lo += p.r_lo * (frame - evm._l_lo)
        outs.append(float((evm._l_hi - evm._l_lo)[0, 0, 0]))
    tail = np.array(outs[n // 2:])
    return (tail.max() - tail.min()) / 2 / amp


# -- coefficients -----------------------------------------------------------------
def test_coefficients_match_contract():
    assert lowpass_coeff(2.0, FS) == pytest.approx(1 - math.exp(-2 * math.pi * 2 / 30))
    assert coeff_q016(2.0) == round((1 - math.exp(-2 * math.pi * 2.0 / 30)) * 65536)
    assert coeff_q016(0.83) == round((1 - math.exp(-2 * math.pi * 0.83 / 30)) * 65536)
    assert alpha_q88(50) == 12800
    assert 0 < coeff_q016(0.83) < coeff_q016(2.0) < 65536


@pytest.mark.parametrize("lo,hi", [(2.0, 1.0), (0.0, 1.0), (1.0, 20.0)])
def test_invalid_params_rejected(lo, hi):
    with pytest.raises(ValueError):
        FilterParams(lo, hi, 50, FS).validate()


# -- filter behaviour ----------------------------------------------------------
@pytest.mark.parametrize("f", [0.1, 0.5, 1.0, 1.2, 1.5, 2.0, 4.0, 8.0])
def test_frequency_response_matches_analytic(f):
    p = FilterParams()
    assert run_sine(f, p) == pytest.approx(analytic_gain(f, p), rel=0.03, abs=1e-3)


def test_passband_beats_stopbands():
    p = FilterParams()
    g_pulse = analytic_gain(1.2, p)  # 72 bpm
    assert g_pulse > 2 * analytic_gain(0.05, p)  # slow lighting drift
    assert g_pulse > 2 * analytic_gain(10.0, p)  # sensor noise


def test_constant_input_gives_zero_output():
    evm = FloatEVM()
    frame = np.full((120, 160, 3), 77, np.uint8)
    for _ in range(10):
        out = evm.process(frame)
        assert out.dtype == np.float32 and out.shape == frame.shape
        assert np.all(out == 0)


def test_reset_and_shape_change_reinitialise():
    evm = FloatEVM()
    evm.process(np.full((8, 8, 3), 10, np.uint8))
    out = evm.process(np.full((8, 8, 3), 200, np.uint8))
    assert np.abs(out).max() > 0
    evm.reset()
    assert np.all(evm.process(np.full((8, 8, 3), 200, np.uint8)) == 0)
    assert np.all(evm.process(np.full((6, 6, 3), 50, np.uint8)) == 0)  # new shape
    assert evm.frames_since_init == 1


def test_output_is_clipped():
    evm = FloatEVM(FilterParams(alpha=200))
    evm.process(np.zeros((4, 4, 3), np.uint8))
    out = evm.process(np.full((4, 4, 3), 255, np.uint8))
    assert out.max() == 127
    i8 = evm.process_int8(np.zeros((4, 4, 3), np.uint8))
    assert i8.dtype == np.int8 and i8.min() >= -127


def test_set_params_keeps_state():
    evm = FloatEVM()
    evm.process(np.full((4, 4, 3), 10, np.uint8))
    evm.process(np.full((4, 4, 3), 30, np.uint8))
    state = evm.band_state().copy()
    evm.set_params(FilterParams(alpha=10))
    assert np.array_equal(evm.band_state(), state)


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        FloatEVM().process(np.zeros((4, 4, 3), np.float32))


# -- pyramid -------------------------------------------------------------------------
@pytest.mark.parametrize("w,levels", [(640, 2), (1280, 3), (320, 1), (160, 0)])
def test_pyramid_levels(w, levels):
    assert pyramid_levels_for(w) == levels


@pytest.mark.parametrize("h,w", [(480, 640), (720, 1280), (481, 641)])
def test_down_up_roundtrip_shape(h, w):
    frame = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    lv = pyramid_levels_for(w)
    small = downscale(frame, lv)
    if (h, w) == (480, 640):
        assert small.shape == (120, 160, 3)
    up = upscale(small.astype(np.float32), lv, (h, w))
    assert up.shape == (h, w, 3)


def test_add_signal_clips():
    frame = np.array([[[250, 5, 128]]], np.uint8)
    out = add_signal(frame, np.array([[[20.0, -20.0, 1.0]]], np.float32))
    assert out.tolist() == [[[255, 0, 129]]]


# -- end to end ----------------------------------------------------------------------
def synthetic_face_frames(n=300, h=480, w=640, f_pulse=1.2, amp=0.6, seed=0):
    """Grey background + skin patch whose green channel pulses at f_pulse."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (90, 110, 150), np.float32)
    y0, y1, x0, x1 = h // 4, 3 * h // 4, w // 3, 2 * w // 3
    for i in range(n):
        f = base.copy()
        f[y0:y1, x0:x1, 1] += amp * math.sin(2 * math.pi * f_pulse * i / FS)
        f += rng.normal(0, 0.8, f.shape)  # sensor noise
        yield np.clip(np.round(f), 0, 255).astype(np.uint8), (y0, y1, x0, x1)


def test_pulse_is_amplified_end_to_end():
    evm = FloatEVM(FilterParams(alpha=50))
    orig_g, mag_g = [], []
    for i, (frame, (y0, y1, x0, x1)) in enumerate(synthetic_face_frames()):
        lv = 2
        sig = evm.process(downscale(frame, lv))
        mag = add_signal(frame, upscale(sig, lv, frame.shape[:2]))
        if i >= 60:  # after settling
            orig_g.append(frame[y0:y1, x0:x1, 1].mean())
            mag_g.append(mag[y0:y1, x0:x1, 1].mean())
    orig_g, mag_g = np.array(orig_g), np.array(mag_g)
    gain = mag_g.std() / orig_g.std()
    # analytic gain at 1.2 Hz is ~0.35 -> 50 * 0.35 ~ 17x on top of the original
    assert gain > 10, gain
    # dominant frequency of the magnified trace is the pulse
    spec = np.abs(np.fft.rfft(mag_g - mag_g.mean()))
    freqs = np.fft.rfftfreq(len(mag_g), 1 / FS)
    assert freqs[spec.argmax()] == pytest.approx(1.2, abs=0.1)


def test_dashboard_headless_on_video_file(tmp_path, capsys):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FS, (640, 480))
    assert writer.isOpened()
    for frame, _ in synthetic_face_frames(n=90):
        writer.write(frame)
    writer.release()

    from evm_float import main

    assert main(["--source", str(path), "--headless", "60"]) == 0
    out = capsys.readouterr().out
    assert "[headless] frames=60" in out
