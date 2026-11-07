"""Tests for evm_enhanced.py (experimental EVM variants).

Includes a small local synthetic-clip helper (skin patch pulsing at 1.2 Hz,
mostly in green, Gaussian noise) and per-pixel lock-in metrics, independent of
evm_eval.py. Run `python tests/test_evm_enhanced.py` to print the preset table.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import FilterParams  # noqa: E402
from evm_enhanced import (  # noqa: E402
    PRESETS,
    EnhancedConfig,
    EnhancedEVM,
    _BGR_TO,
    _make_filter,
    filter_sos,
    peak_gain,
)
from evm_float import FloatEVM  # noqa: E402

FS = 30.0
H, W = 120, 160
PULSE_HZ = 1.2
PULSE_BGR = np.array([0.15, 0.5, 0.25], np.float32)  # amplitude per channel (grey levels)


# --------------------------------------------------------------------------------
# Synthetic clip + metrics (local helper)
# --------------------------------------------------------------------------------
class SyntheticClip:
    """Static textured background + skin ellipse pulsing at PULSE_HZ.

    motion=(frame, dx): from `frame` on the ellipse is shifted by dx pixels
    (a head jump). Frames are generated on the fly (uint8 BGR).
    """

    def __init__(self, n=600, noise=2.0, amp=1.0, motion=None, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.noise, self.amp, self.motion = n, noise, amp, motion
        self.rng = np.random.default_rng(seed + 1)
        tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W, 3)).astype(np.float32), (0, 0), 3)
        bg = 90 + 25 * tex / tex.std()
        cv2.rectangle(bg, (5, 5), (40, 110), (60, 70, 80), -1)  # strong edges
        cv2.rectangle(bg, (125, 20), (155, 60), (200, 190, 180), -1)
        self.bg = bg
        self.skin_tex = 6 * tex / tex.std()
        self.skin_colour = np.array([110, 140, 190], np.float32)

    def mask(self, dx=0) -> np.ndarray:
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (80 + dx, 60), (32, 42), 0, 0, 360, 1, -1)
        return m.astype(bool)

    def pulse(self, i: int) -> float:
        return math.sin(2 * math.pi * PULSE_HZ * i / FS)

    def frame(self, i: int) -> np.ndarray:
        dx = self.motion[1] if self.motion and i >= self.motion[0] else 0
        m = self.mask(dx)
        img = self.bg.copy()
        skin = self.skin_colour + np.roll(self.skin_tex, dx, axis=1)
        skin = skin + self.amp * PULSE_BGR * self.pulse(i)
        img[m] = skin[m]
        img += self.rng.normal(0, self.noise, img.shape).astype(np.float32)
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def evaluate(proc, clip: SyntheticClip, settle: int = 90) -> dict:
    """Per-pixel lock-in at PULSE_HZ on the processor output.

    snr_db:     skin pulse power / skin residual power (per pixel, all channels)
    speckle_db: background output power / skin pulse power (lower = cleaner)
    ms:         mean processing time per frame
    """
    acc = {k: np.zeros((H, W, 3), np.float64) for k in ("s", "c", "m", "q")}
    n, t_tot = 0, 0.0
    for i in range(clip.n):
        f = clip.frame(i)
        t0 = time.perf_counter()
        out = proc.process(f)
        t_tot += time.perf_counter() - t0
        if i < settle:
            continue
        ph = 2 * math.pi * PULSE_HZ * i / FS
        o = out.astype(np.float64)
        acc["s"] += o * math.sin(ph)
        acc["c"] += o * math.cos(ph)
        acc["m"] += o
        acc["q"] += o * o
        n += 1
    a, b = 2 * acc["s"] / n, 2 * acc["c"] / n
    sig = (a * a + b * b) / 2
    var = acc["q"] / n - (acc["m"] / n) ** 2
    res = np.maximum(var - sig, 1e-12)
    skin = cv2.erode(clip.mask().astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    bgm = ~cv2.dilate(clip.mask().astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    s_skin = sig[skin].sum(axis=-1).mean()
    r_skin = res[skin].sum(axis=-1).mean()
    p_bg = var[bgm].sum(axis=-1).mean()
    return {
        "snr_db": 10 * math.log10(s_skin / r_skin),
        "speckle_db": 10 * math.log10(p_bg / s_skin),
        "pulse_amp": math.sqrt(2 * s_skin),
        "ms": 1000 * t_tot / clip.n,
    }


def halo(proc, jump=300, dx=4, n=400) -> float:
    """Output energy in the 1 s after a head jump relative to the static pulse
    energy in the second before it (dB, lower = fewer ghosts)."""
    clip = SyntheticClip(n=n, motion=(jump, dx))
    before, after = [], []
    for i in range(n):
        out = proc.process(clip.frame(i))
        e = float(np.mean(out.astype(np.float64) ** 2))
        if jump - 30 <= i < jump:
            before.append(e)
        elif jump <= i < jump + 30:
            after.append(e)
    return 10 * math.log10(np.mean(after) / np.mean(before))


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def p_unit() -> FilterParams:
    return FilterParams(alpha=1.0)


def measured_gain(kind: str, p: FilterParams, f: float, n: int = 1200) -> float:
    """Steady-state amplitude response of the internal filter to a float sine."""
    filt = _make_filter(kind, p)
    shape = (2, 2, 3)
    filt.init(np.full(shape, 128.0, np.float32))
    outs = []
    for i in range(n):
        x = np.full(shape, 128 + 2.0 * math.sin(2 * math.pi * f * i / p.fs), np.float32)
        outs.append(float(filt.step(x)[0, 0, 1]))
    # least-squares sinusoid fit on the steady-state tail (robust to sampling phase)
    i = np.arange(n // 2, n)
    ph = 2 * math.pi * f * i / p.fs
    basis = np.stack([np.sin(ph), np.cos(ph), np.ones_like(ph)], 1)
    coef, *_ = np.linalg.lstsq(basis, np.array(outs[n // 2:]), rcond=None)
    return math.hypot(coef[0], coef[1]) / 2.0


def random_frames(n=60, seed=0, shape=(H, W, 3)):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 216, shape).astype(np.float32)
    return [np.clip(base + rng.normal(0, 3, shape), 0, 255).astype(np.uint8) for _ in range(n)]


# --------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------
def test_contract_preset_equals_float_bit_for_bit():
    p = FilterParams()
    a, b = PRESETS["contract"](FilterParams()), FloatEVM(p)
    for f in random_frames(80):
        np.testing.assert_array_equal(a.process(f), b.process(f))
    np.testing.assert_array_equal(a.band_state(), b.band_state())
    assert a.frames_since_init == b.frames_since_init
    # also after a live param change (state kept)
    p2 = FilterParams(fc_lo=0.7, fc_hi=2.5, alpha=80)
    a.set_params(p2)
    b.set_params(FilterParams(fc_lo=0.7, fc_hi=2.5, alpha=80))
    for f in random_frames(20, seed=3):
        np.testing.assert_array_equal(a.process(f), b.process(f))


@pytest.mark.parametrize("kind", ["iir1", "iir1x2", "butter2"])
def test_frequency_response_matches_sosfreqz(kind):
    p = p_unit()
    sos = filter_sos(kind, p)
    for f in (0.5, 0.83, 1.2, 2.0, 3.0, 6.0):
        _, h = sps.sosfreqz(sos, worN=[f], fs=p.fs)
        want = abs(h[0])
        got = measured_gain(kind, p, f)
        assert got == pytest.approx(want, rel=0.03, abs=2e-3), (kind, f, got, want)


def test_iir1_sos_matches_contract_analytic():
    p = p_unit()
    for f in (0.3, 1.0, 1.3, 5.0):
        z = np.exp(1j * 2 * math.pi * f / p.fs)
        ana = abs(p.r_hi / (1 - (1 - p.r_hi) / z) - p.r_lo / (1 - (1 - p.r_lo) / z))
        _, h = sps.sosfreqz(filter_sos("iir1", p), worN=[f], fs=p.fs)
        assert abs(h[0]) == pytest.approx(ana, rel=1e-9)
    g, fpk = peak_gain("iir1", p)
    assert g == pytest.approx(0.41, abs=0.01) and 1.1 < fpk < 1.5


def test_sharper_filters_reject_flicker_better():
    p = p_unit()
    rel = {}
    for k in ("iir1", "iir1x2", "butter2"):
        f, h = sps.sosfreqz(filter_sos(k, p), worN=[8.0], fs=p.fs)
        rel[k] = abs(h[0]) / peak_gain(k, p)[0]
    assert rel["iir1x2"] < rel["iir1"] / 2 and rel["butter2"] < rel["iir1"] / 2


@pytest.mark.parametrize("space", ["yiq", "ycrcb"])
def test_colour_space_identity_round_trip(space):
    """attenuation = 1 everywhere -> identical to the BGR result."""
    frames = random_frames(40)
    ref = EnhancedEVM(FilterParams(), EnhancedConfig())
    col = EnhancedEVM(FilterParams(), EnhancedConfig(color_space=space, chroma_attenuation=1.0))
    for f in frames:
        np.testing.assert_allclose(col.process(f), ref.process(f), atol=1e-3)


def test_colour_matrix_equals_filtering_in_yiq():
    """M^-1 D M on the BGR band == filter in YIQ, attenuate chroma, convert back."""
    p = FilterParams(alpha=10)
    cfg = EnhancedConfig(color_space="yiq", chroma_attenuation=0.3, luma_gain=0.8,
                         stage_clip=1e6, clip=1e6)
    evm = EnhancedEVM(p, cfg)
    m = _BGR_TO["yiq"]
    lhi = llo = None
    for f in random_frames(40):
        out = evm.process(f)
        yiq = (f.astype(np.float64) @ m.T)
        if lhi is None:
            lhi, llo = yiq.copy(), yiq.copy()
        lhi += p.r_hi * (yiq - lhi)
        llo += p.r_lo * (yiq - llo)
        band = (lhi - llo) * np.array([0.8, 0.3, 0.3])
        want = p.alpha * band @ np.linalg.inv(m).T
        np.testing.assert_allclose(out, want, atol=2e-3)


def test_green_only():
    evm = EnhancedEVM(FilterParams(), EnhancedConfig(channels="green"))
    for f in random_frames(20):
        out = evm.process(f)
    assert np.all(out[..., [0, 2]] == 0) and np.abs(out[..., 1]).max() > 0


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_post_blur_equals_pre_pooling(levels):
    """Pooling before the (linear) filter == blurring the output (no clipping)."""
    big = dict(stage_clip=1e6, clip=1e6)
    pre = EnhancedEVM(FilterParams(), EnhancedConfig(extra_levels=levels, **big))
    post = EnhancedEVM(FilterParams(), EnhancedConfig(post_blur_levels=levels, **big))
    for f in random_frames(40):
        a, b = pre.process(f), post.process(f)
        np.testing.assert_allclose(a, b, atol=2e-3 * max(1.0, np.abs(a).max()))
    assert pre.band_state().shape == (H, W, 3)


def test_post_blur_on_int8_output_close_to_pre_pooling():
    """Realistic contract path: FPGA int8 (floored, clipped) then laptop blur."""
    pre = EnhancedEVM(FilterParams(), EnhancedConfig(extra_levels=2))
    fl = FloatEVM(FilterParams())
    clip = SyntheticClip(n=150)
    errs, mags = [], []
    for i in range(clip.n):
        f = clip.frame(i)
        a = pre.process(f)
        q = np.floor(fl.process(f)).astype(np.int8).astype(np.float32) + 0.5  # undo floor bias
        b = cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(q))))
        if i > 60:
            errs.append(np.abs(a - b).mean())
            mags.append(np.abs(a).mean())
    assert np.mean(errs) < 0.1 * np.mean(mags) + 0.3


def sine_gain_end_to_end(evm, f, amp=20.0, n=900) -> float:
    """Output amplitude / input amplitude for uniform uint8 frames at f."""
    outs = []
    for i in range(n):
        v = 128 + amp * math.sin(2 * math.pi * f * i / FS)
        outs.append(float(evm.process(np.full((8, 8, 3), round(v), np.uint8))[4, 4, 1]))
    i = np.arange(n // 2, n)
    ph = 2 * math.pi * f * i / FS
    basis = np.stack([np.sin(ph), np.cos(ph), np.ones_like(ph)], 1)
    coef, *_ = np.linalg.lstsq(basis, np.array(outs[n // 2:]), rcond=None)
    return math.hypot(coef[0], coef[1]) / amp


NORM_CASES = {
    "iir1": dict(),
    "iir1x2": dict(filter="iir1x2"),
    "butter2": dict(filter="butter2"),
    "iir1+post_lp": dict(post_temporal="lp"),
    "iir1+post_butter": dict(post_temporal="butter"),
    "split_hw25_x3+post_butter": dict(hw_alpha=25.0, post_gain=3.0, post_temporal="butter"),
}


@pytest.mark.parametrize("case", list(NORM_CASES))
def test_gain_normalise_peak_equals_alpha(case):
    alpha = 2.0
    cfg = EnhancedConfig(gain_normalise=True, stage_clip=1e6, clip=1e6, **NORM_CASES[case])
    evm = EnhancedEVM(FilterParams(alpha=alpha), cfg)
    want = alpha * cfg.post_gain
    assert evm.total_peak_gain == pytest.approx(want, rel=1e-9)
    got = sine_gain_end_to_end(evm, evm._peak_f)
    assert got == pytest.approx(want, rel=0.03), (case, got, want)


def test_post_temporal_cascade_matches_sosfreqz():
    from evm_enhanced import cascade_sos
    for pt in ("lp", "butter"):
        cfg = EnhancedConfig(post_temporal=pt, stage_clip=1e6, clip=1e6)
        sos = cascade_sos(cfg, FilterParams())
        for f in (0.6, 1.2, 3.0, 6.0):
            evm = EnhancedEVM(FilterParams(alpha=1.0), cfg)
            _, h = sps.sosfreqz(sos, worN=[f], fs=FS)
            assert sine_gain_end_to_end(evm, f) == pytest.approx(abs(h[0]), rel=0.03, abs=3e-3), (pt, f)


def test_post_temporal_raises_in_band_noise_fraction():
    """Research note: in-band (0.83-2 Hz) white-noise fraction 26% -> ~43% (lp 2.5)
    -> ~80% (Butterworth 0.75-2.2)."""
    from evm_enhanced import cascade_sos
    frac = {}
    for pt in (None, "lp", "butter"):
        f, h = sps.sosfreqz(cascade_sos(EnhancedConfig(post_temporal=pt), FilterParams()), 8192, fs=FS)
        pw = np.abs(h) ** 2
        frac[pt] = pw[(f >= 0.83) & (f <= 2.0)].sum() / pw.sum()
    assert frac[None] == pytest.approx(0.26, abs=0.03)
    assert frac["lp"] == pytest.approx(0.43, abs=0.04)
    assert frac["butter"] > 0.7


def test_split_gain_register_and_less_stage_clipping():
    p = FilterParams(alpha=100)
    evm = EnhancedEVM(p, EnhancedConfig(hw_alpha=25.0))
    assert evm.alpha_hw == 25.0 and evm._post_mult == pytest.approx(4.0)
    # same total gain, but the FPGA-position output saturates far less on a jump
    def clipped_fraction(cfg):
        e = EnhancedEVM(FilterParams(alpha=100), cfg)
        clip = SyntheticClip(n=320, motion=(300, 6))
        n_sat = 0
        for i in range(clip.n):
            f = clip.frame(i)
            e.process(f)
            if i >= 300:
                band = e.band_state()
                n_sat += int(np.sum(np.abs(e.alpha_hw * band) >= 127))
        return n_sat
    assert clipped_fraction(EnhancedConfig(hw_alpha=25.0)) < 0.5 * clipped_fraction(EnhancedConfig())
    with pytest.raises(ValueError):
        EnhancedEVM(p, EnhancedConfig(hw_alpha=300))


def test_soft_clip_bounds_and_small_signal_linear():
    cfg = EnhancedConfig(soft_clip=10.0, stage_clip=1e6, clip=1e6)
    evm = EnhancedEVM(FilterParams(alpha=500), cfg)
    out = None
    for f in random_frames(30):
        out = evm.process(f)
    assert np.abs(out).max() < 10.0 + 1e-4
    t = np.float32(10) * np.tanh(np.float32(0.5) / np.float32(10))
    assert t == pytest.approx(0.5, rel=0.01)


def test_pulse_dir_render():
    from evm_enhanced import color_matrix
    t = color_matrix(EnhancedConfig(render="pulse_dir"))
    assert np.linalg.matrix_rank(t) == 1
    beat = -PULSE_BGR  # blood arrival darkens, mostly green
    delta = t @ beat
    assert delta[2] > 0 and delta[2] > abs(delta[0]) and delta[2] > abs(delta[1])  # redder
    assert np.allclose(t @ np.ones(3), 0)  # pure luma/illumination flicker discarded
    evm = EnhancedEVM(FilterParams(), EnhancedConfig(render="pulse_dir", post_blur_levels=2))
    assert evm.config.color_space == "pulse_dir"
    for f in random_frames(10):
        assert evm.process(f).shape == (H, W, 3)


def energy_after_jump(cfg, dx=6):
    clip = SyntheticClip(n=360, motion=(300, dx))
    evm = EnhancedEVM(FilterParams(), cfg)
    e = []
    for i in range(clip.n):
        out = evm.process(clip.frame(i))
        if 300 <= i < 330:
            e.append(np.mean(out ** 2))
    return float(np.mean(e))


def test_motion_gate_reduces_output_during_jump():
    e0 = energy_after_jump(EnhancedConfig())
    e1 = energy_after_jump(EnhancedConfig(motion_gate=True, motion_threshold=0.5, soft_reset=False))
    e2 = energy_after_jump(EnhancedConfig(motion_gate=True, motion_threshold=0.5, soft_reset=True))
    assert e1 < 0.7 * e0
    assert e2 < 0.3 * e0


def test_motion_gate_no_false_trigger_on_static_noise():
    evm = EnhancedEVM(FilterParams(), EnhancedConfig(motion_gate=True))
    clip = SyntheticClip(n=120)
    for i in range(clip.n):
        evm.process(clip.frame(i))
        assert evm.gate == 1.0, (i, evm.last_motion)


def test_motion_gate_ramp_and_external_hint():
    cfg = EnhancedConfig(motion_gate=True, motion_threshold=1.0)
    evm = EnhancedEVM(FilterParams(), cfg)
    f = SyntheticClip(n=1).frame(0)
    evm.process(f)
    for _ in range(3):
        evm.set_motion(10.0)
        evm.process(f)
        assert evm.gate == 0.0
    gates = []
    for _ in range(40):
        evm.process(f)
        gates.append(evm.gate)
    hold, ramp = cfg.motion_hold_frames, cfg.motion_ramp_frames
    assert gates[hold - 1] == 0.0
    assert gates[hold + ramp // 2 - 1] == pytest.approx(0.5, abs=0.06)
    assert gates[hold + ramp] == 1.0


@pytest.mark.parametrize("name", list(PRESETS))
def test_constant_input_gives_zero(name):
    evm = PRESETS[name](FilterParams())
    f = np.full((H, W, 3), 117, np.uint8)
    for _ in range(30):
        out = evm.process(f)
        assert out.dtype == np.float32 and out.shape == (H, W, 3)
        assert np.abs(out).max() < 1e-3
    assert np.abs(evm.band_state()).max() < 1e-4


@pytest.mark.parametrize("name", list(PRESETS))
def test_reset_and_shape_change(name):
    evm = PRESETS[name](FilterParams())
    assert evm.band_state() is None
    for f in random_frames(5):
        evm.process(f)
    assert evm.frames_since_init == 5
    evm.reset()
    evm.process(random_frames(1, seed=9)[0])
    assert evm.frames_since_init == 1
    out = evm.process(random_frames(1, seed=4, shape=(60, 80, 3))[0])
    assert out.shape == (60, 80, 3) and evm.frames_since_init == 1
    assert evm.band_state().shape == (60, 80, 3)
    assert np.abs(out).max() <= evm.config.clip + 1e-4


def test_rejects_bad_input_and_config():
    evm = EnhancedEVM()
    with pytest.raises(ValueError):
        evm.process(np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError):
        EnhancedEVM(config=EnhancedConfig(filter="ideal"))
    with pytest.raises(ValueError):
        EnhancedEVM(config=EnhancedConfig(extra_levels=5))


def test_contract_safety_flags():
    safe = {k: PRESETS[k].config.contract_safe for k in PRESETS}
    assert safe["contract"] and safe["pooled"] and safe["pooled_yiq"] and safe["best_safe"]
    assert not safe["butter_pooled"] and not safe["best"]


def test_presets_improve_synthetic_quality():
    base = evaluate(PRESETS["contract"](FilterParams()), SyntheticClip(n=360))
    for name in ("pooled", "best_safe", "best"):
        r = evaluate(PRESETS[name](FilterParams()), SyntheticClip(n=360))
        assert r["snr_db"] > base["snr_db"] + 6, (name, r, base)
        assert r["speckle_db"] < base["speckle_db"] - 6, (name, r, base)


@pytest.mark.parametrize("name", list(PRESETS))
def test_speed_under_5ms(name):
    evm = PRESETS[name](FilterParams())
    frames = random_frames(10)
    for f in frames:
        evm.process(f)
    t0 = time.perf_counter()
    reps = 100
    for i in range(reps):
        evm.process(frames[i % 10])
    ms = 1000 * (time.perf_counter() - t0) / reps
    assert ms < 5.0, (name, ms)


# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'preset':14s} {'safe':5s} {'SNRdB':>7s} {'speckdB':>8s} {'haloDB':>7s} {'amp':>6s} {'ms':>5s}  name")
    for name, make in PRESETS.items():
        r = evaluate(make(FilterParams()), SyntheticClip(n=600))
        hl = halo(make(FilterParams()))
        print(f"{name:14s} {str(make.config.contract_safe):5s} {r['snr_db']:7.1f} {r['speckle_db']:8.1f} "
              f"{hl:7.1f} {r['pulse_amp']:6.1f} {r['ms']:5.2f}  {make(FilterParams()).name}")
