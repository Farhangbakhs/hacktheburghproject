"""Tests for the EVM evaluation harness (evm_eval.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evm_eval as ev  # noqa: E402
from evm_common import FilterParams  # noqa: E402
from evm_float import FloatEVM  # noqa: E402

FS = 30.0
SMALL = dict(n=240, h=240, w=320, fs=FS)  # 8 s, quick


def contract(p: FilterParams) -> FloatEVM:
    return FloatEVM(p)


class ZeroEVM:
    """Adds nothing: output == input."""

    name = "zero"

    def __init__(self, params: FilterParams):
        self.params = params

    def reset(self) -> None:
        pass

    def set_params(self, params: FilterParams) -> None:
        self.params = params

    def band_state(self) -> None:
        return None

    def process(self, small: np.ndarray) -> np.ndarray:
        return np.zeros(small.shape, np.float32)


@pytest.fixture(scope="module")
def clean():
    return ev.synthetic_clip(noise_std=1.0, **SMALL)


@pytest.fixture(scope="module")
def clean_metrics(clean):
    frames, truth = clean
    return ev.evaluate(contract, frames, FS, truth=truth)


# -- synthetic clip -------------------------------------------------------------
def test_synthetic_clip_properties(clean):
    frames, truth = clean
    assert len(frames) == SMALL["n"]
    assert all(f.shape == (240, 320, 3) and f.dtype == np.uint8 for f in frames)
    mask = truth["skin_mask"]
    assert mask.dtype == bool and mask.shape == (240, 320)
    assert 0.05 < mask.mean() < 0.4
    assert truth["pulse_hz"] == 1.2 and truth["fs"] == FS
    assert len(truth["pulse_wave"]) == SMALL["n"]

    stack = np.stack(frames).astype(np.float64)
    skin_means = stack[:, mask, :].mean(axis=1)  # [n, 3] BGR
    ac = skin_means - skin_means.mean(axis=0)
    # Pulse is at 1.2 Hz and strongest in green, weaker in blue and red.
    spec = np.abs(np.fft.rfft(ac[:, 1]))
    freqs = np.fft.rfftfreq(len(ac), 1 / FS)
    assert abs(freqs[np.argmax(spec[1:]) + 1] - 1.2) < 0.15
    amp = ac.std(axis=0)
    assert amp[1] > amp[0] and amp[1] > amp[2]
    assert 0.25 < amp[1] < 0.5  # ~0.5 * shading / sqrt(2)
    # Skin is skin-coloured (R > G > B).
    b, g, r = stack[0][mask].mean(axis=0)
    assert r > g > b


def test_synthetic_deterministic_and_options():
    a, _ = ev.synthetic_clip(n=5, h=60, w=80, seed=3)
    b, _ = ev.synthetic_clip(n=5, h=60, w=80, seed=3)
    c, _ = ev.synthetic_clip(n=5, h=60, w=80, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])

    frames, truth = ev.synthetic_clip(n=10, h=60, w=80, motion_px=4.0, motion_hz=1.5)
    assert callable(truth["skin_mask"])
    assert not np.array_equal(truth["skin_mask"](0), truth["skin_mask"](3))

    dim, _ = ev.synthetic_clip(n=2, h=60, w=80, brightness=0.3, noise_std=0)
    full, _ = ev.synthetic_clip(n=2, h=60, w=80, noise_std=0)
    assert dim[0].mean() == pytest.approx(0.3 * full[0].mean(), rel=0.05)

    jp, _ = ev.synthetic_clip(n=2, h=60, w=80, jpeg_quality=50)
    assert jp[0].shape == (60, 80, 3) and jp[0].dtype == np.uint8

    dr, _ = ev.synthetic_clip(n=60, h=60, w=80, drift=0.2, noise_std=0)
    assert dr[-1].mean() > dr[0].mean() * 1.1


def test_scenarios_exist():
    assert set(ev.SCENARIOS) == {"clean", "noisy", "motion", "drift", "jpeg", "dim"}
    with pytest.raises(KeyError):
        ev.scenario_clip("nope")


# -- metrics --------------------------------------------------------------------
def test_contract_on_clean_clip(clean_metrics):
    m = clean_metrics
    assert m["pulse_snr_db"] > 15
    assert m["pulse_fidelity"] > 0.8
    assert m["pulse_gain"] > 5
    assert abs(m["f0_est_hz"] - 1.2) < 0.1
    assert m["pulse_amp_out"] > 2
    assert m["artifact_ratio"] < 1.5
    assert m["fps_proc"] > 0 and m["fps_pipeline"] > 0
    assert np.isfinite(m["quality_score"])


def test_zero_processor_has_unit_gain(clean):
    frames, truth = clean
    m = ev.evaluate(lambda p: ZeroEVM(p), frames, FS, truth=truth)
    assert m["pulse_gain"] == pytest.approx(1.0, abs=1e-6)
    assert m["pulse_amp_out"] == pytest.approx(0.0, abs=1e-6)
    assert m["background_noise"] == 0.0 and m["skin_speckle"] == 0.0
    assert m["bg_temporal_std"] == 0.0


def test_noise_lowers_snr(clean_metrics):
    frames, truth = ev.synthetic_clip(noise_std=20.0, **SMALL)
    m = ev.evaluate(contract, frames, FS, truth=truth)
    assert m["pulse_snr_db"] < clean_metrics["pulse_snr_db"] - 3
    assert m["skin_speckle"] > clean_metrics["skin_speckle"]
    assert m["quality_score"] < clean_metrics["quality_score"]


def test_motion_raises_artifact_ratio(clean_metrics):
    frames, truth = ev.synthetic_clip(noise_std=1.0, motion_px=4.0, motion_hz=0.5, **SMALL)
    m = ev.evaluate(contract, frames, FS, truth=truth)
    assert m["artifact_ratio"] > 2 * clean_metrics["artifact_ratio"]
    assert m["quality_score"] < clean_metrics["quality_score"]


def test_truth_mask_suppresses_background(clean, clean_metrics):
    frames, truth = clean
    m = ev.evaluate(contract, frames, FS, truth=truth, mask_fn=ev.truth_mask_fn(truth))
    assert m["background_noise"] < 0.2 * clean_metrics["background_noise"]
    assert m["pulse_fidelity"] > 0.8


def test_evaluate_without_truth_uses_fallback_region(clean):
    frames, _ = clean
    m = ev.evaluate(contract, frames, FS)
    assert np.isnan(m["pulse_fidelity"])
    assert abs(m["f0_est_hz"] - 1.2) < 0.15


def test_evaluate_rejects_short_clip():
    frames, _ = ev.synthetic_clip(n=70, h=60, w=80)
    with pytest.raises(ValueError):
        ev.evaluate(contract, frames, FS)


# -- clip I/O and record ----------------------------------------------------------
def _write_video(path: Path, frames: list[np.ndarray], fs: float) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fs, (w, h))
    assert vw.isOpened()
    for f in frames:
        vw.write(f)
    vw.release()


def test_save_load_roundtrip(tmp_path):
    frames, _ = ev.synthetic_clip(n=8, h=48, w=64)
    path = ev.save_clip(frames, tmp_path / "clip", 25.0)
    assert path.suffix == ".npz"
    back, fs = ev.load_clip(path)
    assert fs == 25.0 and len(back) == 8
    assert all(np.array_equal(a, b) for a, b in zip(frames, back))
    assert len(ev.load_clip(path, max_frames=3)[0]) == 3


def test_load_video_file(tmp_path):
    frames, _ = ev.synthetic_clip(n=12, h=48, w=64)
    vid = tmp_path / "v.avi"
    _write_video(vid, frames, 30.0)
    back, fs = ev.load_clip(vid)
    assert len(back) == 12 and fs == pytest.approx(30.0)
    assert back[0].shape == (48, 64, 3)
    with pytest.raises(FileNotFoundError):
        ev.load_clip(tmp_path / "missing.npz")


def test_record_from_video_file(tmp_path, capsys):
    frames, _ = ev.synthetic_clip(n=45, h=48, w=64)
    vid = tmp_path / "src.avi"
    _write_video(vid, frames, 30.0)
    rc = ev.main(["record", "--source", str(vid), "--seconds", "1", "--name", "rec",
                  "--out-dir", str(tmp_path / "recs")])
    assert rc == 0
    back, fs = ev.load_clip(tmp_path / "recs" / "rec.npz")
    assert len(back) == 30 and fs == pytest.approx(30.0)
    assert back[0].shape == (48, 64, 3)
    with np.load(tmp_path / "recs" / "rec.npz") as z:
        assert z["timestamps"][0] == 0.0 and len(z["timestamps"]) == 30
    assert "saved 30 frames" in capsys.readouterr().out


# -- CLI --------------------------------------------------------------------------
def test_registry_has_contract():
    reg = ev.variant_registry()
    assert "contract" in reg
    assert isinstance(reg["contract"](FilterParams()), FloatEVM)


def test_compare_cli_synthetic(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = ev.main(["compare", "--synthetic", "clean", "--variants", "contract", "--seconds", "5",
                  "--size", "160x120", "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "contract" in text and "snr_dB" in text and "quality" in text
    res = json.loads(out.read_text())
    assert res["clean"]["contract"]["pulse_snr_db"] > 10


def test_compare_cli_clip_and_masks(tmp_path, capsys):
    frames, _ = ev.synthetic_clip(n=120, h=120, w=160)
    clip = ev.save_clip(frames, tmp_path / "c.npz", 30.0)
    assert ev.main(["compare", "--clip", str(clip), "--variants", "contract", "--mask", "face"]) == 0
    assert ev.main(["compare", "--synthetic", "motion", "--variants", "contract", "--seconds", "4",
                    "--size", "160x120", "--mask", "truth"]) == 0
    assert "contract" in capsys.readouterr().out


def test_compare_cli_unknown_variant(capsys):
    assert ev.main(["compare", "--variants", "nope", "--seconds", "4", "--size", "160x120"]) == 2
