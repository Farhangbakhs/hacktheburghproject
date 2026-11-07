import math, sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evm_v1 import FixedV1, FloatV1, coeff_q016, alpha_q88, frames_to_words, words_to_frames

VEC = Path(__file__).resolve().parents[3] / "test_vectors" / "v1_contract"


def test_coefficients_analytic():
    assert coeff_q016(2.0) == round((1 - math.exp(-2 * math.pi * 2 / 30)) * 65536)
    assert coeff_q016(0.83) < coeff_q016(2.0) < 65536
    assert alpha_q88(50) == 12800 and alpha_q88(1) == 256


def test_init_frame_outputs_zero_and_constant_stays_zero():
    m = FixedV1()
    f = np.full((4, 5, 3), 117, np.uint8)
    for i in range(20):
        assert not m.process(f, init=(i == 0)).any()


def test_state_stays_in_pixel_range():
    rng = np.random.default_rng(1)
    m = FixedV1()
    for i in range(200):
        m.process(rng.integers(0, 256, (6, 7, 3), dtype=np.uint8), init=(i == 0))
        lh, ll = m.state_int32()
        assert lh.min() >= 0 and lh.max() <= 255 << 16 and ll.min() >= 0 and ll.max() <= 255 << 16


def test_floor_shift_on_negative():
    assert (np.int64(-1) >> 16) == -1  # arithmetic shift semantics the HLS kernel must replicate


def _sine(freq, n=300, amp=40.0):
    t = np.arange(n) / 30.0
    return np.clip(np.round(128 + amp * np.sin(2 * np.pi * freq * t)), 0, 255).astype(np.uint8)


def _out_amp(freq):
    m, mf = FixedV1(alpha=alpha_q88(5)), FloatV1(alpha=5)  # big signal, modest gain: quantisation noise stays small
    res, resf = [], []
    for i, v in enumerate(_sine(freq)):
        fr = np.full((1, 1, 3), v, np.uint8)
        res.append(int(m.process(fr, init=i == 0)[0, 0, 1])); resf.append(float(mf.process(fr, init=i == 0)[0, 0, 1]))
    r, rf = np.array(res[60:]), np.array(resf[60:])
    assert np.abs(r - np.floor(rf)).max() <= 2  # fixed vs float within 2 LSB
    return np.abs(r).max()


def test_frequency_response_is_bandpass():
    mid, low, high = _out_amp(1.2), _out_amp(0.05), _out_amp(10.0)
    assert mid > 3 * low and mid > 3 * high


def test_word_layout_roundtrip():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (3, 4, 5, 3), dtype=np.uint8)
    w = frames_to_words(a)
    assert w.dtype == np.uint32 and (w >> 24).max() == 0
    assert np.array_equal(words_to_frames(w), a)
    s = rng.integers(-127, 128, (2, 4, 5, 3), dtype=np.int8)
    assert np.array_equal(words_to_frames(frames_to_words(s), signed=True), s)


@pytest.mark.skipif(not VEC.exists(), reason="golden vectors not present")
@pytest.mark.parametrize("sub", ["", "edge"])
def test_matches_person2_golden_vectors(sub):
    from verify_vectors import check
    assert check(VEC / sub if sub else VEC)
