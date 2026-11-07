"""Tests for the bit-exact fixed-point model and the golden-vector generator."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import FilterParams, alpha_q88, coeff_q016  # noqa: E402
from evm_fixed import FixedEVM, pack_input, pack_output, unpack_input, unpack_output  # noqa: E402
from evm_float import FloatEVM  # noqa: E402


def scalar_reference(pixels: list[int], r_hi: int, r_lo: int, alpha: int, init_at=(0,)) -> list[int]:
    """Section 3 written out literally for ONE pixel/channel with Python ints."""
    def asr(v: int, s: int) -> int:  # arithmetic shift == floor division by 2^s
        return v // (1 << s)

    out, l_hi, l_lo = [], 0, 0
    for i, p in enumerate(pixels):
        x = p << 16
        if i in init_at:
            l_hi = l_lo = x
        l_hi += asr(r_hi * (x - l_hi), 16)
        l_lo += asr(r_lo * (x - l_lo), 16)
        band = l_hi - l_lo
        out.append(max(-127, min(127, asr(alpha * band, 24))))
        assert -(2**31) <= l_hi < 2**31 and -(2**31) <= l_lo < 2**31  # int32 state
    return out


def test_arithmetic_shift_semantics():
    assert (np.int64(-1) >> 16) == -1 and (np.int64(-65537) >> 16) == -2
    assert -1 // 65536 == -1


@pytest.mark.parametrize("regs", [(22434, 10459, 12800), (0xFFFF, 1, 0xFFFF), (1, 0xFFFF, 256), (0, 0, 0)])
def test_matches_scalar_reference_random(regs):
    r_hi, r_lo, alpha = regs
    rng = np.random.default_rng(sum(regs))
    frames = rng.integers(0, 256, (60, 4, 5, 3), dtype=np.uint8)
    evm = FixedEVM()
    evm.set_registers(*regs)
    got = np.stack([evm.process_int8(f, init=(i == 30)) for i, f in enumerate(frames)])
    for y, x, c in [(0, 0, 0), (1, 2, 1), (3, 4, 2), (2, 3, 0)]:
        ref = scalar_reference(frames[:, y, x, c].tolist(), r_hi, r_lo, alpha, init_at=(0, 30))
        assert got[:, y, x, c].tolist() == ref


def test_registers_from_params():
    evm = FixedEVM(FilterParams())
    assert evm.registers == {"r_hi": coeff_q016(2.0), "r_lo": coeff_q016(0.83), "alpha": alpha_q88(50)}
    # literal values for Jack's HLS defaults (fc 0.83-2.0 Hz, fs 30, alpha 50)
    assert evm.registers == {'r_hi': 22427, 'r_lo': 10457, 'alpha': 12800}


def test_init_frame_outputs_zero_and_state():
    evm = FixedEVM()
    f = np.full((2, 2, 3), 200, np.uint8)
    assert np.all(evm.process_int8(f) == 0)
    l_hi, l_lo = evm.state()
    assert l_hi.dtype == np.int32 and np.all(l_hi == 200 << 16) and np.all(l_lo == 200 << 16)
    assert np.any(evm.process_int8(np.zeros((2, 2, 3), np.uint8)) != 0)
    assert np.all(evm.process_int8(f, init=True) == 0)


def test_state_stays_in_range_under_full_swing():
    evm = FixedEVM()
    evm.set_registers(0xFFFF, 1, 0xFFFF)
    for i in range(200):
        out = evm.process_int8(np.full((1, 1, 3), 255 if i % 2 else 0, np.uint8))
        l_hi, l_lo = evm.state()
        assert 0 <= l_hi.min() and l_hi.max() <= 255 << 16
        assert 0 <= l_lo.min() and l_lo.max() <= 255 << 16
    assert out.min() >= -127 and out.max() <= 127
    assert np.abs(out).max() == 127


def test_close_to_float_reference():
    rng = np.random.default_rng(3)
    p = FilterParams(alpha=50)
    fx, fl = FixedEVM(p), FloatEVM(p)
    base = rng.integers(60, 200, (12, 16, 3))
    diffs = []
    for i in range(300):
        f = np.clip(base + 2 * math.sin(2 * math.pi * 1.2 * i / 30) + rng.normal(0, 1.5, base.shape), 0, 255).astype(np.uint8)
        a = fx.process_int8(f).astype(int)
        b = fl.process(f)
        diffs.append(np.abs(a - b).max())
    # coefficient rounding (Q0.16) + floor shifts: a few LSB at alpha 50
    assert max(diffs[30:]) <= 3, max(diffs[30:])


def test_pack_unpack_layout():
    f = np.array([[[1, 2, 3], [250, 251, 252]]], np.uint8)
    data = pack_input(f)
    assert data == bytes([1, 2, 3, 0, 250, 251, 252, 0])
    assert np.frombuffer(data, "<u4")[0] == 1 | (2 << 8) | (3 << 16)
    assert np.array_equal(unpack_input(data, 1, 2), f)
    o = np.array([[[-127, 0, 127], [-1, 5, -6]]], np.int8)
    od = pack_output(o)
    assert od == bytes([0x81, 0, 127, 0, 0xFF, 5, 0xFA, 0])
    assert np.array_equal(unpack_output(od, 1, 2), o)


def test_gen_vectors_roundtrip_and_verify(tmp_path, capsys):
    import gen_vectors
    from evm_eval import save_clip, synthetic_clip

    fr, _ = synthetic_clip(n=40, h=480, w=640)
    clip = tmp_path / "clip.npz"
    save_clip(fr, clip, 30.0)
    out = tmp_path / "tv"
    assert gen_vectors.main(["--clip", str(clip), "--n", "30", "--out", str(out)]) == 0
    params = json.loads((out / "params.json").read_text())
    n, h, w = params["N"], params["h"], params["w"]
    assert (n, h, w) == (30, 120, 160)
    assert (out / "input_frames.bin").stat().st_size == n * h * w * 4
    assert (out / "expected_output.bin").stat().st_size == n * h * w * 4
    assert (out / "expected_state.bin").stat().st_size == params["state_frames"] * 2 * h * w * 3 * 4
    for k in ("r_hi", "r_lo", "alpha", "N", "w", "h"):
        assert k in params

    # replay: feeding input_frames through a fresh model reproduces expected_output
    data = (out / "input_frames.bin").read_bytes()
    evm = FixedEVM()
    evm.set_registers(params["r_hi"], params["r_lo"], params["alpha"])
    fsz = h * w * 4
    replay = b"".join(pack_output(evm.process_int8(unpack_input(data[i * fsz:(i + 1) * fsz], h, w),
                                                   init=i in params["init_frames"])) for i in range(n))
    assert replay == (out / "expected_output.bin").read_bytes()

    # determinism
    first = (out / "expected_output.bin").read_bytes()
    gen_vectors.main(["--clip", str(clip), "--n", "30", "--out", str(out)])
    assert (out / "expected_output.bin").read_bytes() == first

    # verify: identical passes, one flipped value fails and is located
    assert gen_vectors.main(["verify", "--dir", str(out), "--got", str(out / "expected_output.bin")]) == 0
    bad = bytearray(first)
    bad[5 * fsz + 4 * 10 + 1] ^= 1   # frame 5, pixel 10, G
    (tmp_path / "bad.bin").write_bytes(bad)
    capsys.readouterr()
    assert gen_vectors.main(["verify", "--dir", str(out), "--got", str(tmp_path / "bad.bin")]) == 1
    assert "frame 5 y=0 x=10 ch=G" in capsys.readouterr().out

    # edge set exists, has a mid-stream INIT and exercises clipping in both signs
    ep = json.loads((out / "edge" / "params.json").read_text())
    assert ep["init_frames"] == [0, ep["N"] // 2]
    assert ep["stats"]["output_min"] == -127 and ep["stats"]["output_max"] == 127
