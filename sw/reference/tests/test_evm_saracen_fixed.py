"""Tests for the proposed section 3 v2 (saracen Laplacian) fixed-point split and its vectors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evm_common import FilterParams  # noqa: E402
from evm_saracen_fixed import (Registers, SaracenFixedEVM, SaracenHW, SaracenKernel, laptop_pre,  # noqa: E402
                               level_alphas, pack_words, registers_for, unpack_words)


def scalar_ref(vals, r_hi, r_lo, alpha, init_at=(0,)):
    """v2 kernel spec for ONE value stream, plain Python ints (// is floor = arithmetic shift)."""
    out, lh, ll = [], 0, 0
    for i, v in enumerate(vals):
        x = v << 8
        if i in init_at:
            lh = ll = x
        lh += (r_hi * (x - lh)) // 65536
        ll += (r_lo * (x - ll)) // 65536
        assert -(2**31) <= lh < 2**31 and -(2**31) <= ll < 2**31
        out.append(max(-32767, min(32767, (alpha * (lh - ll)) // 65536)))
    return out


def small_regs(r_hi=26223, r_lo=3343, alphas=(3968, 5120, 5120, 0), counts=(6, 4, 2, 0)):
    return Registers(r_hi, r_lo, list(alphas), list(counts), [(2, 3), (2, 2), (1, 2)])


@pytest.mark.parametrize("regs", [small_regs(), small_regs(0xFFFF, 1, (0xFFFF, 0x8000, 1, 0)),
                                  small_regs(0, 0, (0, 0, 0, 0))])
def test_kernel_matches_scalar_spec(regs):
    rng = np.random.default_rng(regs.r_hi)
    seq = rng.integers(-32767, 32768, (80, regs.total)).astype(np.int16)
    k = SaracenKernel(regs)
    got = np.stack([k.process(w, init=i in (0, 40)) for i, w in enumerate(seq)])
    lvl_of = np.repeat(np.arange(4), regs.count)
    for j in range(regs.total):
        ref = scalar_ref(seq[:, j].tolist(), regs.r_hi, regs.r_lo, regs.alpha[lvl_of[j]], (0, 40))
        assert got[:, j].tolist() == ref


def test_state_bounded_and_clipping():
    regs = small_regs(0xFFFF, 1, (0xFFFF, 0xFFFF, 0xFFFF, 0))
    k = SaracenKernel(regs)
    for i in range(100):
        out = k.process(np.full(regs.total, 32767 if i % 2 else -32767, np.int16))
        lh, ll = k.state()
        assert np.abs(lh).max() <= 32767 << 8 and np.abs(ll).max() <= 32767 << 8
    assert out.max() == 32767 or out.min() == -32767


def test_word_packing():
    v = np.array([-1, 0, 32767, -32767], np.int16)
    b = pack_words(v)
    assert b == bytes([0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0xFF, 0x7F, 0, 0, 0x01, 0x80, 0, 0])
    assert np.array_equal(unpack_words(b), v)
    with pytest.raises(ValueError):
        unpack_words(bytes([0, 0, 1, 0]))


def test_registers_and_alphas_default():
    regs = registers_for(FilterParams(fc_lo=0.25, fc_hi=2.44, alpha=20), SaracenHW(), (240, 320))
    assert regs.shapes == [(120, 160), (60, 80), (30, 40)]
    assert regs.count == [19200, 4800, 1200, 0] and regs.total == 25200
    assert regs.alpha == [3968, 5120, 5120, 0]           # 15.5, 20, 20 in Q8.8
    assert (regs.r_hi, regs.r_lo) == (26223, 3343)
    # BRAM for state: total * 2 * int32
    assert regs.total * 2 * 4 == 201600
    a = level_alphas(20, SaracenHW(), (240, 320))
    assert a[0] < a[1] == a[2] == 20


def test_laptop_pre_shapes():
    f = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    work, lab, words = laptop_pre(f, SaracenHW())
    assert work.shape == (240, 320, 3) and lab.shape == (240, 320, 3)
    assert words.dtype == np.int16 and words.shape == (25200,)


def test_matches_cpp_reference_on_motion():
    """v2 fixed split vs the saracen C++ DLL (chroma 0) on synthetic sub-pixel motion."""
    cpp = pytest.importorskip("evm_saracen_cpp")
    try:
        cpp.load_library()
    except Exception as e:
        pytest.skip(f"DLL not built: {e}")
    import math
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    p = FilterParams(fc_lo=0.25, fc_hi=2.44, alpha=20)
    fx = SaracenFixedEVM(p)
    c = cpp.make_saracen(p, chrom_attenuation=0.0)
    for e in (fx.cfg, c.cfg):
        e.settle_s = e.ramp_s = 1e-9
    diffs, mags = [], []
    for i in range(90):
        s = 0.6 * math.sin(2 * math.pi * 1.2 * i / 30)
        v = 128 + 60 * np.sin((xx - s) / 9.0) * np.cos(yy / 13.0)
        f = np.clip(np.stack([v * 0.8, v * 0.9, v], -1), 0, 255).astype(np.uint8)
        a, b = fx.process(f), c.process(f)
        if i > 40:
            diffs.append(np.abs(a - b).mean())
            mags.append(np.abs(b).mean())
    assert np.mean(mags) > 0.5
    assert np.mean(diffs) < 0.15 * np.mean(mags), (np.mean(diffs), np.mean(mags))


def test_vectors_roundtrip_verify(tmp_path, capsys):
    import gen_vectors_saracen as g
    from evm_eval import save_clip, synthetic_clip

    fr, _ = synthetic_clip(n=24, h=480, w=640, motion_px=2.0)
    clip = tmp_path / "c.npz"
    save_clip(fr, clip, 30.0)
    out = tmp_path / "v2"
    assert g.main(["--clip", str(clip), "--n", "20", "--out", str(out)]) == 0
    prm = json.loads((out / "params.json").read_text())
    assert prm["total_words"] == 25200 and prm["N"] == 20
    assert (out / "input_words.bin").stat().st_size == 20 * 25200 * 4
    assert g.main(["verify", "--dir", str(out), "--got", str(out / "expected_output.bin")]) == 0
    bad = bytearray((out / "expected_output.bin").read_bytes())
    bad[(3 * 25200 + 19500) * 4] ^= 1     # frame 3, level 1 word 300
    (tmp_path / "bad.bin").write_bytes(bad)
    capsys.readouterr()
    assert g.main(["verify", "--dir", str(out), "--got", str(tmp_path / "bad.bin")]) == 1
    assert "frame 3 word 19500 (level 1, index 300)" in capsys.readouterr().out
    ep = json.loads((out / "edge" / "params.json").read_text())
    assert ep["init_frames"] == [0, 32] and ep["stats"]["output_max"] == 32767
