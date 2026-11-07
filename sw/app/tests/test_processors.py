"""Run from repo root:  python -m pytest sw/app/tests -q"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from processors import LocalPython, MockServer  # noqa: E402
from protocol import REQ_HEADER, alpha_q88, coeff_q016, pack_bgr  # noqa: E402
from cpu_backend import CpuBackend  # noqa: E402


def scalar_reference(frames, r_hi, r_lo, alpha):
    """Line-by-line transcription of PLAN.md section 3, plain Python ints."""
    h, w, _ = frames[0].shape
    l_hi = l_lo = None
    outs = []
    for i, f in enumerate(frames):
        if i == 0:
            l_hi = [[[int(f[y, x, c]) << 16 for c in range(3)] for x in range(w)] for y in range(h)]
            l_lo = [[[v for v in px] for px in row] for row in l_hi]
        out = np.zeros((h, w, 3), dtype=np.int8)
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    xx = int(f[y, x, c]) << 16
                    if i > 0:
                        l_hi[y][x][c] += (r_hi * (xx - l_hi[y][x][c])) >> 16
                        l_lo[y][x][c] += (r_lo * (xx - l_lo[y][x][c])) >> 16
                    band = l_hi[y][x][c] - l_lo[y][x][c]
                    out[y, x, c] = max(-127, min(127, (alpha * band) >> 24))
        outs.append(out)
    return outs


def pulsing_frames(n, h=12, w=16, seed=0, amp=3.0, hz=1.2):
    """Static random image plus a small sinusoidal brightness pulse."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 215, size=(h, w, 3)).astype(np.float64)
    return [np.clip(base + amp * math.sin(2 * math.pi * hz * i / 30.0)
                    + rng.normal(0, 0.5, size=base.shape), 0, 255).astype(np.uint8)
            for i in range(n)]


def test_header_and_coefficients():
    assert REQ_HEADER.size == 16
    assert coeff_q016(2.0) == round((1 - math.exp(-2 * math.pi * 2.0 / 30)) * 65536)
    assert coeff_q016(0.83) == round((1 - math.exp(-2 * math.pi * 0.83 / 30)) * 65536)
    assert alpha_q88(50) == 12800


def test_cpu_backend_bit_exact_vs_scalar():
    frames = pulsing_frames(40, h=4, w=5)
    r_hi, r_lo, alpha = coeff_q016(2.0), coeff_q016(0.83), alpha_q88(50)
    expected = scalar_reference(frames, r_hi, r_lo, alpha)
    be = CpuBackend()
    for i, f in enumerate(frames):
        out = be.process(pack_bgr(f), r_hi, r_lo, alpha, init=(i == 0))
        assert np.array_equal(out[:, :, :3], expected[i]), "frame %d" % i
        assert not out[:, :, 3].any()
    assert any(e.any() for e in expected), "test is vacuous: all outputs zero"


def test_first_frame_is_zero_and_pulse_is_amplified():
    frames = pulsing_frames(120)
    p = LocalPython()
    outs = [p.process(f).astype(np.int32) for f in frames]
    assert not outs[0].any()
    # After settling, the output should swing well beyond the +/-3 input pulse.
    assert np.ptp([o.mean() for o in outs[60:]]) > 20


def test_mock_server_matches_local_including_reset():
    frames = pulsing_frames(60, h=120, w=160)
    with LocalPython() as local, MockServer() as mock:
        for i, f in enumerate(frames):
            if i == 30:
                local.reset()
                mock.reset()
            if i == 45:
                local.set_params(alpha=20, fc_hi=3.0)
                mock.set_params(alpha=20, fc_hi=3.0)
            a, b = local.process(f), mock.process(f)
            assert a.dtype == b.dtype == np.int8
            assert a.shape == b.shape == (120, 160, 3)
            assert np.array_equal(a, b), "frame %d" % i


def test_server_survives_reconnect():
    frames = pulsing_frames(5)
    m = MockServer()
    try:
        m.process(frames[0])
        m._sock.close()
        from processors import RemoteProcessor
        r = RemoteProcessor("127.0.0.1", m._server.address[1])
        out = r.process(frames[1])  # new connection -> state reset -> zero output
        assert not out.any()
        r.close()
    finally:
        m.close()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        LocalPython().process(np.zeros((10, 10), dtype=np.uint8))
