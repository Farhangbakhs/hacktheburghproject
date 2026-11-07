"""DMA backend against the simulated board.  python -m pytest sw/app/tests -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from processors import LocalPython, MockServer  # noqa: E402
from cpu_backend import CpuBackend  # noqa: E402
from dma_backend import DmaBackend  # noqa: E402
import mock_pynq  # noqa: E402
from kernel_regs import AP_DONE, AP_IDLE, AP_START, REG_CTRL, REG_HEIGHT, REG_INIT, REG_WIDTH  # noqa: E402
from protocol import alpha_q88, coeff_q016, pack_bgr  # noqa: E402
from test_processors import pulsing_frames  # noqa: E402

R_HI, R_LO, ALPHA = coeff_q016(2.0), coeff_q016(0.83), alpha_q88(50)


def run_both(frames, events=None):
    """Feed the same frames to CpuBackend and DmaBackend(mock); yield both outputs."""
    events = events or {}
    cpu, fpga = CpuBackend(), DmaBackend(mock=True)
    r_hi, alpha = R_HI, ALPHA
    for i, f in enumerate(frames):
        ev = events.get(i)
        if ev == "reset":
            cpu.reset()
            fpga.reset()
        elif ev == "params":
            r_hi, alpha = coeff_q016(3.0), alpha_q88(20)
        words = pack_bgr(f)
        yield i, cpu.process(words, r_hi, R_LO, alpha, i == 0), fpga.process(words, r_hi, R_LO, alpha, i == 0)


def test_mock_fpga_bit_exact_vs_cpu():
    frames = pulsing_frames(60, h=120, w=160)
    nonzero = False
    for i, a, b in run_both(frames, {30: "reset", 45: "params"}):
        assert b.dtype == np.int8 and b.shape == (120, 160, 4)
        assert np.array_equal(a, b), "frame %d" % i
        nonzero |= bool(b.any())
    assert nonzero


def test_size_change_reinitialises():
    big, small = pulsing_frames(10, h=120, w=160), pulsing_frames(10, h=60, w=80)
    cpu, fpga = CpuBackend(), DmaBackend(mock=True)
    for i, f in enumerate(big + small):
        a = cpu.process(pack_bgr(f), R_HI, R_LO, ALPHA, i == 0)
        b = fpga.process(pack_bgr(f), R_HI, R_LO, ALPHA, i == 0)
        assert np.array_equal(a, b), "frame %d" % i
    assert fpga.kernel.regs[REG_WIDTH] == 80 and fpga.kernel.regs[REG_HEIGHT] == 60


def test_frame_bigger_than_synthesised_kernel_is_refused():
    fpga = DmaBackend(mock=True, max_pixels=160 * 120)
    with pytest.raises(mock_pynq.DmaTimeout):
        fpga.process(pack_bgr(np.zeros((240, 320, 3), np.uint8)), R_HI, R_LO, ALPHA, True)


def test_init_register_only_set_when_needed():
    fpga = DmaBackend(mock=True)
    f = pack_bgr(pulsing_frames(1)[0])
    fpga.process(f, R_HI, R_LO, ALPHA, False)
    assert fpga.kernel.regs[REG_INIT] == 1  # first frame forces init
    fpga.process(f, R_HI, R_LO, ALPHA, False)
    assert fpga.kernel.regs[REG_INIT] == 0
    fpga.reset()
    fpga.process(f, R_HI, R_LO, ALPHA, False)
    assert fpga.kernel.regs[REG_INIT] == 1


# --- the mock should catch driver mistakes that would hang a real board

def _board(w=16, h=12):
    ol = mock_pynq.MockOverlay(max_pixels=w * h)
    k = ol.evm_kernel_0
    k.write(REG_WIDTH, w)
    k.write(REG_HEIGHT, h)
    k.write(REG_INIT, 1)
    return ol, mock_pynq.allocate((w * h,), np.uint32), mock_pynq.allocate((w * h,), np.uint32)


def test_stream_without_ap_start_hangs():
    ol, src, dst = _board()
    ol.axi_dma_0.recvchannel.transfer(dst)
    with pytest.raises(mock_pynq.DmaTimeout):
        ol.axi_dma_0.sendchannel.transfer(src)
        ol.axi_dma_0.sendchannel.wait()


def test_wrong_length_hangs():
    ol, _, dst = _board()
    ol.evm_kernel_0.write(REG_CTRL, AP_START)
    ol.axi_dma_0.recvchannel.transfer(dst)
    with pytest.raises(mock_pynq.DmaTimeout):
        ol.axi_dma_0.sendchannel.transfer(mock_pynq.allocate((10,), np.uint32))


def test_plain_numpy_array_rejected():
    ol, _, _ = _board()
    with pytest.raises(TypeError):
        ol.axi_dma_0.sendchannel.transfer(np.zeros(192, np.uint32))


def test_recv_without_send_times_out():
    ol, _, dst = _board()
    ol.evm_kernel_0.write(REG_CTRL, AP_START)
    ol.axi_dma_0.recvchannel.transfer(dst)
    with pytest.raises(mock_pynq.DmaTimeout):
        ol.axi_dma_0.recvchannel.wait()


def test_ap_done_is_clear_on_read():
    ol, src, dst = _board()
    k = ol.evm_kernel_0
    k.write(REG_CTRL, AP_START)
    ol.axi_dma_0.recvchannel.transfer(dst)
    ol.axi_dma_0.sendchannel.transfer(src)
    assert k.read(REG_CTRL) & AP_DONE
    assert k.read(REG_CTRL) == AP_IDLE


def test_app_through_server_with_mock_fpga():
    frames = pulsing_frames(40, h=120, w=160)
    with LocalPython() as local, MockServer(backend="mockfpga") as fpga:
        assert fpga.name == "mockfpga"
        for i, f in enumerate(frames):
            assert np.array_equal(local.process(f), fpga.process(f)), "frame %d" % i
