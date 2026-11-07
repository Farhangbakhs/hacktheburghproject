"""FPGA backend: drives the HLS kernel through AXI DMA (PLAN.md section 5, Jack step 5).

Same interface as CpuBackend, so server.py needs no changes:
    reset(); process(words_in, r_hi, r_lo, alpha, init) -> words_out

    DmaBackend()                     on the board: loads hw/overlay/evm.bit via pynq
    DmaBackend(mock=True)            on a laptop: uses mock_pynq's model of the board

Buffers are allocated once and reused; they're reallocated only when the frame
size changes (which also forces INIT, since the kernel's state is per pixel).
Check against the real hardware: IP names (axi_dma_0, evm_kernel_0) and the
register offsets in kernel_regs.py.
"""
from __future__ import annotations

import os

import numpy as np

from kernel_regs import (AP_DONE, AP_START, REG_ALPHA, REG_CTRL, REG_HEIGHT, REG_INIT,
                         REG_R_HI, REG_R_LO, REG_WIDTH)

DEFAULT_BITFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "hw", "overlay", "evm.bit")


class DmaBackend:
    name = "fpga"

    def __init__(self, bitfile: str = DEFAULT_BITFILE, mock: bool = False,
                 max_pixels: int = 160 * 120) -> None:
        if mock:
            import mock_pynq as pynq_api
            self.overlay = pynq_api.MockOverlay(bitfile, max_pixels=max_pixels)
            self.name = "mockfpga"
        else:
            import pynq as pynq_api  # only exists on the board
            self.overlay = pynq_api.Overlay(bitfile)
        self._allocate = pynq_api.allocate
        self.dma = self.overlay.axi_dma_0
        self.kernel = self.overlay.evm_kernel_0
        self._shape = None
        self._in = self._out = None
        self._need_init = True

    def reset(self) -> None:
        self._need_init = True

    def _ensure_buffers(self, h: int, w: int) -> None:
        if self._shape == (h, w):
            return
        for buf in (self._in, self._out):
            if buf is not None:
                buf.freebuffer()
        self._in = self._allocate(shape=(h * w,), dtype=np.uint32)
        self._out = self._allocate(shape=(h * w,), dtype=np.uint32)
        self._shape = (h, w)
        self._need_init = True
        self.kernel.write(REG_WIDTH, w)
        self.kernel.write(REG_HEIGHT, h)

    def process(self, words_in: np.ndarray, r_hi: int, r_lo: int, alpha: int,
                init: bool) -> np.ndarray:
        h, w, _ = words_in.shape
        self._ensure_buffers(h, w)
        self._in[:] = np.ascontiguousarray(words_in).view(np.uint32).ravel()

        self.kernel.write(REG_R_HI, r_hi)
        self.kernel.write(REG_R_LO, r_lo)
        self.kernel.write(REG_ALPHA, alpha)
        self.kernel.write(REG_INIT, 1 if (init or self._need_init) else 0)
        self._need_init = False

        self.kernel.write(REG_CTRL, AP_START)
        self.dma.recvchannel.transfer(self._out)
        self.dma.sendchannel.transfer(self._in)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        for _ in range(1000):  # done can trail the last output word by a few cycles
            if self.kernel.read(REG_CTRL) & AP_DONE:
                break
        else:
            raise RuntimeError("kernel did not report ap_done after the frame")

        return np.array(self._out).view(np.int8).reshape(h, w, 4)
