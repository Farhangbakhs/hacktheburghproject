"""Fake PYNQ board for developing the DMA backend on a laptop.

Imitates the slice of the pynq API that dma_backend.py uses:

    allocate(shape, dtype)                 -> buffer with .physical_address, flush/invalidate/freebuffer
    MockOverlay(...).axi_dma_0             -> .sendchannel / .recvchannel with .transfer(buf), .wait()
    MockOverlay(...).evm_kernel_0          -> .write(offset, value), .read(offset)

and a model of the HLS kernel behind them that behaves like hardware rather
than like friendly Python:
  * filter state is int32 (wraps, never widens), held across frames;
  * nothing happens until ap_start is written to CTRL;
  * the DMA length must equal width*height words or the "hardware" hangs
    (here: DMA wait() raises);
  * frames larger than the synthesised maximum are refused;
  * state powers up as zero and is only (re)initialised by the INIT register.
The filter itself is section 3 of docs/PLAN.md, written independently of
cpu_backend.py (operating on packed 32-bit words) so each checks the other.
"""
from __future__ import annotations

import numpy as np

from kernel_regs import (AP_DONE, AP_IDLE, AP_START, REG_ALPHA, REG_CTRL, REG_HEIGHT,
                         REG_INIT, REG_R_HI, REG_R_LO, REG_WIDTH)

CLOCK_HZ = 100e6
PIPELINE_DEPTH = 40  # cycles of latency before the first output pixel


class DmaTimeout(RuntimeError):
    """What a hung AXI-Stream looks like from Python on the real board (a wait that never returns)."""


# ------------------------------------------------------------------ buffers

class MockBuffer(np.ndarray):
    _next_addr = 0x1000_0000

    def __new__(cls, shape, dtype):
        obj = np.zeros(shape, dtype=dtype).view(cls)
        obj.physical_address = MockBuffer._next_addr
        MockBuffer._next_addr += (obj.nbytes + 0xFFF) & ~0xFFF
        obj.freed = False
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.physical_address = getattr(obj, "physical_address", 0)
            self.freed = getattr(obj, "freed", False)

    def flush(self):
        pass

    def invalidate(self):
        pass

    def freebuffer(self):
        self.freed = True


def allocate(shape, dtype=np.uint32, **_kw):
    return MockBuffer(shape, dtype)


# ------------------------------------------------------------------ kernel model

class EvmKernelModel:
    """Behavioural model of the HLS kernel (AXI-Lite registers + AXI-Stream in/out)."""

    def __init__(self, max_pixels: int = 160 * 120) -> None:
        self.max_pixels = max_pixels
        self.regs = {REG_CTRL: AP_IDLE, REG_WIDTH: 0, REG_HEIGHT: 0, REG_R_HI: 0,
                     REG_R_LO: 0, REG_ALPHA: 0, REG_INIT: 0}
        self.l_hi = np.zeros(max_pixels * 3, dtype=np.int32)  # "BRAM", powers up zeroed
        self.l_lo = np.zeros(max_pixels * 3, dtype=np.int32)
        self.out_fifo = None
        self.frames = 0
        self.last_cycles = 0

    # AXI-Lite
    def write(self, offset: int, value: int) -> None:
        if offset not in self.regs:
            raise KeyError("no register at 0x%02x" % offset)
        if offset == REG_CTRL:
            if value & AP_START:
                self.regs[REG_CTRL] = AP_START  # running: not idle, not done
            return
        self.regs[offset] = int(value) & 0xFFFFFFFF

    def read(self, offset: int) -> int:
        value = self.regs[offset]
        if offset == REG_CTRL and value & AP_DONE:
            self.regs[REG_CTRL] = AP_IDLE  # ap_done is clear-on-read
        return value

    @property
    def running(self) -> bool:
        return bool(self.regs[REG_CTRL] & AP_START)

    # AXI-Stream in -> out
    def consume(self, words: np.ndarray) -> None:
        w, h = self.regs[REG_WIDTH] & 0xFFFF, self.regs[REG_HEIGHT] & 0xFFFF
        n = w * h
        if not self.running:
            raise DmaTimeout("stream sent but kernel not started (write AP_START to CTRL first)")
        if n == 0 or n > self.max_pixels:
            raise DmaTimeout("kernel sized for %d px, registers say %dx%d" % (self.max_pixels, w, h))
        if words.size != n:
            raise DmaTimeout("kernel expects %d words (%dx%d), DMA sent %d" % (n, w, h, words.size))

        words = words.astype(np.uint32)
        # unpack byte lanes: byte0=B, byte1=G, byte2=R
        p = np.stack([(words >> (8 * c)) & 0xFF for c in range(3)], axis=1).astype(np.int64).ravel()
        x = p << 16
        k = n * 3
        r_hi = self.regs[REG_R_HI] & 0xFFFF
        r_lo = self.regs[REG_R_LO] & 0xFFFF
        alpha = self.regs[REG_ALPHA] & 0xFFFF

        if self.regs[REG_INIT] & 1:
            l_hi = x
            l_lo = x
        else:
            l_hi = self.l_hi[:k].astype(np.int64)
            l_lo = self.l_lo[:k].astype(np.int64)
            l_hi = l_hi + ((r_hi * (x - l_hi)) >> 16)
            l_lo = l_lo + ((r_lo * (x - l_lo)) >> 16)
        self.l_hi[:k] = l_hi.astype(np.int32)  # registers are 32 bits wide
        self.l_lo[:k] = l_lo.astype(np.int32)

        band = self.l_hi[:k].astype(np.int64) - self.l_lo[:k].astype(np.int64)
        out = np.clip((alpha * band) >> 24, -127, 127).astype(np.int64) & 0xFF
        out = out.reshape(n, 3)
        self.out_fifo = (out[:, 0] | (out[:, 1] << 8) | (out[:, 2] << 16)).astype(np.uint32)

        self.frames += 1
        self.last_cycles = n + PIPELINE_DEPTH
        self.regs[REG_CTRL] = AP_DONE | AP_IDLE


# ------------------------------------------------------------------ DMA

class _Channel:
    def __init__(self, dma, direction: str) -> None:
        self._dma, self._dir = dma, direction
        self._buf = None
        self.idle = True

    def transfer(self, buf, nbytes: int = None) -> None:
        if getattr(buf, "freed", False):
            raise RuntimeError("transfer from a freed buffer")
        if not hasattr(buf, "physical_address"):
            raise TypeError("DMA needs a buffer from pynq.allocate(), not a plain array")
        if not self.idle:
            raise RuntimeError("%s channel busy: wait() for the previous transfer" % self._dir)
        self._buf = buf
        self.idle = False
        self._dma._kick()

    def wait(self) -> None:
        if self.idle:
            return
        self._dma._kick()
        if not self.idle:
            raise DmaTimeout("%s transfer never completed (kernel not started, or nothing to receive)"
                             % self._dir)


class MockDma:
    def __init__(self, kernel: EvmKernelModel) -> None:
        self.kernel = kernel
        self.sendchannel = _Channel(self, "send")
        self.recvchannel = _Channel(self, "recv")

    def _kick(self) -> None:
        """Move data whenever both ends are ready, like the AXI-Stream handshake."""
        send, recv = self.sendchannel, self.recvchannel
        if not send.idle and self.kernel.running and self.kernel.out_fifo is None:
            self.kernel.consume(np.asarray(send._buf).ravel())
            send.idle, send._buf = True, None
        if not recv.idle and self.kernel.out_fifo is not None:
            out = recv._buf.view(np.uint32).ravel()
            if out.size < self.kernel.out_fifo.size:
                raise DmaTimeout("recv buffer %d words, kernel produced %d"
                                 % (out.size, self.kernel.out_fifo.size))
            out[:self.kernel.out_fifo.size] = self.kernel.out_fifo
            self.kernel.out_fifo = None
            recv.idle, recv._buf = True, None


class MockOverlay:
    """Stands in for pynq.Overlay('evm.bit'). IP names match the planned block design."""

    def __init__(self, bitfile: str = "evm.bit", max_pixels: int = 160 * 120) -> None:
        self.bitfile = bitfile
        self.evm_kernel_0 = EvmKernelModel(max_pixels)
        self.axi_dma_0 = MockDma(self.evm_kernel_0)
