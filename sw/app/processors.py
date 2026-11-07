"""Swappable processing backends for the app.

All share one interface:
    p.process(small_bgr)  (H, W, 3) uint8 BGR -> (H, W, 3) int8 amplified signal
    p.set_params(alpha=..., fc_hi=..., fc_lo=...)
    p.reset()             re-initialise filter state on the next frame
    p.close()
    p.last_latency_ms     wall time of the last process() call

    LocalPython()          in-process fixed-point model, no network
    MockServer()           starts server.py on localhost and talks TCP to it
    MockServer("mockfpga") same, with the DMA backend driving a simulated FPGA
    PynqRemote(host)       same TCP client, pointed at the board
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time

import numpy as np

_SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pynq_server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from cpu_backend import CpuBackend  # noqa: E402
from protocol import (  # noqa: E402
    DEFAULT_ALPHA, DEFAULT_FC_HI, DEFAULT_FC_LO, FPS, PORT, RESP_HEADER,
    alpha_q88, coeff_q016, encode_request, pack_bgr, recv_exact, unpack_signal,
)

PYNQ_DEFAULT_IP = "192.168.2.99"  # PYNQ image's default static IP on the Ethernet port


class Processor:
    name = "base"

    def __init__(self, alpha: float = DEFAULT_ALPHA, fc_hi: float = DEFAULT_FC_HI,
                 fc_lo: float = DEFAULT_FC_LO, fs: float = FPS) -> None:
        self.fs = fs
        self.alpha = alpha
        self.fc_hi = fc_hi
        self.fc_lo = fc_lo
        self._init_pending = True
        self.last_latency_ms = 0.0

    def set_params(self, alpha: float = None, fc_hi: float = None, fc_lo: float = None) -> None:
        if alpha is not None:
            self.alpha = alpha
        if fc_hi is not None:
            self.fc_hi = fc_hi
        if fc_lo is not None:
            self.fc_lo = fc_lo

    def reset(self) -> None:
        self._init_pending = True

    def _regs(self):
        return coeff_q016(self.fc_hi, self.fs), coeff_q016(self.fc_lo, self.fs), alpha_q88(self.alpha)

    def process(self, small_bgr: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        r_hi, r_lo, alpha = self._regs()
        init, self._init_pending = self._init_pending, False
        out = self._process_words(pack_bgr(small_bgr), r_hi, r_lo, alpha, init)
        self.last_latency_ms = 1000 * (time.perf_counter() - t)
        return unpack_signal(out)

    def _process_words(self, words, r_hi, r_lo, alpha, init):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LocalPython(Processor):
    """Runs the fixed-point model in-process.

    Uses the server's CpuBackend (an implementation of PLAN.md section 3).
    TODO: once Person 2's sw/reference/evm_fixed.py exists, cross-check against
    it / the golden vectors, and switch to it if they differ.
    """
    name = "local"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._backend = CpuBackend()

    def _process_words(self, words, r_hi, r_lo, alpha, init):
        return self._backend.process(words, r_hi, r_lo, alpha, init)


class RemoteProcessor(Processor):
    """TCP client for server.py. One request in flight at a time."""
    name = "remote"

    def __init__(self, host: str, port: int = PORT, timeout: float = 2.0, **kw) -> None:
        super().__init__(**kw)
        self.host, self.port = host, port
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._frame_id = 0

    def _process_words(self, words, r_hi, r_lo, alpha, init):
        h, w, _ = words.shape
        fid = self._frame_id
        self._frame_id = (fid + 1) & 0xFFFFFFFF
        self._sock.sendall(encode_request(fid, words, r_hi, r_lo, alpha, init))
        (got_id,) = RESP_HEADER.unpack(recv_exact(self._sock, RESP_HEADER.size))
        payload = recv_exact(self._sock, w * h * 4)
        if got_id != fid:
            raise ConnectionError("frame_id mismatch: sent %d, got %d" % (fid, got_id))
        return np.frombuffer(payload, dtype=np.int8).reshape(h, w, 4)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class PynqRemote(RemoteProcessor):
    name = "pynq"

    def __init__(self, host: str = PYNQ_DEFAULT_IP, port: int = PORT, **kw) -> None:
        super().__init__(host, port, **kw)


class MockServer(RemoteProcessor):
    """Starts a local server.py in a thread and connects to it.

    backend="cpu" uses the NumPy model; backend="mockfpga" runs the real DMA
    backend code against mock_pynq's simulated board.
    """
    name = "mock"

    def __init__(self, backend: str = "cpu", **kw) -> None:
        from server import FrameServer, make_backend  # sw/pynq_server/server.py
        self.name = "mock" if backend == "cpu" else backend
        self._server = FrameServer(make_backend(backend), host="127.0.0.1", port=0, verbose=False)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        super().__init__("127.0.0.1", self._server.address[1], **kw)

    def close(self) -> None:
        super().close()
        self._server.shutdown()
        self._thread.join(timeout=2.0)


def make_processor(kind: str, host: str = PYNQ_DEFAULT_IP, port: int = PORT, **kw) -> Processor:
    if kind == "local":
        return LocalPython(**kw)
    if kind == "mock":
        return MockServer(**kw)
    if kind == "mockfpga":
        return MockServer(backend="mockfpga", **kw)
    if kind == "pynq":
        return PynqRemote(host, port, **kw)
    raise ValueError("unknown processor %r" % kind)
