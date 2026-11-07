"""TCP frame server (PLAN.md section 3). Runs on the PYNQ ARM or on the laptop.

    python server.py                  # CPU backend, 0.0.0.0:5005
    python server.py --backend dma    # FPGA via AXI DMA (dma_backend.py), on the board
    python server.py --backend mockfpga   # same DMA code against a simulated board

Copy this whole folder (protocol.py, cpu_backend.py, server.py) to the board.
One client at a time; filter state resets for each new connection.
"""
from __future__ import annotations

import argparse
import socket
import threading
import time

import numpy as np

from cpu_backend import CpuBackend
from protocol import PORT, REQ_HEADER, encode_response, recv_exact

MAX_PIXELS = 1920 * 1080


def make_backend(name: str):
    if name == "cpu":
        return CpuBackend()
    if name == "dma":
        from dma_backend import DmaBackend  # real FPGA, needs pynq + evm.bit on the board
        return DmaBackend()
    if name == "mockfpga":
        from dma_backend import DmaBackend  # same code path against mock_pynq's model
        return DmaBackend(mock=True)
    raise ValueError("unknown backend %r" % name)


class FrameServer:
    def __init__(self, backend, host: str = "0.0.0.0", port: int = PORT, verbose: bool = True):
        self.backend = backend
        self.verbose = verbose
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(1)
        self._sock.settimeout(0.2)  # so shutdown() is noticed
        self.address = self._sock.getsockname()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print("[server] " + msg, flush=True)

    def serve_forever(self) -> None:
        self._log("backend=%s listening on %s:%d" % (self.backend.name, self.address[0], self.address[1]))
        try:
            while not self._stop.is_set():
                try:
                    conn, peer = self._sock.accept()
                except socket.timeout:
                    continue
                self._log("client %s:%d connected" % peer)
                try:
                    self._handle(conn)
                except (ConnectionError, OSError) as e:
                    self._log("client gone: %s" % e)
                finally:
                    conn.close()
        finally:
            self._sock.close()

    def shutdown(self) -> None:
        self._stop.set()

    def _handle(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(None)
        self.backend.reset()
        frames, t0, busy = 0, time.perf_counter(), 0.0
        while not self._stop.is_set():
            frame_id, w, h, r_hi, r_lo, alpha, init, _pad = REQ_HEADER.unpack(
                recv_exact(conn, REQ_HEADER.size))
            if w == 0 or h == 0 or w * h > MAX_PIXELS:
                raise ConnectionError("bad frame size %dx%d" % (w, h))
            payload = recv_exact(conn, w * h * 4)
            words_in = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 4)

            t = time.perf_counter()
            words_out = self.backend.process(words_in, r_hi, r_lo, alpha, bool(init))
            busy += time.perf_counter() - t

            conn.sendall(encode_response(frame_id, words_out))

            frames += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 5.0:
                self._log("%dx%d  %.1f fps  backend %.2f ms/frame"
                          % (w, h, frames / elapsed, 1000 * busy / frames))
                frames, t0, busy = 0, time.perf_counter(), 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--backend", choices=["cpu", "dma", "mockfpga"], default="cpu")
    args = ap.parse_args()
    server = FrameServer(make_backend(args.backend), args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
