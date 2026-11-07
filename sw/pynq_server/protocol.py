"""Wire format and parameter encoding from docs/PLAN.md section 3.

Shared by the laptop client (sw/app/processors.py) and the PYNQ server, so it
must stay importable on the PYNQ image: Python 3.8+, NumPy only.
"""
from __future__ import annotations

import math
import socket
import struct

import numpy as np

PORT = 5005

# uint32 frame_id, uint16 w, h, r_hi, r_lo, alpha, uint8 init, uint8 pad
REQ_HEADER = struct.Struct("<IHHHHHBB")
RESP_HEADER = struct.Struct("<I")
assert REQ_HEADER.size == 16

FPS = 30.0
DEFAULT_FC_HI = 2.0
DEFAULT_FC_LO = 0.83
DEFAULT_ALPHA = 50.0


# ---------------------------------------------------------------- parameters

def coeff_q016(fc: float, fs: float = FPS) -> int:
    """r = 1 - exp(-2*pi*fc/fs) as unsigned Q0.16."""
    r = 1.0 - math.exp(-2.0 * math.pi * fc / fs)
    return min(max(int(round(r * 65536)), 0), 0xFFFF)


def alpha_q88(alpha: float) -> int:
    """Gain as unsigned Q8.8."""
    return min(max(int(round(alpha * 256)), 0), 0xFFFF)


# ---------------------------------------------------------------- pixel packing

def pack_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 BGR -> (H, W, 4) uint8 words: B, G, R, 0."""
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("expected (H, W, 3) uint8, got %s %s" % (frame_bgr.shape, frame_bgr.dtype))
    h, w, _ = frame_bgr.shape
    words = np.zeros((h, w, 4), dtype=np.uint8)
    words[:, :, :3] = frame_bgr
    return words


def unpack_signal(words: np.ndarray) -> np.ndarray:
    """(H, W, 4) int8 output words -> (H, W, 3) int8 signal (B, G, R)."""
    return words[:, :, :3]


# ---------------------------------------------------------------- messages

def encode_request(frame_id: int, words: np.ndarray, r_hi: int, r_lo: int,
                   alpha: int, init: bool) -> bytes:
    h, w, _ = words.shape
    header = REQ_HEADER.pack(frame_id & 0xFFFFFFFF, w, h, r_hi, r_lo, alpha, 1 if init else 0, 0)
    return header + np.ascontiguousarray(words, dtype=np.uint8).tobytes()


def encode_response(frame_id: int, out_words: np.ndarray) -> bytes:
    return RESP_HEADER.pack(frame_id & 0xFFFFFFFF) + np.ascontiguousarray(out_words).tobytes()


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, or raise ConnectionError if the peer closes."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed after %d of %d bytes" % (got, n))
        got += k
    return bytes(buf)
