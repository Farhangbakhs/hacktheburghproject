"""Run ON the PYNQ board in the folder holding loopback.bit/.hwh:
    sudo python3 loopback_test.py [path/to/loopback.bit]
Sends NumPy arrays through DMA -> FIFO -> DMA and checks they come back identical.
Sizes include one full 160x120 frame (19200 words), the real payload."""
import sys, time
import numpy as np
from pynq import Overlay, allocate

bit = sys.argv[1] if len(sys.argv) > 1 else "loopback.bit"
ol = Overlay(bit)
dma = ol.axi_dma_0

ok = True
for n_words in (16, 1024, 19200):          # 32-bit words; FIFO depth is 4096
    src = allocate(shape=(n_words,), dtype=np.uint32)
    dst = allocate(shape=(n_words,), dtype=np.uint32)
    src[:] = np.random.randint(0, 2**32, n_words, dtype=np.uint32)
    dst[:] = 0
    t0 = time.perf_counter()
    dma.recvchannel.transfer(dst)           # start receive first so the FIFO drains
    dma.sendchannel.transfer(src)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    dt = (time.perf_counter() - t0) * 1e3
    same = np.array_equal(src, dst)
    ok &= same
    print(f"{n_words:6d} words: {'PASS' if same else 'FAIL'}  {dt:.2f} ms")
    src.freebuffer(); dst.freebuffer()

print("LOOPBACK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
