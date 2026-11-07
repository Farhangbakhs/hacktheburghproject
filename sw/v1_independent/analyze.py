"""Quick objective look at a clip: is there a ~1 Hz pulse in the raw signal, and how strong is the filtered output?
    .venv/bin/python analyze.py CLIP_OR_VECTOR_DIR [--roi x0 y0 x1 y1]   (fractions of the frame; default centre 50%)
Mean green over the ROI -> FFT peak in 0.8-2.0 Hz (BPM guess) and peak/median spectral ratio (>~10 = clear peak),
plus the amplitude of the v1 filter output over the same box. Run it on a face box AND a background box as a control.
"""
import sys
import numpy as np
from evm_v1 import FixedV1
from render_clip import read_small_frames

args = sys.argv[1:]
src = args[0]
roi = [float(v) for v in args[args.index("--roi") + 1: args.index("--roi") + 5]] if "--roi" in args else [.25, .25, .75, .75]
small, _ = read_small_frames(src, 600)
n, h, w, _ = small.shape
x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
g = small[:, y0:y1, x0:x1, 1].reshape(n, -1).mean(1)
g = g - g.mean()
spec = np.abs(np.fft.rfft(g * np.hanning(n))) ** 2
f = np.fft.rfftfreq(n, 1 / 30.0)
band = (f >= 0.8) & (f <= 2.0)
k = np.argmax(np.where(band, spec, 0))
print(f"ROI x {x0}-{x1} y {y0}-{y1}: {n} frames; green std {g.std():.3f} levels")
print(f"  peak in 0.8-2.0 Hz: {f[k]:.2f} Hz = {f[k]*60:.0f} bpm; peak/median(band) = {spec[k]/np.median(spec[band]):.1f}")
m = FixedV1(); o = np.stack([m.process(small[i], init=i == 0)[y0:y1, x0:x1] for i in range(n)])
print(f"  v1 output: mean |out| {np.abs(o).mean():.2f}, 99th pct {np.percentile(np.abs(o), 99):.0f} (clip 127)")
