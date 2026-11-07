"""Is there a pulse in this clip? Per-ROI spectrum, control regions, and drift-corrected face signal.
    .venv/bin/python analyze_pulse.py CLIP --shot close|far
ROIs are fractions (x0,y0,x1,y1) of the frame, chosen by eye for the two framings we recorded.
Checks:
  * exposure steadiness: std of the (drift-removed-free) mean green in the control regions
  * per-ROI peak in 0.8-2.0 Hz and a peak-vs-rest-of-band SNR in dB
  * the face signal after regressing out the control-region signal (cancels shared exposure/motion drift)
  * first-half vs second-half agreement of the pooled-face peak (a real pulse should be stable)
Crude by design: one clip, hand-picked boxes. Not a medical measurement.
"""
import argparse
import numpy as np
from render_clip import read_small_frames

SHOTS = {
    "close": dict(face=dict(forehead=(.30, .17, .70, .27), cheekL=(.15, .43, .33, .55), cheekR=(.68, .43, .88, .55)),
                  ctrl=dict(chair=(0.0, .25, .08, .60), shirt=(.20, .80, .80, .98))),
    "far": dict(face=dict(forehead=(.38, .22, .60, .30), cheekL=(.33, .40, .43, .47), cheekR=(.56, .40, .65, .47)),
                ctrl=dict(wall=(.05, .03, .45, .18), shirt=(.25, .70, .75, .95))),
}
FS = 30.0


def sig(small, r, ch=1):
    n, h, w, _ = small.shape
    x0, y0, x1, y1 = int(r[0] * w), int(r[1] * h), int(r[2] * w), int(r[3] * h)
    return small[:, y0:y1, x0:x1, ch].reshape(n, -1).mean(1).astype(float)


def detrend(g, k=45):  # remove drift slower than ~0.7 Hz (1.5 s moving average)
    pad = np.pad(g, (k // 2, k - k // 2 - 1), mode="edge")
    return g - np.convolve(pad, np.ones(k) / k, mode="valid")


def peak(g):
    g = detrend(g); g = g - g.mean(); n = len(g)
    nfft = 8192
    sp = np.abs(np.fft.rfft(g * np.hanning(n), nfft)) ** 2; f = np.fft.rfftfreq(nfft, 1 / FS)
    b = (f >= .8) & (f <= 2.0); k = np.argmax(np.where(b, sp, 0))
    w = (f >= f[k] - .12) & (f <= f[k] + .12); wide = (f >= .7) & (f <= 3.5)
    snr = 10 * np.log10(sp[w].sum() / (sp[wide & ~w].sum() * w.sum() / (wide & ~w).sum()))
    return f[k] * 60, snr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clip"); ap.add_argument("--shot", choices=SHOTS, required=True)
    ap.add_argument("--skip", type=float, default=1.0, help="seconds to skip at start (camera settling)")
    a = ap.parse_args()
    small, _ = read_small_frames(a.clip, 900)
    small = small[int(a.skip * FS):]
    n = len(small); S = SHOTS[a.shot]
    print(f"{a.clip}: {n} frames ({n/FS:.1f} s) at {small.shape[2]}x{small.shape[1]}")
    faces = {k: sig(small, r) for k, r in S["face"].items()}
    ctrls = {k: sig(small, r) for k, r in S["ctrl"].items()}
    for k, g in ctrls.items():
        print(f"  ctrl {k:8s}: raw std {g.std():5.2f} lvl, drift-removed std {detrend(g).std():.3f} lvl", end="")
        bpm, snr = peak(g); print(f"   peak {bpm:5.1f} bpm SNR {snr:+5.1f} dB")
    for k, g in faces.items():
        bpm, snr = peak(g)
        print(f"  face {k:8s}: raw std {g.std():5.2f} lvl, drift-removed std {detrend(g).std():.3f} lvl   peak {bpm:5.1f} bpm SNR {snr:+5.1f} dB")
    pooled = np.mean([detrend(g) / detrend(g).std() for g in faces.values()], axis=0)
    bpm, snr = peak(pooled); print(f"  POOLED FACE: peak {bpm:.1f} bpm SNR {snr:+.1f} dB")
    # regress out the control signal (shared exposure/motion drift), per face ROI, then pool
    C = np.stack([detrend(g) for g in ctrls.values()] + [np.ones(n)], 1)
    res = []
    for k, g in faces.items():
        d = detrend(g); coef, *_ = np.linalg.lstsq(C, d, rcond=None); r = d - C @ coef
        res.append(r / r.std())
        b, s = peak(r); print(f"  face {k:8s} minus controls: peak {b:5.1f} bpm SNR {s:+5.1f} dB")
    pr = np.mean(res, axis=0); b, s = peak(pr); print(f"  POOLED FACE minus controls: peak {b:.1f} bpm SNR {s:+.1f} dB")
    h = n // 2
    b1, s1 = peak(pr[:h]); b2, s2 = peak(pr[h:])
    print(f"  stability (pooled, drift-corrected): 1st half {b1:.0f} bpm ({s1:+.1f} dB), 2nd half {b2:.0f} bpm ({s2:+.1f} dB)")


if __name__ == "__main__":
    main()
