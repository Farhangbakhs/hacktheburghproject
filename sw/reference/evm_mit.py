"""Offline replica of MIT's EVM colour magnification (Wu et al. 2012).

Mirrors `amplify_spatial_Gdown_temporal_ideal.m` from the MIT release, which
produced the famous face/pulse videos:

  1. RGB -> YIQ (NTSC), float in [0, 1]
  2. Gaussian down-sample each frame to pyramid level L (face: L=4)
  3. *Ideal* temporal bandpass: FFT along time over the whole clip, zero all
     bins outside [fl, fh] (face: 50/60 - 60/60 Hz, i.e. 50-60 bpm), inverse FFT
  4. multiply by alpha (face: 50), attenuate I/Q by chrom_attenuation (face: 1)
  5. resize back to full size, add to the original YIQ, -> RGB, clip

This is non-causal (needs the whole clip), so it is a *diagnostic / gold
standard*: if this does not show a clear pulse on a recording, the problem is
capture (light, exposure, motion, compression), not the live algorithm.

`--auto-band` replaces the hand-picked band with f0 +/- bw, where f0 is the
strongest 0.7-3 Hz peak of the skin-mean green signal (MIT picked the band
by hand for each subject; 0.83-1.0 Hz only suits ~50-60 bpm).

Usage:
  python evm_mit.py recordings/me.npz                     # MIT face defaults
  python evm_mit.py recordings/me.npz --auto-band --alpha 100 --level 4
  python evm_mit.py clip.mp4 --out recordings/me_mit.mp4 --no-show
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# NTSC RGB<->YIQ, same matrices as MIT rgb2ntsc / ntsc2rgb.
_RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                     [0.596, -0.274, -0.322],
                     [0.211, -0.523, 0.312]], np.float32)
_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


@dataclass
class MITParams:
    alpha: float = 50.0
    level: int = 4
    fl: float = 50 / 60
    fh: float = 60 / 60
    chrom_attenuation: float = 1.0
    fs: float = 30.0


def bgr_to_yiq(frames_bgr: np.ndarray) -> np.ndarray:
    rgb = frames_bgr[..., ::-1].astype(np.float32) / 255.0
    return rgb @ _RGB2YIQ.T


def yiq_to_bgr(yiq: np.ndarray) -> np.ndarray:
    rgb = yiq @ _YIQ2RGB.T
    return (np.clip(rgb, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)


def gaussian_stack(yiq: np.ndarray, level: int) -> np.ndarray:
    """(N,H,W,3) -> (N,h,w,3): each frame blurred+decimated `level` times."""
    out = []
    for f in yiq:
        for _ in range(level):
            f = cv2.pyrDown(f)
        out.append(f)
    return np.stack(out)


def ideal_bandpass(stack: np.ndarray, fl: float, fh: float, fs: float) -> np.ndarray:
    """MIT ideal_bandpassing: zero FFT bins outside [fl, fh] along axis 0."""
    n = stack.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = np.fft.rfft(stack, axis=0)
    keep = (freqs >= fl) & (freqs <= fh)
    spec[~keep] = 0
    return np.fft.irfft(spec, n=n, axis=0).astype(np.float32)


def estimate_pulse_hz(stack_yiq: np.ndarray, fs: float, skin: np.ndarray | None = None,
                      band: tuple[float, float] = (0.7, 3.0)) -> tuple[float, float]:
    """Strongest in-band frequency of the (skin-)mean signal and its SNR in dB.

    Uses the green channel reconstructed from YIQ (pulse is strongest there).
    """
    rgb = stack_yiq @ _YIQ2RGB.T
    g = rgb[..., 1]
    if skin is not None:
        m = cv2.resize(skin.astype(np.float32), (g.shape[2], g.shape[1]), interpolation=cv2.INTER_AREA)
        sig = (g * m).sum(axis=(1, 2)) / max(m.sum(), 1e-6)
    else:
        h, w = g.shape[1:]
        sig = g[:, h // 4: 3 * h // 4, w // 3: 2 * w // 3].mean(axis=(1, 2))
    sig = sig - np.polyval(np.polyfit(np.arange(len(sig)), sig, 2), np.arange(len(sig)))
    nfft = max(4096, len(sig))
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    inband = (freqs >= band[0]) & (freqs <= band[1])
    k = int(np.argmax(np.where(inband, spec, 0)))
    f0 = float(freqs[k])
    near = inband & ((np.abs(freqs - f0) < 0.1) | (np.abs(freqs - 2 * f0) < 0.1))
    snr = 10 * np.log10(spec[near].sum() / max(spec[inband & ~near].sum(), 1e-12))
    return f0, float(snr)


def magnify(frames_bgr: np.ndarray, p: MITParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns (magnified uint8 frames, amplified YIQ signal at the pooled size).

    Frame-by-frame so memory stays at uint8 clip size (+ the small stack).
    """
    stack = np.stack([gaussian_stack(bgr_to_yiq(f[None]), p.level)[0] for f in frames_bgr])
    filt = ideal_bandpass(stack, p.fl, p.fh, p.fs) * p.alpha
    filt[..., 1:] *= p.chrom_attenuation
    h, w = frames_bgr.shape[1:3]
    out = np.empty_like(frames_bgr)
    for i, f in enumerate(frames_bgr):
        up = cv2.resize(filt[i], (w, h), interpolation=cv2.INTER_CUBIC)
        out[i] = yiq_to_bgr(bgr_to_yiq(f) + up)
    return out, filt


def load_frames(path: str) -> tuple[np.ndarray, float]:
    if path.endswith(".npz"):
        d = np.load(path)
        return d["frames"], float(d["fs"]) if "fs" in d else 30.0
    cap = cv2.VideoCapture(path)
    fs = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames), fs


def skin_mask_for(frames: np.ndarray) -> np.ndarray | None:
    """Average face-tracker skin mask over a few frames (None if no face)."""
    try:
        from face_tracker import FaceTracker
    except Exception:
        return None
    tr = FaceTracker()
    acc, n = None, 0
    for i, f in enumerate(frames[: min(len(frames), 90)]):
        r = tr.update(f, i / 30.0)
        if r.found and r.skin_mask is not None:
            acc = r.skin_mask.copy() if acc is None else acc + r.skin_mask
            n += 1
    return None if acc is None else acc / n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--fl", type=float, default=50 / 60)
    ap.add_argument("--fh", type=float, default=60 / 60)
    ap.add_argument("--chrom", type=float, default=1.0, help="chrom attenuation")
    ap.add_argument("--auto-band", action="store_true", help="centre band on measured pulse")
    ap.add_argument("--bw", type=float, default=0.1, help="half bandwidth for --auto-band (Hz)")
    ap.add_argument("--out", default=None, help="write side-by-side mp4 here")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    frames, fs = load_frames(args.clip)
    p = MITParams(args.alpha, args.level, args.fl, args.fh, args.chrom, fs)
    skin = skin_mask_for(frames)
    small = np.stack([gaussian_stack(bgr_to_yiq(f[None]), p.level)[0] for f in frames])
    f0, snr = estimate_pulse_hz(small, fs, skin)
    print(f"[mit] {len(frames)} frames @ {fs:.1f} fps, {frames.shape[2]}x{frames.shape[1]}; "
          f"skin mask {'yes' if skin is not None else 'no'}; "
          f"pulse peak {f0:.2f} Hz = {f0 * 60:.0f} bpm, SNR {snr:.1f} dB")
    if args.auto_band:
        p.fl, p.fh = max(0.3, f0 - args.bw), f0 + args.bw
    print(f"[mit] band {p.fl:.2f}-{p.fh:.2f} Hz ({p.fl * 60:.0f}-{p.fh * 60:.0f} bpm), "
          f"alpha {p.alpha}, level {p.level}, chrom {p.chrom_attenuation}")
    mag, _ = magnify(frames, p)
    side = np.concatenate([frames, mag], axis=2)

    out = args.out or str(Path(args.clip).with_suffix("")) + "_mit.mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fs, (side.shape[2], side.shape[1]))
    for f in side:
        vw.write(f)
    vw.release()
    print(f"[mit] wrote {out}")
    for i in (len(side) // 3, len(side) // 3 + int(fs / f0 / 2)):  # peak & trough-ish pair
        cv2.imwrite(str(Path(out).with_suffix("")) + f"_f{i}.png", side[i])

    if not args.no_show:
        win = "MIT EVM replica (original | magnified)  q: quit"
        i = 0
        while True:
            cv2.imshow(win, side[i % len(side)])
            k = cv2.waitKey(int(1000 / fs)) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) == 0:
                break
            i += 1
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
