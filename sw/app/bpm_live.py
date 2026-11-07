"""Temporary live BPM estimator for the app.

Stand-in until Person 2's sw/reference/bpm.py lands (PLAN.md 5, Person 2 step 5);
face ROI -> mean colour -> CHROM combination -> FFT over a rolling window -> peak in band.
(Plan says mean green; CHROM did much better in a live test, see chrom().)
Swap it out, or keep whichever is more accurate against a watch.
"""
from __future__ import annotations

import collections

import cv2
import numpy as np


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.0) -> np.ndarray:
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


def chrom(bgr: np.ndarray, fs: float) -> np.ndarray:
    """CHROM pulse signal (de Haan & Jeanne 2013) from (N, 3) mean B, G, R traces.

    Combines colour channels so that brightness changes from movement and
    lighting largely cancel, leaving the blood-volume colour change. On a 30 s
    lossless capture it held the right peak where plain green drifted to ~50 BPM.
    """
    n = bgr / bgr.mean(axis=0)
    b, g, r = n[:, 0], n[:, 1], n[:, 2]
    x = _bandpass(3 * r - 2 * g, fs)
    y = _bandpass(1.5 * r + g - 1.5 * b, fs)
    sy = y.std()
    return x - (x.std() / sy) * y if sy > 0 else x


class BpmEstimator:
    def __init__(self, window_s: float = 12.0, min_s: float = 6.0,
                 band_hz=(0.7, 3.0), detect_every: int = 15,
                 min_confidence: float = 4.0) -> None:
        # Band is 42-180 BPM. The plan's 0.8-2.0 Hz caps at 120 BPM, so a
        # post-exercise heart rate could never be found.
        self.min_confidence = min_confidence
        self.window_s = window_s
        self.min_s = min_s
        self.band = band_hz
        self.detect_every = detect_every
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.reset()

    def reset(self) -> None:
        self._t = collections.deque()
        self._g = collections.deque()
        self._n = 0
        self.face = None   # (x, y, w, h) in frame coordinates, or None
        self.rois = []     # [(x, y, w, h)] actually sampled
        self.bpm = None
        self.confidence = 0.0
        self.fill = 0.0    # 0..1, how much of min_s has been collected

    def _detect(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        scale = 320.0 / w if w > 320 else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = self._cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            self.face = tuple(int(v / scale) for v in (x, y, fw, fh))

    # Sample regions as fractions of the face box (x0, y0, x1, y1): forehead above
    # the eyebrows, and both cheeks below the eyes. Avoids eyes, brows and mouth.
    FACE_REGIONS = ((0.30, 0.06, 0.70, 0.20),
                    (0.15, 0.50, 0.38, 0.70),
                    (0.62, 0.50, 0.85, 0.70))

    def _rois(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if self.face is not None:
            x, y, fw, fh = self.face
            return [(x + int(x0 * fw), y + int(y0 * fh), int((x1 - x0) * fw), int((y1 - y0) * fh))
                    for x0, y0, x1, y1 in self.FACE_REGIONS]
        return [(w // 2 - w // 10, h // 3 - h // 12, w // 5, h // 6)]  # centre box guess

    def update(self, frame: np.ndarray, t: float) -> None:
        if self._n % self.detect_every == 0:
            self._detect(frame)
        self._n += 1
        self.rois = self._rois(frame)
        patches = [frame[y:y + rh, x:x + rw].reshape(-1, 3) for x, y, rw, rh in self.rois]
        patches = [p for p in patches if p.size]
        if not patches:
            return
        self._t.append(t)
        self._g.append(np.mean([p.mean(axis=0) for p in patches], axis=0))  # mean B, G, R
        while self._t and t - self._t[0] > self.window_s:
            self._t.popleft()
            self._g.popleft()
        span = self._t[-1] - self._t[0]
        self.fill = min(1.0, span / self.min_s)
        if span >= self.min_s and self._n % 5 == 0:
            self._estimate()

    def _estimate(self) -> None:
        t = np.asarray(self._t)
        bgr = np.asarray(self._g)
        fs = 30.0
        tu = np.arange(t[0], t[-1], 1.0 / fs)
        if len(tu) < 32:
            return
        sig = chrom(np.stack([np.interp(tu, t, bgr[:, c]) for c in range(3)], axis=1), fs)
        sig *= np.hanning(len(sig))
        nfft = 8192
        spec = np.abs(np.fft.rfft(sig, nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        band = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if not band.any() or spec[band].sum() <= 0:
            return
        bf, bs = freqs[band], spec[band]
        k = int(np.argmax(bs))
        peak_f = bf[k]
        # Harmonic guard: the pulse waveform isn't a sine, so 2x the heart rate
        # also shows up. If there's a strong peak at half the frequency, prefer it.
        half = np.abs(bf - peak_f / 2) <= 0.05
        if half.any() and bs[half].max() >= 0.5 * bs[k]:
            k = int(np.flatnonzero(half)[np.argmax(bs[half])])
            peak_f = bf[k]
        self.confidence = float(bs[k] / bs.mean())
        if self.confidence < self.min_confidence:
            return  # keep the last good value; .reliable reports it as stale
        bpm = 60.0 * peak_f
        if self.bpm is None or abs(bpm - self.bpm) > 15:
            self.bpm = bpm  # big confident change (e.g. after exercise): jump, don't crawl
        else:
            self.bpm = 0.7 * self.bpm + 0.3 * bpm

    @property
    def reliable(self) -> bool:
        return self.bpm is not None and self.confidence >= self.min_confidence
