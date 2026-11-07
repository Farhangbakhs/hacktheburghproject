"""Face skin ROI tracking (forehead + both cheeks) with OpenCV's bundled Haar cascade.

Cheap and dependency-free; detection runs every `detect_every` frames on a
downscaled grey image and the box is exponentially smoothed so the ROI does not
jitter (jitter injects noise into the pulse trace). Reused by bpm.py (step 5),
where a MediaPipe landmark ROI may replace it.
"""
from __future__ import annotations

import cv2
import numpy as np

# Sub-boxes as fractions of the face box (x0, y0, x1, y1): forehead, left cheek,
# right cheek. Cheeks start below typical glasses frames.
SKIN_PATCHES = (
    (0.28, 0.06, 0.72, 0.24),
    (0.14, 0.55, 0.38, 0.75),
    (0.62, 0.55, 0.86, 0.75),
)


class FaceROI:
    def __init__(self, detect_every: int = 5, smooth: float = 0.3, detect_width: int = 320,
                 max_missed: int = 30):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"could not load Haar cascade from {path}")
        self.detect_every = detect_every
        self.smooth = smooth
        self.detect_width = detect_width
        self.max_missed = max_missed
        self.box: np.ndarray | None = None  # float (x, y, w, h) in full-frame pixels
        self._frame_i = 0
        self._missed = 0

    @property
    def found(self) -> bool:
        return self.box is not None

    def update(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Track the largest face; returns smoothed (x, y, w, h) or None."""
        self._frame_i += 1
        if self._frame_i % self.detect_every != 1 and self.detect_every > 1 and self.box is not None:
            return self.box
        h, w = frame_bgr.shape[:2]
        s = min(1.0, self.detect_width / w)
        grey = cv2.cvtColor(cv2.resize(frame_bgr, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)
        faces = self.cascade.detectMultiScale(grey, 1.15, 5, minSize=(40, 40))
        if len(faces) == 0:
            self._missed += 1
            if self._missed > self.max_missed // max(1, self.detect_every):
                self.box = None
            return self.box
        self._missed = 0
        biggest = max(faces, key=lambda f: f[2] * f[3]).astype(np.float64) / s
        if self.box is None:
            self.box = biggest
        else:
            self.box = (1 - self.smooth) * self.box + self.smooth * biggest
        return self.box

    def patches(self, scale: float = 1.0) -> list[tuple[int, int, int, int]]:
        """Skin patch rectangles (x0, y0, x1, y1), scaled e.g. to pyramid level."""
        if self.box is None:
            return []
        x, y, w, h = self.box * scale
        return [(int(x + fx0 * w), int(y + fy0 * h), int(x + fx1 * w), int(y + fy1 * h))
                for fx0, fy0, fx1, fy1 in SKIN_PATCHES]

    def mask(self, shape_hw: tuple[int, int], scale: float = 1.0) -> np.ndarray | None:
        """Boolean mask of skin patches at a given image size."""
        rects = self.patches(scale)
        if not rects:
            return None
        m = np.zeros(shape_hw, bool)
        H, W = shape_hw
        for x0, y0, x1, y1 in rects:
            m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        return m if m.any() else None
