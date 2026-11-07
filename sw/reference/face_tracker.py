"""Face tracking + feathered facial skin mask for skin-only EVM / rPPG.

Replaces the Haar box + rectangles of face_roi.py with landmark-shaped regions
and a soft skin mask that excludes eyes, brows, nostrils, mouth, hair, glasses
frames and beard.

Backend choice (measured on this laptop, OpenCV 4.14 CPU, 640x480 frames):

* Detection: YuNet (cv2.FaceDetectorYN, opencv_zoo face_detection_yunet_2023mar,
  MIT). ~5.7 ms on a 320x240 copy of the frame, ~1.7 ms at 160x120, accurate
  (WIDER easy AP 0.88), robust to pose/lighting, and gives a score plus 5
  keypoints. Haar is ~10 ms and has many false positives/negatives; the
  ONNX-runtime detectors (RetinaFace, SCRFD) need extra deps or are heavier.
  We run it only every `detect_every` frames (and when motion is large or the
  track is lost), so it costs ~1.2 ms/frame on average.
* Dense landmarks: FacemarkLBF (cv2.face, 68 iBUG 300-W points, lbfmodel.yaml
  from the OpenCV facemark tutorial). ~3.5 ms per fit, loads in ~1.5 s.
  FacemarkKazemi needs a separately trained model with no official download;
  FacemarkAAM is slower and less accurate. opencv_zoo has no face-landmark
  model (checked Nov 2026: only YuNet/SFace/FER for faces), and 106-point /
  PFLD / face-mesh ONNX models would mean non-official weights of unclear
  licence. 68 points are enough to outline the oval, eyes, brows, nose and lips.
* Tracking: LBF is seeded with an "anchor" box (a YuNet box) translated by
  the face motion since the anchor frame, measured with pyramidal LK optical
  flow of 18 rigid landmarks from the anchor frame directly to the current frame
  at half resolution (~0.3 ms; no frame-to-frame integration, so no drift).
  Seeding LBF from its own previous output makes the box random-walk (measured
  2-3x the jitter); LBF also rounds its box to integers, so the seed has a 2 px
  deadband. YuNet re-detects every `detect_every` frames in a window around the
  face (fixed 160x160 input, ~1.5 ms) and only re-anchors if it disagrees by
  > 8 % of the face size, the flow is unreliable, or the anchor is > 3 s old.
  A miss retries full-frame next frame; two misses in a row end the track. Fast
  motion (> 6 % of face width per frame) forces an early detect. When nearly
  static (< 2.5 %/frame) LBF is refit only every 2nd frame and the last fit is
  moved by the flow in between.
* Jitter: a One Euro filter (Casiez et al. 2012) on all landmarks, sharing one
  speed estimate (normalised by face width) so the shape stays coherent: heavy
  smoothing when still, low lag when moving.
* Mask: face oval (jaw + elliptical forehead arc extrapolated above the brows)
  minus eye, brow, nostril and mouth holes, AND a per-frame adaptive skin-colour
  model (robust YCrCb statistics sampled from forehead + cheeks, EMA'd across
  frames, Mahalanobis gate in CrCb plus a luma floor). Then Gaussian feathering.
  Everything is computed inside the face ROI only. If the colour model keeps
  < 35 % of the oval (bad light, odd white balance) it is ignored for that frame.

Measured (640x480, this laptop, other jobs running): static face 5.5-7.5 ms
mean per frame (LBF ~3.5, mask ~1, flow ~0.4, detection ~0.5 amortised);
moving face 7-10 ms; no face ~6 ms (full-frame YuNet each frame). Run
tests/test_face_tracker.py -s for the benchmark.

Fallback chain if model files are missing (attribute `backend`):
    "yunet+lbf" -> "yunet" (5 keypoints, ellipse geometry) -> "haar" (box only,
    keypoints synthesised from box fractions). update() never raises.

Conventions: all coordinates are full-resolution pixels of the frame passed in
(not mirrored). Region names 'left_*' / 'right_*' mean IMAGE left/right (so
'left_cheek' is the subject's right cheek). Landmarks are (68,2) for yunet+lbf,
(5,2) for the fallbacks (YuNet keypoint order: image-left eye, image-right eye,
nose tip, image-left mouth corner, image-right mouth corner).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
LBF_FILE = "lbfmodel.yaml"

# iBUG 68-point index groups
JAW = list(range(0, 17))
BROW_L = list(range(17, 22))   # image-left brow (subject's right)
BROW_R = list(range(22, 27))
NOSE_BRIDGE = list(range(27, 31))
NOSE_BASE = list(range(31, 36))
EYE_L = list(range(36, 42))
EYE_R = list(range(42, 48))
LIPS_OUTER = list(range(48, 60))

# well-textured, rigid-ish points for LK flow: brows, nose, eye corners, mouth corners
FLOW_POINTS = [17, 19, 21, 22, 24, 26, 27, 29, 30, 31, 33, 35, 36, 39, 42, 45, 48, 54]

HOLE_PREFIX = "hole_"
SEED_DEADBAND = 2.0  # px; LBF seed box hysteresis
REANCHOR_FRAC = 0.08  # re-anchor when a detection disagrees by > this x face width
REANCHOR_FRAMES = 90  # ... or the anchor frame is older than this (appearance change)
ROI_DET = 160  # YuNet input size when re-detecting around the tracked face
SLOW_MOTION = 0.025  # below this, LBF is refit only every 2nd frame (flow in between)
FAST_MOTION = 0.06  # mean landmark step / face width per frame that forces a re-detect
MIN_SKIN_KEEP = 0.35  # below this fraction of the oval, ignore the colour model


@dataclass
class FaceResult:
    found: bool
    box: np.ndarray | None = None          # float (x, y, w, h), full-res pixels
    landmarks: np.ndarray | None = None    # (N, 2) float32 full-res pixels (smoothed)
    # name -> int32 polygon (K, 2). 'face_skin' is the outer oval; the holes cut
    # out of it (eyes, brows, nostrils, mouth) are given as 'hole_*' polygons.
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    skin_mask: np.ndarray | None = None    # float32 HxW in [0, 1], feathered
    motion: float = 0.0                    # mean landmark displacement / face width
    confidence: float = 0.0                # 0..1
    timing_ms: float = 0.0

    def mask_resized(self, shape_hw: tuple[int, int]) -> np.ndarray | None:
        """skin_mask resampled to another size (e.g. a pyramid level), float32 [0,1]."""
        if self.skin_mask is None:
            return None
        h, w = shape_hw
        if self.skin_mask.shape[:2] == (h, w):
            return self.skin_mask
        return cv2.resize(self.skin_mask, (w, h), interpolation=cv2.INTER_AREA)


class OneEuro:
    """One Euro filter over an (N, 2) point set with a shared, scale-normalised speed."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 4.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _alpha(dt: float, fc: float) -> float:
        tau = 1.0 / (2 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float, scale: float) -> np.ndarray:
        x = x.astype(np.float64)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = float(np.clip(t - self.t, 1e-3, 0.25))
        self.t = t
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        speed = float(np.linalg.norm(self.dx, axis=1).mean()) / max(scale, 1.0)
        a = self._alpha(dt, self.min_cutoff + self.beta * speed)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _ellipse(center, axes, angle_deg: float, n: int = 20) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    ex, ey = axes[0] * np.cos(th), axes[1] * np.sin(th)
    return np.stack([center[0] + ex * ca - ey * sa, center[1] + ex * sa + ey * ca], 1)


def _shrink(poly: np.ndarray, f: float) -> np.ndarray:
    c = poly.mean(0)
    return c + (poly - c) * f


class FaceTracker:
    def __init__(self, models_dir: Path | None = None, detect_every: int = 5,
                 feather_px: int = 9, use_skin_colour: bool = True,
                 detect_width: int = 320, max_missed_detections: int = 2):
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self.detect_every = max(1, int(detect_every))
        self.feather_px = max(0, int(feather_px))
        self.use_skin_colour = use_skin_colour
        self.detect_width = detect_width
        self.max_missed = max_missed_detections
        self.yunet = None
        self.yunet_roi = None
        self.lbf = None
        self.haar = None
        self.backend = "none"
        self.load_errors: list[str] = []
        self._load_models()
        self._euro = OneEuro()
        self._k3 = np.ones((3, 3), np.uint8)
        self._lk_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        self._last_error: str | None = None
        self.reset()

    # -- setup --------------------------------------------------------------
    def _load_models(self) -> None:
        yunet_path = self.models_dir / YUNET_FILE
        lbf_path = self.models_dir / LBF_FILE
        if yunet_path.is_file():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 240), 0.6, 0.3, 50)
                self.yunet_roi = cv2.FaceDetectorYN.create(str(yunet_path), "", (ROI_DET, ROI_DET),
                                                           0.6, 0.3, 10)
                self._full_size = (320, 240)
            except Exception as e:  # corrupt file / old OpenCV
                self.load_errors.append(f"yunet: {e}")
        else:
            self.load_errors.append(f"missing {yunet_path}")
        if self.yunet is not None:
            if lbf_path.is_file():
                try:
                    lbf = cv2.face.createFacemarkLBF()
                    lbf.loadModel(str(lbf_path))
                    self.lbf = lbf
                except Exception as e:
                    self.load_errors.append(f"lbf: {e}")
            else:
                self.load_errors.append(f"missing {lbf_path}")
            self.backend = "yunet+lbf" if self.lbf is not None else "yunet"
        else:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            haar = cv2.CascadeClassifier(path)
            if haar.empty():
                self.load_errors.append(f"haar: could not load {path}")
                self.backend = "none"
            else:
                self.haar = haar
                self.backend = "haar"
        if self.load_errors:
            print(f"[face_tracker] backend={self.backend} ({'; '.join(self.load_errors)})")

    def reset(self) -> None:
        self._euro.reset()
        self._frame_i = 0
        self._tracking = False
        self._misses = 0
        self._force_detect = True
        self._box: np.ndarray | None = None
        self._smoothed: np.ndarray | None = None
        self._score = 0.0
        self._anchor: np.ndarray | None = None   # LBF seed box at the last detection
        self._anchor_c = np.zeros(2)              # smoothed landmark centre at that time
        self._seed: np.ndarray | None = None      # last LBF seed box actually used
        self._anchor_age = 0
        self._last_raw: np.ndarray | None = None  # last LBF output and the flow shift then
        self._last_shift = np.zeros(2)
        self._last_motion = 1.0
        self._anchor_grey: np.ndarray | None = None  # half-res grey of the anchor frame
        self._anchor_pts: np.ndarray | None = None   # inner landmarks there (half-res)
        self._skin_stats: tuple[np.ndarray, np.ndarray, float, float] | None = None

    # -- public -------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray, t: float | None = None) -> FaceResult:
        """Track the face in one BGR frame. `t` (seconds) defaults to perf_counter()."""
        t0 = time.perf_counter()
        try:
            res = self._update(frame_bgr, t0 if t is None else t)
        except Exception as e:  # never take the dashboard down
            msg = f"{type(e).__name__}: {e}"
            if msg != self._last_error:
                print(f"[face_tracker] update failed: {msg}")
                self._last_error = msg
            self.reset()
            res = FaceResult(found=False)
        res.timing_ms = (time.perf_counter() - t0) * 1000
        return res

    # -- detection ------------------------------------------------------------
    def _detect(self, frame: np.ndarray, around: np.ndarray | None = None
                ) -> list[tuple[np.ndarray, np.ndarray | None, float]]:
        """Faces as (box xywh, 5 keypoints or None, score) in full-res pixels.

        With `around` (current track box) YuNet only searches a square window of
        2x the box, warped to a fixed ROI_DET x ROI_DET input (~1.5 ms instead of
        ~6 ms). Separate detector instances keep fixed input sizes: changing
        setInputSize every call forces a slow network reshape.
        """
        H, W = frame.shape[:2]
        out = []
        if self.yunet is not None:
            if around is not None and self.yunet_roi is not None:
                cx, cy = around[0] + around[2] / 2, around[1] + around[3] / 2
                side = 2.0 * max(around[2], around[3], 16.0)
                k = ROI_DET / side
                M = np.array([[k, 0, -k * (cx - side / 2)], [0, k, -k * (cy - side / 2)]])
                small = cv2.warpAffine(frame, M, (ROI_DET, ROI_DET), flags=cv2.INTER_AREA,
                                       borderMode=cv2.BORDER_CONSTANT)
                det, s, off = self.yunet_roi, k, np.array([cx - side / 2, cy - side / 2])
            else:
                s = self.detect_width / W
                small = cv2.resize(frame, (self.detect_width, int(round(H * s))),
                                   interpolation=cv2.INTER_AREA)
                det, off = self.yunet, np.zeros(2)
                if self._full_size != small.shape[1::-1]:
                    self._full_size = small.shape[1::-1]
                    det.setInputSize(self._full_size)
            _, faces = det.detect(small)
            if faces is not None:
                for f in faces:
                    box = f[:4].astype(np.float64) / s
                    box[:2] += off
                    kps = f[4:14].reshape(5, 2).astype(np.float64) / s + off
                    out.append((box, kps, float(f[14])))
        elif self.haar is not None:
            s = min(1.0, self.detect_width / W)
            small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
            grey = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            for b in self.haar.detectMultiScale(grey, 1.15, 5, minSize=(30, 30)):
                out.append((b.astype(np.float64) / s, None, 0.8))
        return out

    def _choose(self, dets):
        if not dets:
            return None
        if self._tracking and self._box is not None:
            best = max(dets, key=lambda d: _iou(d[0], self._box))
            if _iou(best[0], self._box) > 0.2:
                return best
        return max(dets, key=lambda d: d[0][2] * d[0][3])

    # -- LBF helpers ------------------------------------------------------------
    @staticmethod
    def _lm_frame(lm: np.ndarray) -> tuple[np.ndarray, float]:
        c = lm.mean(0)
        s = float(np.sqrt(((lm - c) ** 2).sum(1).mean()))
        return c, max(s, 1.0)

    def _flow_shift(self, half_grey: np.ndarray) -> tuple[np.ndarray, bool]:
        """Face translation (full-res px) from the anchor frame to now, and whether
        the flow was reliable (else falls back to the smoothed-landmark shift)."""
        if self._anchor_grey is None or self._anchor_grey.shape != half_grey.shape:
            return self._lm_shift(), False
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._anchor_grey, half_grey, self._anchor_pts, None,
                                             winSize=(15, 15), maxLevel=2, criteria=self._lk_crit)
        good = st.reshape(-1) == 1
        if good.sum() < 0.6 * len(good):
            return self._lm_shift(), False
        d = (p1 - self._anchor_pts).reshape(-1, 2)[good]
        med = np.median(d, axis=0)
        spread = float(np.median(np.abs(d - med).sum(1)))
        return med * 2.0, spread < 1.5  # half-res px; large spread = rotation/deformation

    def _lm_shift(self) -> np.ndarray:
        if self._smoothed is None:
            return np.zeros(2)
        return self._smoothed.mean(0) - self._anchor_c

    @staticmethod
    def _shifted(box: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.array([box[0] + d[0], box[1] + d[1], box[2], box[3]])

    def _fit_lbf(self, grey: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        H, W = grey.shape[:2]
        b = box.copy()
        if b[2] < 20 or b[3] < 20 or b[0] > W or b[1] > H or b[0] + b[2] < 0 or b[1] + b[3] < 0:
            return None
        ok, lms = self.lbf.fit(grey, np.array([b], dtype=np.float32))
        if not ok or len(lms) == 0:
            return None
        lm = lms[0].reshape(-1, 2).astype(np.float64)
        return lm if np.all(np.isfinite(lm)) else None

    # -- main step --------------------------------------------------------------
    def _lost(self) -> FaceResult:
        self._tracking = False
        self._box = None
        self._anchor = None
        self._seed = None
        self._last_raw = None
        self._last_motion = 1.0
        self._anchor_grey = None
        self._smoothed = None
        self._euro.reset()
        self._skin_stats = None
        self._force_detect = True
        return FaceResult(found=False)

    def _update(self, frame: np.ndarray, t: float) -> FaceResult:
        if self.backend == "none" or frame is None or frame.ndim != 3 or frame.size == 0:
            return FaceResult(found=False)
        H, W = frame.shape[:2]
        self._frame_i += 1
        every = 1 if self.backend == "yunet" else self.detect_every
        det = None
        if (not self._tracking or self._force_detect or self._frame_i % every == 0):
            self._force_detect = False
            around = self._box if (self._tracking and self._misses == 0) else None
            det = self._choose(self._detect(frame, around))
            if det is None:
                self._misses += 1
                self._force_detect = True  # retry next frame, full frame
                if not self._tracking or self._misses >= self.max_missed:
                    return self._lost()
            else:
                self._misses = 0
                self._score = 0.6 * self._score + 0.4 * det[2] if self._tracking else det[2]
                self._tracking = True

        reanchor = False
        if self.backend == "yunet+lbf":
            # LBF seed = "anchor" box (a YuNet box) translated by the face motion
            # since the anchor frame, measured by pyramidal LK flow of the inner
            # landmarks from the anchor frame straight to this one (no
            # frame-to-frame integration, so no random walk). Detections only
            # re-anchor when they disagree by > REANCHOR_FRAC of the face size, the
            # flow degrades, or the anchor is old: YuNet's box wobbles by a few px
            # frame to frame and every seed change moves LBF's output.
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            half = cv2.pyrDown(grey)
            shift, flow_ok = (np.zeros(2), False) if self._anchor is None else self._flow_shift(half)
            if self._anchor is not None and self._smoothed is not None:
                lm_shift = self._lm_shift()
                if not flow_ok or np.abs(shift - lm_shift).max() > 0.15 * self._anchor[2]:
                    shift, flow_ok = lm_shift, False   # flow unreliable: follow LBF, re-detect
                    self._force_detect = True
            self._anchor_age += 1
            if det is not None:
                cur = None if self._anchor is None else self._shifted(self._anchor, shift)
                if cur is None or _iou(cur, det[0]) < 0.5:  # (re)acquired or lost lock
                    self._anchor = det[0].copy()
                    self._smoothed = None
                    self._euro.reset()
                    reanchor = True
                elif (not flow_ok or self._anchor_age > REANCHOR_FRAMES
                      or np.abs(det[0] - cur).max() > REANCHOR_FRAC * cur[2]):
                    self._anchor = 0.5 * cur + 0.5 * det[0]
                    reanchor = True
                if reanchor:
                    shift = np.zeros(2)
            seed = self._shifted(self._anchor, shift)
            # LBF rounds its box to integers, so sub-pixel seed wobble makes the
            # landmarks jump by 1-2 px. Keep the previous seed unless it moved
            # by more than SEED_DEADBAND px.
            if self._seed is not None and np.abs(seed - self._seed).max() < SEED_DEADBAND:
                seed = self._seed
            self._seed = seed
            # Near-static face: refit LBF only every 2nd frame and move the last fit
            # by the optical-flow delta in between (halves the dominant cost and
            # adds no LBF noise on the skipped frames).
            if (det is None and flow_ok and self._last_raw is not None
                    and self._last_motion < SLOW_MOTION and self._frame_i % 2 == 1):
                raw = self._last_raw + (shift - self._last_shift)
            else:
                raw = self._fit_lbf(grey, seed)
                if raw is None:
                    return self._lost()
                self._last_raw, self._last_shift = raw, shift
            self._box = seed
        elif det is not None:
            raw = det[1] if det[1] is not None else self._kps_from_box(det[0])
            self._box = det[0]
        else:  # haar between detections: hold
            raw = self._smoothed

        c, s = self._lm_frame(raw)
        face_w = float(np.ptp(raw[:, 0])) if len(raw) > 5 else float(np.linalg.norm(raw[1] - raw[0]) * 2.2)
        sm = self._euro(raw, t, max(face_w, 1.0))
        motion = 0.0
        if self._smoothed is not None and self._smoothed.shape == sm.shape:
            motion = float(np.linalg.norm(sm - self._smoothed, axis=1).mean()) / max(face_w, 1.0)
        self._smoothed = sm
        self._last_motion = motion
        if motion > FAST_MOTION:
            self._force_detect = True
        if reanchor:
            self._anchor_age = 0
            self._anchor_c = sm.mean(0)
            self._anchor_grey = half
            self._anchor_pts = (sm[FLOW_POINTS] * 0.5).astype(np.float32).reshape(-1, 1, 2)
        lmf = sm.astype(np.float32)

        # sanity: face mostly off-frame or collapsed -> lost
        if face_w < 12 or c[0] < -0.2 * W or c[0] > 1.2 * W or c[1] < -0.2 * H or c[1] > 1.2 * H:
            return self._lost()

        if len(lmf) == 68:
            oval, holes, regs = self._geometry_68(lmf)
        else:
            oval, holes, regs = self._geometry_5(lmf, self._box)
        regions = {"face_skin": oval, **regs, **{HOLE_PREFIX + k: v for k, v in holes.items()}}
        regions = {k: self._clip_poly(v, W, H) for k, v in regions.items()}
        mask = self._build_mask(frame, regions)
        conf = float(np.clip(self._score * (0.75 if self._misses else 1.0), 0, 1))
        return FaceResult(found=True, box=self._box.astype(np.float64).copy(), landmarks=lmf,
                          regions=regions, skin_mask=mask, motion=motion, confidence=conf)

    # -- geometry ------------------------------------------------------------------
    @staticmethod
    def _clip_poly(p: np.ndarray, W: int, H: int) -> np.ndarray:
        q = np.round(p).astype(np.int32)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        return q

    @staticmethod
    def _kps_from_box(box: np.ndarray) -> np.ndarray:
        x, y, w, h = box
        f = np.array([[0.32, 0.40], [0.68, 0.40], [0.50, 0.60], [0.36, 0.78], [0.64, 0.78]])
        return np.stack([x + f[:, 0] * w, y + f[:, 1] * h], 1)

    @staticmethod
    def _geometry_68(p: np.ndarray):
        e = p[45] - p[36]
        iod = float(np.linalg.norm(e))
        ex = e / max(iod, 1e-6)
        up = np.array([ex[1], -ex[0]])          # perpendicular to the eye line, pointing up
        angle = float(np.degrees(np.arctan2(ex[1], ex[0])))
        brow_mid = (p[19] + p[24]) / 2
        third = abs(float(np.dot(brow_mid - p[33], up)))  # brow -> nose base height

        # oval: jaw + elliptical forehead arc from p16 over the brows to p0
        C = (p[0] + p[16]) / 2
        a = 0.97 * float(np.linalg.norm(p[16] - p[0])) / 2
        b = max(float(np.dot(brow_mid - C, up)) + 0.85 * third, 1.0)
        th = np.linspace(0, np.pi, 15)[1:-1]
        arc = C + np.outer(a * np.cos(th), ex) + np.outer(b * np.sin(th), up)
        oval = _shrink(np.vstack([p[JAW], arc]), 0.97)

        holes = {}
        for name, idx in (("left_eye", EYE_L), ("right_eye", EYE_R)):
            ew = float(np.linalg.norm(p[idx[3]] - p[idx[0]]))
            holes[name] = _ellipse(p[idx].mean(0), (0.85 * ew, 0.55 * ew), angle)
        t_up, t_dn = 0.10 * iod, 0.09 * iod
        for name, idx in (("left_brow", BROW_L), ("right_brow", BROW_R)):
            q = p[idx]
            q = np.vstack([q[0] - 0.15 * (q[-1] - q[0]), q, q[-1] + 0.15 * (q[-1] - q[0])])
            holes[name] = np.vstack([q + up * t_up, (q - up * t_dn)[::-1]])
        nose_len = float(np.linalg.norm(p[33] - p[27]))
        holes["nostrils"] = _ellipse(p[33] + up * 0.08 * nose_len,
                                     (0.62 * float(np.linalg.norm(p[35] - p[31])), 0.22 * nose_len), angle)
        holes["mouth"] = _shrink(p[LIPS_OUTER], 1.22)

        brow_top = p[18:26] + up * (t_up + 0.08 * iod)
        forehead = np.vstack([brow_top, (brow_top + up * 0.5 * third)[::-1]])
        regs = {
            "forehead": forehead,
            "left_cheek": _shrink(p[[1, 2, 3, 4, 48, 31, 40, 41]], 0.75),
            "right_cheek": _shrink(p[[15, 14, 13, 12, 54, 35, 47, 46]], 0.75),
        }
        return oval, holes, regs

    @staticmethod
    def _geometry_5(k: np.ndarray, box: np.ndarray | None):
        le, re, nose, lm, rm = k
        e = re - le
        iod = max(float(np.linalg.norm(e)), 1.0)
        ex = e / iod
        up = np.array([ex[1], -ex[0]])
        angle = float(np.degrees(np.arctan2(ex[1], ex[0])))
        eye_c, mouth_c = (le + re) / 2, (lm + rm) / 2
        em = max(float(np.dot(eye_c - mouth_c, up)), 0.5 * iod)  # eye-mouth height
        center = eye_c - up * 0.15 * em   # forehead top ~1.4 em above eyes, chin ~1.7 em below
        oval = _ellipse(center, (0.95 * iod, 1.55 * em), angle, 32)
        holes = {
            "left_eye": _ellipse(le, (0.32 * iod, 0.2 * iod), angle),
            "right_eye": _ellipse(re, (0.32 * iod, 0.2 * iod), angle),
            "left_brow": _ellipse(le + up * 0.38 * iod, (0.36 * iod, 0.1 * iod), angle),
            "right_brow": _ellipse(re + up * 0.38 * iod, (0.36 * iod, 0.1 * iod), angle),
            "nostrils": _ellipse(nose - up * 0.12 * em, (0.28 * iod, 0.1 * em), angle),
            "mouth": _ellipse(mouth_c, (0.62 * float(np.linalg.norm(rm - lm)) + 0.05 * iod, 0.28 * em), angle),
        }
        fc = eye_c + up * 0.85 * iod
        regs = {
            "forehead": _ellipse(fc, (0.55 * iod, 0.22 * iod), angle, 12),
            "left_cheek": _ellipse(le - up * 0.62 * em - ex * 0.08 * iod, (0.22 * iod, 0.2 * em), angle, 12),
            "right_cheek": _ellipse(re - up * 0.62 * em + ex * 0.08 * iod, (0.22 * iod, 0.2 * em), angle, 12),
        }
        return oval, holes, regs

    # -- mask ------------------------------------------------------------------------
    def _build_mask(self, frame: np.ndarray, regions: dict[str, np.ndarray]) -> np.ndarray | None:
        """Feathered float32 mask. Built at half resolution inside the face ROI
        (polygons drawn with sub-pixel shift=1), then upsampled: the Gaussian
        feather hides the resampling and it is ~3x cheaper than full res."""
        H, W = frame.shape[:2]
        oval = regions["face_skin"]
        pad = 2 * self.feather_px + 2
        x0, y0 = np.maximum(oval.min(0) - pad, 0)
        x1, y1 = np.minimum(oval.max(0) + pad + 1, [W, H])
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        off = np.array([x0, y0], np.int32)
        rw, rh = int(x1 - x0), int(y1 - y0)
        hw, hh = (rw + 1) // 2, (rh + 1) // 2
        geo = np.zeros((hh, hw), np.uint8)
        cv2.fillPoly(geo, [oval - off], 255, cv2.LINE_8, 1)
        cv2.fillPoly(geo, [v - off for k, v in regions.items() if k.startswith(HOLE_PREFIX)], 0,
                     cv2.LINE_8, 1)
        bin_mask = geo
        if self.use_skin_colour:
            small = cv2.resize(frame[y0:y1, x0:x1], (hw, hh), interpolation=cv2.INTER_AREA)
            skin = self._skin_colour(small, regions, off, geo)
            if skin is not None:
                bin_mask = cv2.bitwise_and(geo, skin)
        roi = bin_mask.astype(np.float32) * (1.0 / 255)
        if self.feather_px > 0:
            k = self.feather_px | 1
            roi = cv2.GaussianBlur(roi, (k, k), self.feather_px / 4.0)
        full = np.zeros((H, W), np.float32)
        full[y0:y1, x0:x1] = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_LINEAR)
        return full

    def _update_skin_stats(self, ycc, geo, regions, off) -> bool:
        sample = np.zeros(geo.shape, np.uint8)
        cv2.fillPoly(sample, [regions[k] - off for k in ("forehead", "left_cheek", "right_cheek")],
                     255, cv2.LINE_8, 1)
        sample &= geo
        px = ycc[sample > 0].astype(np.float32)
        if len(px) > 1500:
            px = px[:: len(px) // 1000]
        if len(px) < 50:
            return self._skin_stats is not None
        med = np.median(px, 0)
        mad = np.median(np.abs(px - med), 0) * 1.4826 + 1.0
        inl = px[np.all(np.abs(px - med) < 2.5 * mad, 1)]
        if len(inl) < 30:
            return self._skin_stats is not None
        mu = inl[:, 1:].mean(0)
        cov = np.cov(inl[:, 1:].T) + np.eye(2) * 4.0
        y_mu, y_sd = float(inl[:, 0].mean()), float(inl[:, 0].std()) + 2.0
        if self._skin_stats is not None:  # EMA across refreshes for stability
            m0, c0, ym0, ys0 = self._skin_stats
            a = 0.4
            mu, cov = (1 - a) * m0 + a * mu, (1 - a) * c0 + a * cov
            y_mu, y_sd = (1 - a) * ym0 + a * y_mu, (1 - a) * ys0 + a * y_sd
        self._skin_stats = (mu, cov, y_mu, y_sd)
        return True

    def _skin_colour(self, small_bgr: np.ndarray, regions, off, geo) -> np.ndarray | None:
        """Binary (uint8 0/255, half-res ROI) adaptive skin-colour mask, or None if unreliable."""
        ycc = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb)
        # colour statistics: refresh every 3rd frame (EMA'd), classify every frame
        if self._skin_stats is None or self._frame_i % 3 == 0:
            if not self._update_skin_stats(ycc, geo, regions, off):
                return None
        mu, cov, y_mu, y_sd = self._skin_stats
        inv = np.linalg.inv(cov).astype(np.float32)
        d_cr = ycc[..., 1].astype(np.float32) - np.float32(mu[0])
        d_cb = ycc[..., 2].astype(np.float32) - np.float32(mu[1])
        d2 = inv[0, 0] * d_cr * d_cr + 2 * inv[0, 1] * d_cr * d_cb + inv[1, 1] * d_cb * d_cb
        y_floor = max(y_mu - 3.0 * y_sd, 0.35 * y_mu)
        skin = ((d2 < 12.0) & (ycc[..., 0] > y_floor)).astype(np.uint8) * 255
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, self._k3)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, self._k3)
        n_geo = cv2.countNonZero(geo)
        if n_geo == 0 or cv2.countNonZero(cv2.bitwise_and(skin, geo)) < MIN_SKIN_KEEP * n_geo:
            return None
        return skin


# -- drawing ---------------------------------------------------------------------------
_COLOURS = {"face_skin": (90, 220, 90), "forehead": (255, 200, 0), "left_cheek": (0, 190, 255),
            "right_cheek": (0, 190, 255)}


def draw(frame_bgr: np.ndarray, result: FaceResult, show_mask: bool = False) -> None:
    """In-place overlay: optional mask tint, region polygons, landmark dots."""
    if not result.found:
        return
    if show_mask and result.skin_mask is not None:
        m = result.skin_mask
        ys, xs = np.nonzero(m > 0.01)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
            a = (m[y0:y1, x0:x1] * 0.45)[..., None]
            tint = np.array([60, 255, 60], np.float32)
            frame_bgr[y0:y1, x0:x1] = (roi * (1 - a) + tint * a).astype(np.uint8)
    for name, poly in result.regions.items():
        col = (120, 120, 120) if name.startswith(HOLE_PREFIX) else _COLOURS.get(name, (200, 200, 200))
        cv2.polylines(frame_bgr, [poly.reshape(-1, 1, 2)], True, col, 1, cv2.LINE_AA)
    if result.landmarks is not None:
        for x, y in result.landmarks:
            cv2.circle(frame_bgr, (int(round(x)), int(round(y))), 1, (0, 255, 255), -1, cv2.LINE_AA)
