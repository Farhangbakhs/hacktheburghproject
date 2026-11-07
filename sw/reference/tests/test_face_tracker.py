"""Tests for face_tracker.py.

Real-face tests only run if tests/data/face.jpg exists (any frontal face photo
you have the right to use; it is not shipped). Without models/ files the
YuNet+LBF tests skip; fallback tests use an empty temp models dir (Haar).
Run with -s to see the timing benchmark.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_tracker import HOLE_PREFIX, MODELS_DIR, FaceResult, FaceTracker, draw  # noqa: E402

FACE_JPG = Path(__file__).parent / "data" / "face.jpg"
HAVE_MODELS = (MODELS_DIR / "face_detection_yunet_2023mar.onnx").is_file() and (
    MODELS_DIR / "lbfmodel.yaml").is_file()
REQUIRED_REGIONS = ("face_skin", "forehead", "left_cheek", "right_cheek")


@pytest.fixture(scope="module")
def tracker_full():
    if not HAVE_MODELS:
        pytest.skip("models missing: run models/download_models.py")
    return FaceTracker()


@pytest.fixture(scope="module")
def face_frame():
    if not FACE_JPG.is_file():
        pytest.skip(f"no real face image at {FACE_JPG}")
    img = cv2.imread(str(FACE_JPG))
    assert img is not None
    # letterbox to 640x480 like the webcam
    s = min(640 / img.shape[1], 480 / img.shape[0])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = np.zeros((480, 640, 3), np.uint8)
    y0, x0 = (480 - img.shape[0]) // 2, (640 - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return out


def noisy_frames(img: np.ndarray, n: int, sigma: float = 3.0, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.clip(img.astype(np.int16) + rng.normal(0, sigma, img.shape).astype(np.int16),
                    0, 255).astype(np.uint8) for _ in range(n)]


def synthetic_no_face(h: int = 480, w: int = 640) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // w), (yy * 255 // h), np.full_like(xx, 90)], -1).astype(np.uint8)
    cv2.rectangle(img, (100, 100), (220, 300), (30, 200, 30), -1)
    cv2.circle(img, (450, 240), 60, (200, 60, 200), -1)
    return img


def synthetic_cartoon_face(h: int = 480, w: int = 640) -> np.ndarray:
    """A crude skin-coloured face; detectors may or may not fire on it."""
    img = np.full((h, w, 3), (90, 110, 70), np.uint8)
    c = (w // 2, h // 2)
    cv2.ellipse(img, c, (95, 125), 0, 0, 360, (120, 150, 200), -1)
    for dx in (-38, 38):
        cv2.ellipse(img, (c[0] + dx, c[1] - 25), (18, 9), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(img, (c[0] + dx, c[1] - 25), 6, (40, 30, 20), -1)
        cv2.line(img, (c[0] + dx - 22, c[1] - 48), (c[0] + dx + 22, c[1] - 50), (40, 40, 50), 5)
    cv2.line(img, (c[0], c[1] - 15), (c[0] - 8, c[1] + 25), (90, 110, 160), 3)
    cv2.ellipse(img, (c[0], c[1] + 55), (32, 12), 0, 0, 360, (80, 80, 170), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def check_result(res: FaceResult, shape_hw: tuple[int, int]) -> None:
    H, W = shape_hw
    assert res.found
    assert res.box is not None and res.box.shape == (4,)
    assert res.landmarks is not None and res.landmarks.dtype == np.float32
    assert res.landmarks.ndim == 2 and res.landmarks.shape[1] == 2
    for name in REQUIRED_REGIONS:
        assert name in res.regions, name
    for name, poly in res.regions.items():
        assert poly.dtype == np.int32 and poly.ndim == 2 and poly.shape[1] == 2 and len(poly) >= 3, name
        assert poly[:, 0].min() >= 0 and poly[:, 0].max() < W, name
        assert poly[:, 1].min() >= 0 and poly[:, 1].max() < H, name
    m = res.skin_mask
    assert m is not None and m.dtype == np.float32 and m.shape == (H, W)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m.max() > 0.9, "mask should have a solid interior"
    assert np.count_nonzero((m > 0.05) & (m < 0.95)) > 50, "mask should be feathered"
    assert 0.0 <= res.confidence <= 1.0
    assert res.motion >= 0.0 and np.isfinite(res.motion)
    assert res.timing_ms > 0


# -- construction / fallback -----------------------------------------------------------
def test_backend_full(tracker_full):
    assert tracker_full.backend == "yunet+lbf"


def test_fallback_empty_models_dir(tmp_path):
    tr = FaceTracker(models_dir=tmp_path)
    assert tr.backend == "haar"
    res = tr.update(synthetic_no_face())
    assert not res.found and res.skin_mask is None


def test_fallback_yunet_only(tmp_path):
    src = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    if not src.is_file():
        pytest.skip("yunet model missing")
    (tmp_path / src.name).write_bytes(src.read_bytes())
    tr = FaceTracker(models_dir=tmp_path)
    assert tr.backend == "yunet"


def test_corrupt_model_falls_back(tmp_path):
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"not an onnx file")
    tr = FaceTracker(models_dir=tmp_path)
    assert tr.backend == "haar"
    assert tr.load_errors


# -- no face ------------------------------------------------------------------------------
@pytest.mark.parametrize("use_models", [True, False])
def test_no_face(use_models, tmp_path):
    if use_models and not HAVE_MODELS:
        pytest.skip("models missing")
    tr = FaceTracker(models_dir=None if use_models else tmp_path)
    for img in (synthetic_no_face(), np.zeros((480, 640, 3), np.uint8)):
        for _ in range(3):
            res = tr.update(img)
            assert isinstance(res, FaceResult)
            assert res.found is False
            assert res.skin_mask is None and res.landmarks is None and res.box is None
            assert res.regions == {}


def test_bad_input_never_raises(tmp_path):
    tr = FaceTracker(models_dir=tmp_path)
    for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8)):
        res = tr.update(bad)
        assert res.found is False
    if HAVE_MODELS:
        tr2 = FaceTracker()
        assert tr2.update(np.zeros((4, 4, 3), np.uint8)).found is False


def test_draw_no_face_is_noop():
    img = synthetic_no_face()
    before = img.copy()
    draw(img, FaceResult(found=False), show_mask=True)
    assert np.array_equal(img, before)


# -- geometry / mask on a synthetic face (mask construction path, all backends) -------
def test_mask_from_injected_landmarks(tmp_path):
    """Exercise geometry + mask code without relying on a detector firing."""
    img = synthetic_cartoon_face()
    tr = FaceTracker(models_dir=tmp_path)  # backend irrelevant for this path
    # mean-ish 68-point face scaled into the cartoon face
    lm = _template_68(center=(320, 255), scale=95.0)
    oval, holes, regs = tr._geometry_68(lm)
    regions = {"face_skin": oval, **regs, **{HOLE_PREFIX + k: v for k, v in holes.items()}}
    regions = {k: tr._clip_poly(v, 640, 480) for k, v in regions.items()}
    for use_colour in (False, True):
        tr.use_skin_colour = use_colour
        tr._skin_stats = None
        m = tr._build_mask(img, regions)
        assert m is not None and m.dtype == np.float32 and m.shape == (480, 640)
        assert 0 <= m.min() and m.max() <= 1 and m.max() > 0.9
        assert np.count_nonzero((m > 0.05) & (m < 0.95)) > 50
        # eye centres are holes, cheeks are skin, far background is zero
        for eye in (lm[36:42].mean(0), lm[42:48].mean(0)):
            assert m[int(eye[1]), int(eye[0])] < 0.2
        cheek = regions["left_cheek"].mean(0).astype(int)
        assert m[cheek[1], cheek[0]] > 0.8
        assert m[20, 20] == 0.0
    # 5-point geometry too
    k5 = np.array([lm[36:42].mean(0), lm[42:48].mean(0), lm[30], lm[48], lm[54]])
    oval, holes, regs = tr._geometry_5(k5, None)
    for name in ("forehead", "left_cheek", "right_cheek"):
        assert name in regs


def _template_68(center, scale) -> np.ndarray:
    """Approximate iBUG-68 mean shape (unit ~ half face width)."""
    jaw = [(np.cos(a), np.sin(a)) for a in np.linspace(np.pi + 0.3, -0.3, 17)]
    brow_l = [(-0.75 + 0.13 * i, -0.62 - 0.06 * np.sin(np.pi * i / 4)) for i in range(5)]
    brow_r = [(0.23 + 0.13 * i, -0.62 - 0.06 * np.sin(np.pi * i / 4)) for i in range(5)]
    nose_b = [(0.0, -0.45 + 0.13 * i) for i in range(4)]
    nose_l = [(-0.18 + 0.09 * i, 0.08 + 0.03 * (1 - abs(i - 2) / 2)) for i in range(5)]

    def eye(cx):
        return [(cx - 0.2, -0.35), (cx - 0.1, -0.42), (cx + 0.05, -0.42), (cx + 0.15, -0.35),
                (cx + 0.05, -0.30), (cx - 0.1, -0.30)]

    lips = [(0.32 * np.cos(a), 0.42 + 0.12 * np.sin(a)) for a in np.linspace(np.pi, -np.pi, 12, endpoint=False)]
    inner = [(0.2 * np.cos(a), 0.42 + 0.05 * np.sin(a)) for a in np.linspace(np.pi, -np.pi, 8, endpoint=False)]
    pts = np.array(jaw + brow_l + brow_r + nose_b + nose_l + eye(-0.42) + eye(0.47) + lips + inner)
    assert pts.shape == (68, 2)
    return (pts * scale + np.array(center)).astype(np.float32)


def test_one_euro_reduces_jitter():
    from face_tracker import OneEuro

    rng = np.random.default_rng(1)
    f = OneEuro()
    truth = np.full((68, 2), 200.0)
    raw = [truth + rng.normal(0, 1.0, truth.shape) for _ in range(120)]
    out = np.array([f(r, i / 30, 150.0) for i, r in enumerate(raw)])
    assert out[30:].std(0).mean() < 0.5 * np.array(raw)[30:].std(0).mean()
    # follows a step quickly enough
    f.reset()
    for i in range(30):
        f(truth, i / 30, 150.0)
    for i in range(30, 45):
        y = f(truth + 30.0, i / 30, 150.0)
    assert np.abs(y - (truth + 30.0)).mean() < 3.0


# -- real face (optional) -----------------------------------------------------------------------
def test_real_face_full(tracker_full, face_frame):
    tracker_full.reset()
    frames = noisy_frames(face_frame, 60)
    results = [tracker_full.update(f, t=i / 30) for i, f in enumerate(frames)]
    res = results[-1]
    check_result(res, face_frame.shape[:2])
    assert res.landmarks.shape == (68, 2)
    assert res.confidence > 0.5
    assert res.motion < 0.02  # static image
    lm = np.array([r.landmarks for r in results[20:]])
    jitter = float(lm.std(0).mean())
    print(f"\n[face_tracker] static-image landmark jitter {jitter:.3f} px")
    assert jitter < 1.5
    out = face_frame.copy()
    draw(out, res, show_mask=True)
    assert not np.array_equal(out, face_frame)


@pytest.mark.parametrize("which", ["yunet", "haar"])
def test_real_face_fallbacks(which, face_frame, tmp_path):
    if which == "yunet":
        src = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        if not src.is_file():
            pytest.skip("yunet model missing")
        (tmp_path / src.name).write_bytes(src.read_bytes())
    tr = FaceTracker(models_dir=tmp_path)
    assert tr.backend == which
    res = None
    for i, f in enumerate(noisy_frames(face_frame, 15)):
        res = tr.update(f, t=i / 30)
    check_result(res, face_frame.shape[:2])
    assert res.landmarks.shape == (5, 2)


def test_real_face_lost_when_removed(tracker_full, face_frame):
    tracker_full.reset()
    for i, f in enumerate(noisy_frames(face_frame, 10)):
        tracker_full.update(f, t=i / 30)
    blank = synthetic_no_face()
    found = [tracker_full.update(blank, t=(10 + i) / 30).found for i in range(15)]
    assert not found[-1], "track should end once the face is gone"


# -- benchmark --------------------------------------------------------------------------------
def test_benchmark(tracker_full, tmp_path):
    """Prints per-frame timing (yunet+lbf + fallbacks). Uses face.jpg if present."""
    img = None
    if FACE_JPG.is_file():
        img = cv2.imread(str(FACE_JPG))
        s = min(640 / img.shape[1], 480 / img.shape[0])
        img = cv2.resize(img, None, fx=s, fy=s)
        pad = np.zeros((480, 640, 3), np.uint8)
        pad[:img.shape[0], :img.shape[1]] = img
        img = pad
    else:
        img = synthetic_cartoon_face()
    frames = noisy_frames(img, 20)
    for name, tr in (("yunet+lbf", tracker_full), ("haar", FaceTracker(models_dir=tmp_path))):
        tr.reset()
        times, found = [], 0
        for i in range(150):
            t0 = time.perf_counter()
            r = tr.update(frames[i % 20], t=i / 30)
            times.append((time.perf_counter() - t0) * 1000)
            found += r.found
        tt = np.array(times[10:])
        print(f"\n[benchmark] {name:10s} 640x480: mean {tt.mean():.2f} ms, median {np.median(tt):.2f}, "
              f"p95 {np.percentile(tt, 95):.2f}, max {tt.max():.2f}, found {found}/150 "
              f"({'face.jpg' if FACE_JPG.is_file() else 'synthetic'})")
        assert tt.mean() < 50  # loose sanity bound; target is < 8 ms on a laptop


def test_download_script_verifies_present_models():
    """Idempotence check without network: present files pass size + hash."""
    sys.path.insert(0, str(MODELS_DIR))
    import download_models as dm

    present = [n for n in dm.MODELS if (MODELS_DIR / n).is_file()]
    if not present:
        pytest.skip("no models downloaded")
    for name in present:
        url, size, kind, digest = dm.MODELS[name]
        assert dm.is_valid(MODELS_DIR / name, size, kind, digest), name
