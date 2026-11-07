"""Objective evaluation harness for EVM pulse-magnification variants.

Measures how well a processor (anything with the FloatEVM interface:
``name``, ``process(small_bgr) -> float32 signal``, ``reset()``,
``set_params()``, ``band_state()``) magnifies the facial pulse while keeping
noise, speckle and motion halos down. Runs the same pipeline as the dashboard::

    small    = downscale(frame, levels)
    sig      = proc.process(small)
    sig_full = upscale(sig, levels, frame.shape[:2])  [* mask_full]
    out      = add_signal(frame, sig_full)

This module does not touch the docs/PLAN.md section 3 contract; it only
measures processors.

CLI::

    # compare variants on a synthetic scenario (or "all" scenarios)
    python evm_eval.py compare --synthetic clean
    python evm_eval.py compare --synthetic all --variants contract
    # on a recorded / any video clip
    python evm_eval.py compare --clip recordings/me.npz --mask face
    # record 10 s lossless from the webcam into recordings/me.npz
    python evm_eval.py record --name me --seconds 10 --preview

Synthetic scenarios: clean, noisy, motion, drift, jpeg, dim.
Variants: 'contract' (evm_float.FloatEVM) plus evm_enhanced.PRESETS if present.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from scipy.signal import periodogram

from evm_common import (
    FS_DEFAULT,
    FilterParams,
    add_signal,
    downscale,
    pyramid_levels_for,
    upscale,
)
from evm_float import FloatEVM

HERE = Path(__file__).resolve().parent
RECORDINGS_DIR = HERE / "recordings"

# Analysis band for the pulse (Hz): 30-240 bpm.
PULSE_BAND = (0.5, 4.0)
# Half-width of the spectral window around f0 / 2*f0 (Hz); widened to the Hann
# main-lobe half-width 2/T for short clips so a pure tone is not split.
PEAK_HALF_WIDTH = 0.1
MAX_FIDELITY_LAG_S = 0.5

# Blood-volume-pulse signature of skin in *normalised* (AC/DC) R, G, B:
# Pbv = [0.33, 0.77, 0.53]  (de Haan & van Leest 2014, "Improved motion
# robustness of remote-PPG by using the blood volume pulse signature",
# Physiol. Meas. 35:1913). Multiplied by the skin DC level per channel this
# gives absolute amplitudes of roughly B:G:R = 0.52 : 1 : 0.58 for the skin
# colour below (green strongest, as in Verkruysse et al. 2008).
PBV_RGB = (0.33, 0.77, 0.53)
SKIN_BGR = (105.0, 140.0, 190.0)

ProcessorFactory = Callable[[FilterParams], Any]
# Processing mask: (frame_bgr, frame_index) -> float32 HxW in 0..1, or None (no mask).
MaskFn = Callable[[np.ndarray, int], "np.ndarray | None"]
# Metric region: static bool HxW, or callable frame_index -> bool HxW.
SkinMask = "np.ndarray | Callable[[int], np.ndarray]"


# ---------------------------------------------------------------------------
# Synthetic clips
# ---------------------------------------------------------------------------
def pulse_waveform(t: np.ndarray, pulse_hz: float) -> np.ndarray:
    """PPG-like waveform (fundamental + 2nd harmonic, sharp systolic upstroke),
    normalised to peak-to-peak 2 (so ``amp * w`` has amplitude ``amp``)."""
    phi = 2 * math.pi * pulse_hz * np.asarray(t, np.float64)
    w = np.sin(phi) + 0.25 * np.sin(2 * phi - math.pi / 2)
    w -= w.mean()
    return w / ((w.max() - w.min()) / 2 + 1e-12)


def _background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Static textured background: blurry colour blobs plus a few hard edges."""
    low = rng.uniform(50, 190, (max(2, h // 40), max(2, w // 40), 3)).astype(np.float32)
    bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(6):  # shelves / door frames: hard edges that move-artifact if shaken
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(w // 12, w // 4)), y0 + int(rng.integers(h // 12, h // 4))
        col = tuple(float(c) for c in rng.uniform(30, 220, 3))
        cv2.rectangle(bg, (x0, y0), (x1, y1), col, -1)
    bg += rng.normal(0, 4, bg.shape).astype(np.float32)  # fine static texture
    return np.clip(cv2.GaussianBlur(bg, (0, 0), 0.8), 0, 255)


def _face_layers(h: int, w: int, rng: np.random.Generator, pulse_amp: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Face at rest position: (colour HxWx3, alpha HxW, pulse map HxWx3, skin HxW).

    pulse map = per-channel pixel amplitude of the pulse for w(t)=1.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.48
    ax, ay = w * 0.17, h * 0.32
    r = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
    edge_px = (1.0 - r) * min(ax, ay)  # approx. distance inside the outline
    alpha = np.clip(0.5 + edge_px / 1.2, 0, 1)

    # Smooth shading: dome + light from the left + mild skin texture.
    dome = np.sqrt(np.clip(1 - r ** 2, 0, 1))
    lam = 0.62 + 0.30 * dome + 0.08 * (cx - xx) / ax
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    lam = np.clip(lam + 0.02 * tex, 0.3, 1.1).astype(np.float32)
    colour = lam[..., None] * np.array(SKIN_BGR, np.float32)

    # Non-pulsing features (eyes, brows, mouth): darker, excluded from skin.
    feat = np.zeros((h, w), np.float32)
    for sx in (-1, 1):
        ex = cx + sx * 0.38 * ax
        cv2.ellipse(feat, (int(ex), int(cy - 0.12 * ay)), (int(0.18 * ax), int(0.07 * ay)), 0, 0, 360, 1.0, -1)
        cv2.ellipse(feat, (int(ex), int(cy - 0.27 * ay)), (int(0.22 * ax), int(0.03 * ay)), 0, 0, 360, 1.0, -1)
    cv2.ellipse(feat, (int(cx), int(cy + 0.5 * ay)), (int(0.3 * ax), int(0.07 * ay)), 0, 0, 360, 1.0, -1)
    feat = cv2.GaussianBlur(feat, (0, 0), 1.0)
    feat_col = np.array((55.0, 50.0, 70.0), np.float32)
    colour = colour * (1 - feat[..., None]) + feat_col * feat[..., None]

    skin = np.clip(alpha * (1 - feat) * (r < 0.97), 0, 1).astype(np.float32)
    pbv_bgr = np.array(PBV_RGB[::-1], np.float32)
    dc = np.array(SKIN_BGR, np.float32)
    rel = (pbv_bgr * dc) / (pbv_bgr[1] * dc[1])  # absolute amplitude relative to green
    pulse_map = (pulse_amp * rel)[None, None, :] * (lam * skin)[..., None]
    return colour.astype(np.float32), alpha.astype(np.float32), pulse_map.astype(np.float32), skin


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if dx == 0 and dy == 0:
        return img
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def synthetic_clip(n: int = 300, h: int = 480, w: int = 640, fs: float = 30.0,
                   pulse_hz: float = 1.2, pulse_amp: float = 0.5, noise_std: float = 2.0,
                   motion_px: float = 0.0, motion_hz: float = 0.3, drift: float = 0.0,
                   jpeg_quality: int | None = None, seed: int = 0,
                   brightness: float = 1.0) -> tuple[list[np.ndarray], dict]:
    """Render a synthetic face clip with a known pulse.

    Scene: textured static background, elliptical skin-coloured face with
    smooth shading and dark eye/brow/mouth features. The pulse multiplies skin
    colour with the BVP signature (green strongest, see PBV_RGB); ``pulse_amp``
    is the green amplitude in pixel units at full shading. Then:

    * ``motion_px``: sinusoidal head translation at ``motion_hz``
      (dx = m sin(wt), dy = 0.3 m sin(wt + 1)), sub-pixel, soft face edge.
    * ``drift``: global illumination gain 1 + drift*(t/T - 0.5) + 0.3*drift*sin(2pi 0.15 t).
    * ``brightness``: scales the whole scene (pulse included, it is relative).
    * Gaussian sensor noise ``noise_std``, rounding to uint8, then optional JPEG.

    Returns (frames, truth) with truth keys: ``skin_mask`` (bool HxW, or a
    callable i -> bool HxW when there is motion), ``pulse_hz``, ``pulse_amp``,
    ``fs``, ``pulse_wave`` (unit waveform per frame, float64 [n]),
    ``offsets`` (n x 2 head offsets in px).
    """
    rng = np.random.default_rng(seed)
    bg = _background(h, w, rng)
    colour, alpha, pulse_map, skin = _face_layers(h, w, rng, pulse_amp)
    face4 = np.dstack([colour, alpha])
    pulse4 = np.dstack([pulse_map, skin])

    t = np.arange(n) / fs
    wave = pulse_waveform(t, pulse_hz)
    om = 2 * math.pi * motion_hz * t
    offsets = np.stack([motion_px * np.sin(om), 0.3 * motion_px * np.sin(om + 1.0)], axis=1)
    dur = n / fs
    gain = brightness * (1 + drift * (t / dur - 0.5) + 0.3 * drift * np.sin(2 * math.pi * 0.15 * t))

    def composite(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
        """(static scene, pulse layer) with the face shifted by (dx, dy)."""
        f4 = _shift(face4, dx, dy)
        p4 = _shift(pulse4, dx, dy)
        a = f4[..., 3:4]
        return bg * (1 - a) + f4[..., :3] * a, p4[..., :3] * a

    static = None if motion_px else composite(0.0, 0.0)
    frames: list[np.ndarray] = []
    for i in range(n):
        base, pulse = static if static is not None else composite(*offsets[i])
        img = (base + pulse * np.float32(wave[i])) * np.float32(gain[i])
        if noise_std > 0:
            img = img + rng.standard_normal(img.shape, dtype=np.float32) * noise_std
        frame = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        if jpeg_quality is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        frames.append(frame)

    if motion_px:
        def skin_mask(i: int) -> np.ndarray:
            dx, dy = offsets[min(max(i, 0), n - 1)]
            return _shift(skin, dx, dy) > 0.5
        mask: Any = skin_mask
    else:
        mask = skin > 0.5
    truth = {"skin_mask": mask, "pulse_hz": pulse_hz, "pulse_amp": pulse_amp, "fs": fs,
             "pulse_wave": wave, "offsets": offsets}
    return frames, truth


SCENARIOS: dict[str, dict[str, Any]] = {
    "clean": dict(noise_std=1.0),
    "noisy": dict(noise_std=6.0),
    "motion": dict(noise_std=2.0, motion_px=6.0, motion_hz=0.3),
    "drift": dict(noise_std=2.0, drift=0.15),
    "jpeg": dict(noise_std=2.0, jpeg_quality=70),
    "dim": dict(noise_std=5.0, brightness=0.3),
}


def scenario_clip(name: str, **overrides: Any) -> tuple[list[np.ndarray], dict]:
    """synthetic_clip() with a named scenario's settings (see SCENARIOS)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return synthetic_clip(**{**SCENARIOS[name], **overrides})


# ---------------------------------------------------------------------------
# Clip I/O
# ---------------------------------------------------------------------------
def save_clip(frames: Sequence[np.ndarray], path: str | Path, fs: float,
              timestamps: Sequence[float] | None = None) -> Path:
    """Save frames losslessly as compressed npz (uint8 N x H x W x 3 + fs)."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(f, np.uint8) for f in frames])
    extra = {} if timestamps is None else {"timestamps": np.asarray(timestamps, np.float64)}
    np.savez_compressed(path, frames=arr, fs=np.float64(fs), **extra)
    return path


def load_clip(path: str | Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    """Load (frames, fs) from an .npz written by save_clip or any video file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            arr = z["frames"][:max_frames]
            fs = float(z["fs"]) if "fs" in z else FS_DEFAULT
        return [np.ascontiguousarray(f) for f in arr], fs
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    fs = cap.get(cv2.CAP_PROP_FPS)
    fs = float(fs) if fs and fs > 1 else FS_DEFAULT
    frames: list[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def default_face_region(h: int, w: int) -> np.ndarray:
    """Fallback metric region for clips with no mask: central face-sized ellipse."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, int(h * 0.45)), (int(w * 0.13), int(h * 0.22)), 0, 0, 360, 1, -1)
    return m.astype(bool)


@dataclass
class _Regions:
    skin: np.ndarray    # bool, pulse-averaging region
    inner: np.ndarray   # eroded skin (speckle, artifact denominator)
    band: np.ndarray    # dilated minus eroded (boundary halos)
    bg: np.ndarray      # outside dilated skin
    weights: np.ndarray  # float32 [H*W, 4] indicator columns (skin, inner, band, bg)
    counts: tuple[float, float, float, float]  # pixels per region (>= 1)


def _regions(skin: np.ndarray, radius: int) -> _Regions:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = skin.astype(np.uint8)
    dil = cv2.dilate(m, k).astype(bool)
    ero = cv2.erode(m, k).astype(bool)
    if not ero.any():
        ero = skin.astype(bool)
    skin_b = skin.astype(bool)
    masks = (skin_b, ero, dil & ~ero, ~dil)
    weights = np.stack([x.reshape(-1) for x in masks], axis=1).astype(np.float32)
    counts = tuple(max(1.0, float(x.sum())) for x in masks)
    return _Regions(*masks, weights=weights, counts=counts)  # type: ignore[arg-type]


def _psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nfft = max(4096, 1 << int(math.ceil(math.log2(8 * len(x)))))
    return periodogram(x, fs, window="hann", nfft=nfft, detrend="linear", scaling="density")


def _db(x: float) -> float:
    return 10 * math.log10(max(x, 1e-12))


def evaluate(processor_factory: ProcessorFactory, frames: Sequence[np.ndarray], fs: float,
             levels: int | None = None, skin_mask: Any = None, mask_fn: MaskFn | None = None,
             params: FilterParams | None = None, settle_s: float = 2.0,
             truth: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Run a processor over a clip and return objective metrics.

    Args:
        processor_factory: callable(FilterParams) -> processor.
        frames: uint8 BGR frames, all the same size.
        fs: frame rate (Hz).
        levels: pyramid levels (default: auto to ~160 wide, as the dashboard).
        skin_mask: *metric* region, bool HxW or callable(i) -> bool HxW. Defaults
            to truth["skin_mask"], else the processing mask (>0.5), else a
            central ellipse (default_face_region).
        mask_fn: optional *processing* mask, callable(frame, i) -> float32 HxW
            0..1 (or None = unmasked); multiplied into the upscaled signal.
        params: filter params (default FilterParams(fs=fs)).
        settle_s: seconds skipped before measuring (filter start-up transient).
        truth: synthetic truth dict; gives pulse_hz and pulse_wave (fidelity).

    Metrics (all over frames after settling):
        pulse_snr_db: skin-mean green of the *output*; power within
            +/-max(0.1 Hz, 2/T) of f0 and 2*f0 vs the rest of 0.5-4 Hz.
            f0 = truth pulse_hz, else the spectral peak of that signal.
        f0_hz / f0_est_hz: frequency used / spectral peak of the output (Hz).
        pulse_gain: amplitude ratio output/input of the skin-mean green at f0
            (1.0 = no magnification).
        pulse_amp_out: amplitude (px) of the skin-mean amplified green at f0.
        background_noise: RMS of the amplified signal outside the dilated skin
            (all channels, px; lower is better).
        bg_temporal_std: per-pixel temporal std of (output - input) outside the
            dilated skin, averaged over pixels (px).
        skin_speckle: spatial std of the amplified signal inside the eroded skin
            per frame and channel, averaged (px; colour noise / speckle).
        artifact_ratio: mean signal energy in the boundary band (dilated minus
            eroded skin) / mean energy inside the eroded skin (motion halos).
        pulse_cnr_db: 20 log10(pulse_amp_out / sqrt(skin_speckle^2 +
            background_noise^2)): how far the visible pulse colour change stands
            above the per-pixel amplified noise.
        pulse_fidelity: max Pearson correlation (lags 0..0.5 s) between the
            skin-mean amplified green and truth pulse_wave (NaN without truth).
        fps_proc: frames/s of proc.process() alone; fps_pipeline: full
            downscale+process+upscale+mask+add.
        quality_score: pulse_cnr_db - 10 log10(1 + artifact_ratio)
            - max(0, 10 - pulse_snr_db). I.e. visible pulse contrast over pixel
            noise, penalised for halos and for an impure (non-periodic) skin
            colour signal below 10 dB SNR. Higher is better; a processor that
            adds nothing scores very low because pulse_amp_out ~ 0.
    """
    if len(frames) == 0:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    params = params or FilterParams(fs=fs)
    proc = processor_factory(params)
    proc.reset()
    levels = pyramid_levels_for(w) if levels is None else levels
    n = len(frames)
    n_settle = int(round(settle_s * fs))
    if n - n_settle < 2 * fs:
        raise ValueError(f"need >= 2 s of frames after settling ({n} frames, settle {n_settle})")
    radius = max(2, int(round(min(h, w) / 40)))

    if skin_mask is None and truth is not None:
        skin_mask = truth.get("skin_mask")
    fallback = default_face_region(h, w)
    static_regions = _regions(skin_mask, radius) if isinstance(skin_mask, np.ndarray) else None

    m = n - n_settle
    g_in, g_out, s_sig = (np.zeros(m) for _ in range(3))
    speckle = np.zeros(m)
    bg_e = bg_c = band_e = band_c = in_e = in_c = 0.0
    t_s1 = np.zeros((h, w, 3), np.float32)
    t_s2 = np.zeros((h, w, 3), np.float32)
    t_c = np.zeros((h, w), np.float32)
    t_proc = t_pipe = 0.0

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        small = downscale(frame, levels)
        t1 = time.perf_counter()
        sig = proc.process(small)
        t2 = time.perf_counter()
        sig_full = upscale(np.asarray(sig, np.float32), levels, (h, w))
        pmask = mask_fn(frame, i) if mask_fn is not None else None
        if pmask is not None:
            sig_full = sig_full * np.asarray(pmask, np.float32)[..., None]
        out = add_signal(frame, sig_full)
        t3 = time.perf_counter()
        t_proc += t2 - t1
        t_pipe += t3 - t0
        if i < n_settle:
            continue

        if static_regions is not None:
            reg = static_regions
        elif callable(skin_mask):
            reg = _regions(np.asarray(skin_mask(i), bool), radius)
        elif pmask is not None:
            reg = _regions(np.asarray(pmask) > 0.5, radius)
        else:
            reg = _regions(fallback, radius)
            static_regions = reg if mask_fn is None else None
        if not reg.skin.any():
            reg = _regions(fallback, radius)

        # Masked sums as mat-vec products (much faster than boolean indexing).
        j = i - n_settle
        wts = reg.weights
        s_flat = sig_full.reshape(-1, 3)
        sq_flat = s_flat * s_flat
        s_sum = s_flat.T @ wts          # [3 channels, 4 regions]
        sq_sum = sq_flat.T @ wts
        n_skin, n_inner, n_band, n_bg = reg.counts
        g_in[j] = float(frame[..., 1].reshape(-1).astype(np.float32) @ wts[:, 0]) / n_skin
        g_out[j] = float(out[..., 1].reshape(-1).astype(np.float32) @ wts[:, 0]) / n_skin
        s_sig[j] = s_sum[1, 0] / n_skin
        mean_in = s_sum[:, 1] / n_inner
        speckle[j] = float(np.sqrt(np.maximum(sq_sum[:, 1] / n_inner - mean_in ** 2, 0)).mean())
        in_e += float(sq_sum[:, 1].sum()); in_c += 3 * n_inner
        band_e += float(sq_sum[:, 2].sum()); band_c += 3 * n_band
        bg_e += float(sq_sum[:, 3].sum()); bg_c += 3 * n_bg
        d = cv2.subtract(out, frame, dtype=cv2.CV_32F)
        if static_regions is not None:
            t_s1 += d
            t_s2 += d * d
            t_c += 1.0
        else:
            wbg = reg.bg.astype(np.float32)
            t_s1 += d * wbg[..., None]
            t_s2 += d * d * wbg[..., None]
            t_c += wbg

    # --- spectral pulse metrics ---
    f, p_out = _psd(g_out, fs)
    _, p_in = _psd(g_in, fs)
    _, p_sig = _psd(s_sig, fs)
    df = f[1] - f[0]
    in_band = (f >= PULSE_BAND[0]) & (f <= PULSE_BAND[1])
    f0_est = float(f[in_band][np.argmax(p_out[in_band])])
    f0 = float(truth["pulse_hz"]) if truth is not None and "pulse_hz" in truth else f0_est
    hw = max(PEAK_HALF_WIDTH, 2.0 * fs / m)
    fund = np.abs(f - f0) <= hw
    sig_bins = (fund | (np.abs(f - 2 * f0) <= hw)) & in_band
    noise_bins = in_band & ~sig_bins
    snr_db = _db(p_out[sig_bins].sum()) - _db(p_out[noise_bins].sum())
    gain = math.sqrt(p_out[fund].sum() / max(p_in[fund].sum(), 1e-20))
    amp_out = math.sqrt(2 * p_sig[fund].sum() * df)

    # --- spatial noise / artifact metrics ---
    bg_rms = math.sqrt(bg_e / bg_c) if bg_c else float("nan")
    if static_regions is not None:
        t_c = t_c * static_regions.bg
    valid = t_c >= max(1.0, 0.5 * m)
    if valid.any():
        c = t_c[valid][:, None]
        var = np.maximum(t_s2[valid] / c - (t_s1[valid] / c) ** 2, 0)
        bg_tstd = float(np.sqrt(var).mean())
    else:
        bg_tstd = float("nan")
    speck = float(speckle.mean())
    artifact = (band_e / band_c) / max(in_e / in_c, 1e-12) if band_c and in_c else float("nan")
    noise_px = math.sqrt(speck ** 2 + (bg_rms if math.isfinite(bg_rms) else 0.0) ** 2)
    cnr_db = 20 * math.log10(max(amp_out, 1e-6) / max(noise_px, 1e-3))

    # --- fidelity vs ground truth ---
    fidelity = float("nan")
    wave = None if truth is None else truth.get("pulse_wave")
    if wave is not None:
        wave = np.asarray(wave, np.float64)
        best = -1.0
        for lag in range(0, min(int(MAX_FIDELITY_LAG_S * fs), n_settle) + 1):
            ref = wave[n_settle - lag:n - lag]
            if s_sig.std() < 1e-9 or ref.std() < 1e-9:
                continue
            best = max(best, float(np.corrcoef(s_sig, ref)[0, 1]))
        fidelity = best

    quality = (cnr_db - _db(1 + (artifact if math.isfinite(artifact) else 0.0))
               - max(0.0, 10.0 - snr_db))
    return {
        "pulse_snr_db": snr_db,
        "f0_hz": f0,
        "f0_est_hz": f0_est,
        "pulse_gain": gain,
        "pulse_amp_out": amp_out,
        "background_noise": bg_rms,
        "bg_temporal_std": bg_tstd,
        "skin_speckle": speck,
        "artifact_ratio": artifact,
        "pulse_cnr_db": cnr_db,
        "pulse_fidelity": fidelity,
        "fps_proc": n / max(t_proc, 1e-9),
        "fps_pipeline": n / max(t_pipe, 1e-9),
        "quality_score": quality,
        "levels": float(levels),
        "n_frames": float(n),
    }


# ---------------------------------------------------------------------------
# Variants and masks
# ---------------------------------------------------------------------------
def variant_registry(verbose: bool = False) -> dict[str, ProcessorFactory]:
    """'contract' = FloatEVM plus evm_enhanced.PRESETS (if importable)."""
    reg: dict[str, ProcessorFactory] = {"contract": lambda p: FloatEVM(p)}
    try:
        from evm_enhanced import PRESETS  # type: ignore[import-not-found]
    except ImportError:
        return reg
    except Exception as exc:  # half-written module from a parallel edit
        if verbose:
            print(f"[eval] evm_enhanced failed to import: {exc!r}", file=sys.stderr)
        return reg
    for name, factory in dict(PRESETS).items():
        reg.setdefault(str(name), factory)
    return reg


def truth_mask_fn(truth: Mapping[str, Any], blur_px: float = 3.0) -> MaskFn:
    """Processing mask from the synthetic truth skin mask (softened edge)."""
    sm = truth["skin_mask"]
    cache: dict[str, np.ndarray] = {}

    def soft(b: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(b.astype(np.float32), (0, 0), blur_px)

    def fn(frame: np.ndarray, i: int) -> np.ndarray:
        if callable(sm):
            return soft(sm(i))
        if "m" not in cache:
            cache["m"] = soft(sm)
        return cache["m"]
    return fn


def face_mask_fn() -> MaskFn | None:
    """Processing mask from face_tracker.FaceTracker (None if unavailable).

    A fresh tracker is created per call, so use one per evaluated variant.
    """
    try:
        from face_tracker import FaceTracker  # type: ignore[import-not-found]
        tracker = FaceTracker()
    except Exception as exc:
        print(f"[eval] face tracker unavailable ({exc!r}); running unmasked", file=sys.stderr)
        return None

    def fn(frame: np.ndarray, i: int) -> np.ndarray | None:
        res = tracker.update(frame)
        m = getattr(res, "skin_mask", None) if res is not None else None
        return None if m is None else np.asarray(m, np.float32)
    return fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
COLUMNS = (
    ("variant", "{:<16}", None),
    ("snr_dB", "{:>7.1f}", "pulse_snr_db"),
    ("gain", "{:>6.1f}", "pulse_gain"),
    ("amp", "{:>6.2f}", "pulse_amp_out"),
    ("cnr_dB", "{:>7.1f}", "pulse_cnr_db"),
    ("bg_rms", "{:>7.2f}", "background_noise"),
    ("bg_tstd", "{:>7.2f}", "bg_temporal_std"),
    ("speckle", "{:>7.2f}", "skin_speckle"),
    ("artif", "{:>6.2f}", "artifact_ratio"),
    ("fidel", "{:>6.3f}", "pulse_fidelity"),
    ("f0_est", "{:>6.2f}", "f0_est_hz"),
    ("fps", "{:>7.0f}", "fps_proc"),
    ("quality", "{:>8.1f}", "quality_score"),
)


def format_table(rows: Sequence[tuple[str, Mapping[str, float]]]) -> str:
    head = " ".join([f"{COLUMNS[0][0]:<16}"]
                    + [f"{label:>{len(fmt.format(0.0))}}" for label, fmt, _ in COLUMNS[1:]])
    lines = [head, "-" * len(head)]
    for name, r in rows:
        cells = [COLUMNS[0][1].format(name[:16])]
        cells += [fmt.format(r.get(key, float("nan"))) for _, fmt, key in COLUMNS[1:]]
        lines.append(" ".join(cells))
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    registry = variant_registry(verbose=True)
    names = list(registry) if not args.variants else [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in registry]
    if unknown:
        print(f"unknown variants {unknown}; available: {sorted(registry)}", file=sys.stderr)
        return 2

    if args.clip:
        frames, fs = load_clip(args.clip, args.max_frames)
        clips = [(Path(args.clip).name, frames, fs, None)]
    else:
        scen = list(SCENARIOS) if args.synthetic == "all" else [args.synthetic]
        clips = []
        for s in scen:
            w, h = (int(v) for v in args.size.lower().split("x"))
            fr, tr = scenario_clip(s, n=int(round(args.seconds * 30)), seed=args.seed, h=h, w=w)
            clips.append((s, fr, tr["fs"], tr))

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    for label, frames, fs, truth in clips:
        params = FilterParams(fc_lo=args.fc_lo, fc_hi=args.fc_hi, alpha=args.alpha, fs=fs)
        rows = []
        for v in names:
            mask_fn: MaskFn | None = None
            if args.mask == "truth":
                if truth is None:
                    print("[eval] --mask truth needs a synthetic clip; using face", file=sys.stderr)
                    mask_fn = face_mask_fn()
                else:
                    mask_fn = truth_mask_fn(truth)
            elif args.mask == "face":
                mask_fn = face_mask_fn()
            r = evaluate(registry[v], frames, fs, levels=args.levels, mask_fn=mask_fn,
                         params=params, settle_s=args.settle, truth=truth)
            rows.append((v, r))
        h, w = frames[0].shape[:2]
        print(f"\n== {label}: {len(frames)} frames {w}x{h} @ {fs:g} fps, mask={args.mask}, "
              f"alpha={args.alpha:g} band={args.fc_lo:g}-{args.fc_hi:g} Hz ==")
        print(format_table(rows))
        all_results[label] = {v: r for v, r in rows}
    if args.json:
        Path(args.json).write_text(json.dumps(all_results, indent=2))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from camera import CameraConfig, FrameSource

    cfg = CameraConfig(source=args.source, width=args.width, height=args.height, fps=args.fps,
                       mjpg=not args.no_mjpg, lock_exposure=not args.no_lock,
                       exposure=args.exposure, lock_wb=not args.no_lock,
                       wb_temperature=args.wb, loop_video=False)
    if args.backend:
        cfg.backend = args.backend
    src = FrameSource(cfg)
    print(src.report)
    fs = src.nominal_fps if src.is_file else cfg.fps
    n_target = int(round(args.seconds * fs))
    frames: list[np.ndarray] = []
    stamps: list[float] = []
    try:
        if not src.is_file and args.warmup > 0:
            t_end = time.perf_counter() + args.warmup
            while time.perf_counter() < t_end:
                src.read()
        print(f"[record] recording {n_target} frames ({args.seconds:g} s @ {fs:g} fps)...")
        while len(frames) < n_target:
            ok, frame, t = src.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            stamps.append(t)
            if args.preview:
                view = frame.copy()
                cv2.putText(view, f"REC {len(frames) / fs:4.1f}/{args.seconds:g}s", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("evm_eval record", view)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        src.release()
        if args.preview:
            cv2.destroyAllWindows()
    if not frames:
        print("[record] no frames captured", file=sys.stderr)
        return 1
    ts = np.asarray(stamps) - stamps[0]
    if not src.is_file and len(ts) > 1:
        measured = (len(ts) - 1) / max(ts[-1], 1e-9)
        print(f"[record] measured {measured:.2f} fps (nominal {fs:g})")
        if abs(measured - fs) > 0.05 * fs:
            print("[record] WARNING: frame rate differs >5% from nominal; pulse frequency "
                  "estimates will be scaled. Check exposure lock / lighting.", file=sys.stderr)
    out_dir = Path(args.out_dir) if args.out_dir else RECORDINGS_DIR
    path = save_clip(frames, out_dir / f"{args.name}.npz", fs, ts)
    h, w = frames[0].shape[:2]
    print(f"[record] saved {len(frames)} frames {w}x{h} -> {path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EVM objective evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="evaluate variants on a clip and print a table")
    src = c.add_mutually_exclusive_group()
    src.add_argument("--clip", help=".npz from `record` or any video file")
    src.add_argument("--synthetic", default="clean", choices=[*SCENARIOS, "all"],
                     help="synthetic scenario (default clean)")
    c.add_argument("--variants", default="", help="comma list (default: all registered)")
    c.add_argument("--levels", type=int, default=None, help="pyramid levels (default auto ~160 wide)")
    c.add_argument("--mask", default="none", choices=["truth", "face", "none"],
                   help="processing mask applied to the signal (default none)")
    c.add_argument("--alpha", type=float, default=FilterParams.alpha)
    c.add_argument("--fc-lo", type=float, default=FilterParams.fc_lo)
    c.add_argument("--fc-hi", type=float, default=FilterParams.fc_hi)
    c.add_argument("--settle", type=float, default=2.0, help="seconds skipped at start")
    c.add_argument("--seconds", type=float, default=10.0, help="synthetic clip length")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--size", default="640x480", help="synthetic frame size WxH")
    c.add_argument("--max-frames", type=int, default=None, help="limit frames loaded from --clip")
    c.add_argument("--json", default=None, help="also write results to this JSON file")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("record", help="record a lossless clip from a camera (or video file)")
    r.add_argument("--name", required=True, help="output name -> recordings/<name>.npz")
    r.add_argument("--seconds", type=float, default=10.0)
    r.add_argument("--source", default="0", help="camera index or video file")
    r.add_argument("--backend", default=None, help="dshow | msmf | auto")
    r.add_argument("--width", type=int, default=640)
    r.add_argument("--height", type=int, default=480)
    r.add_argument("--fps", type=float, default=FS_DEFAULT)
    r.add_argument("--no-lock", action="store_true", help="leave auto exposure / WB on")
    r.add_argument("--exposure", type=float, default=None)
    r.add_argument("--wb", type=float, default=None)
    r.add_argument("--no-mjpg", action="store_true")
    r.add_argument("--warmup", type=float, default=1.0, help="camera seconds discarded first")
    r.add_argument("--preview", action="store_true", help="show a live window while recording")
    r.add_argument("--out-dir", default=None, help="default sw/reference/recordings")
    r.set_defaults(func=cmd_record)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
