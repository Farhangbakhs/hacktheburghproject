"""Live, pulse-locked, face-only EVM (the real-time route to MIT-quality pulse video).

Why the plain filter "pulses randomly"
--------------------------------------
* Its passband is effectively 0.5-3.4 Hz, so noise/lighting wobble across that
  whole range is amplified as much as the heartbeat. MIT's face video used an
  ideal 0.83-1.0 Hz band picked by hand around that subject's pulse.
* It amplifies raw BGR, which at webcam light levels is dominated by
  *luminance* noise. Measured on our recordings, a chrominance signal
  (POS / G-(R+B)/2 on normalised colour) has 3-8 dB more pulse SNR than green.
* Plain pyramid pooling mixes background into the face at coarse levels.

What this processor does (FPGA part unchanged = section 3 contract)
------------------------------------------------------------------
  FPGA / contract : FloatEVM difference-of-lowpass IIR, alpha_hw register
  laptop          :
    1. Lock: skin-weighted mean RGB of the raw small frame -> POS (Wang 2017,
       1.6 s windows, overlap-add) over `window_s` -> spectrum peak 0.7-3 Hz
       with parabolic refinement + SNR gate. f0 smoothed and jump-limited.
    2. Local signal: contract band / slow per-pixel mean (normalised colour),
       projected to chrominance s = Gn - (Rn+Bn)/2, then *mask-normalised*
       pooling P(s*m)/P(m) so background never leaks into the face.
    3. Global signal: the same chrominance of the skin-mean colour.
    4. Narrow per-pixel / scalar Butterworth bandpass f0 +/- bw on both.
    5. Render: s_mix = mix*global + (1-mix)*local, drawn back along the
       blood-volume-pulse colour direction (Pbv, de Haan & van Leest 2014),
       scaled by each pixel's own colour, only on the (feathered) skin mask.
       Gain ramps in only while locked: no lock -> no fake pulsing.

`mix` = 0 keeps the rendering purely per-pixel from the FPGA output; mix = 1
renders the face-wide measured pulse uniformly over the skin (cleanest look).
Both are this person's real signal; say which in the demo.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.signal as sps

from evm_common import FilterParams, upscale
from evm_enhanced import _SOS
from evm_float import FloatEVM

# Blood-volume-pulse signature in normalised RGB (de Haan & van Leest 2014),
# reordered to BGR. Its chrominance projection G - (R+B)/2:
PBV_BGR = np.array([0.53, 0.77, 0.33], np.float32)
_CHROMA_W = np.array([-0.5, 1.0, -0.5], np.float32)  # BGR weights for Gn - (Rn+Bn)/2
_PBV_S = float(PBV_BGR @ _CHROMA_W)  # 0.34: s of a unit Pbv excursion


@dataclass
class LockConfig:
    alpha_hw: float = 25.0       # FPGA alpha register (keeps motion edges unclipped)
    pool_levels: int = 2         # mask-normalised pooling (160x120 -> 40x30)
    window_s: float = 8.0        # lock spectrum window
    min_window_s: float = 5.0
    update_every: int = 10       # frames between lock updates
    band: tuple[float, float] = (0.7, 3.0)
    bw: float = 0.12             # narrow passband half-width (Hz)
    order: int = 2
    snr_lock_db: float = 1.0
    snr_hold_db: float = -2.0
    max_jump_hz: float = 0.25
    ramp_frames: int = 30
    mix: float = 0.6             # 0 = per-pixel (FPGA) only, 1 = face-wide pulse only
    mean_tau_s: float = 2.0      # slow per-pixel mean for colour normalisation
    soft_clip: float = 40.0
    pos_win_s: float = 1.6


def contract_gain(f: float, p: FilterParams) -> float:
    z = np.exp(1j * 2 * math.pi * f / p.fs)
    h = lambda r: r / (1 - (1 - r) / z)  # noqa: E731
    return abs(h(p.r_hi) - h(p.r_lo))


def pos_signal(rgb_bgr: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """POS (Wang et al. 2017) on a (T,3) BGR mean-colour trace, overlap-add."""
    c = rgb_bgr[:, ::-1].astype(np.float64)  # -> RGB
    n, w = len(c), max(2, int(round(win_s * fs)))
    h = np.zeros(n)
    for t in range(0, n - w + 1):
        cn = c[t:t + w] / np.maximum(c[t:t + w].mean(0), 1e-6)
        s1 = cn[:, 1] - cn[:, 2]
        s2 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]
        p = s1 + (s1.std() / (s2.std() + 1e-12)) * s2
        h[t:t + w] += p - p.mean()
    return h


def spectrum_peak(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[float, float] | None:
    """(peak Hz with parabolic refinement, SNR dB incl. 1st harmonic) or None."""
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 2), t)
    nfft = 4096
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    inband = (freqs >= band[0]) & (freqs <= band[1])
    if spec[inband].max() <= 1e-20:
        return None
    k = int(np.argmax(np.where(inband, spec, 0)))
    k_f = float(k)
    if 0 < k < len(spec) - 1:
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-30)
        den = a - 2 * b + c
        if den != 0:
            k_f = k + 0.5 * (a - c) / den
    f = float(np.clip(k_f * fs / nfft, *band))
    near = inband & ((np.abs(freqs - f) < 0.1) | (np.abs(freqs - 2 * f) < 0.1))
    snr = 10 * np.log10(spec[near].sum() / max(spec[inband & ~near].sum(), 1e-30))
    return f, float(snr)


def _pool(img: np.ndarray, levels: int) -> np.ndarray:
    for _ in range(levels):
        img = cv2.pyrDown(img)
    return img


def _centre_weights(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, int(h * 0.45)), (int(w * 0.16), int(h * 0.24)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 2)


class PulseLockedEVM:
    """FloatEVM-compatible processor (process/reset/set_params/band_state)."""

    def __init__(self, params: FilterParams | None = None, config: LockConfig | None = None):
        self.params = params or FilterParams()
        self.cfg = config or LockConfig()
        self._hw = FloatEVM(self._hw_params(self.params))
        self._mask: np.ndarray | None = None
        self._rgb: deque[np.ndarray] = deque(maxlen=int(self.cfg.window_s * self.params.fs))
        self.f0: float | None = None
        self.snr_db = -99.0
        self.locked = False
        self.pulse_value = 0.0       # current narrowband face-wide pulse (normalised units)
        self._gain = 0.0
        self._n = 0
        self._reset_state()

    # -- interface ---------------------------------------------------------------
    @property
    def name(self) -> str:
        s = f"locked {self.f0 * 60:.0f}bpm" if self.locked and self.f0 else "locking..."
        return f"{s} snr{self.snr_db:+.0f}dB"

    @property
    def alpha_hw(self) -> float:
        return self.cfg.alpha_hw

    @property
    def gate(self) -> float:
        return self._gain

    def _hw_params(self, p: FilterParams) -> FilterParams:
        return FilterParams(p.fc_lo, p.fc_hi, self.cfg.alpha_hw, p.fs)

    def set_params(self, params: FilterParams) -> None:
        params.validate()
        self.params = params
        self._hw.set_params(self._hw_params(params))

    def set_mask(self, mask_small: np.ndarray | None) -> None:
        """Skin weights (float 0..1) at the small (processing) size, or None."""
        self._mask = mask_small

    def _reset_state(self) -> None:
        self._mean_px: np.ndarray | None = None
        self._g_mean: np.ndarray | None = None
        self._narrow_px: _SOS | None = None
        self._narrow_g: _SOS | None = None
        self.frames_since_init = 0

    def reset(self) -> None:
        self._hw.reset()
        self._rgb.clear()
        self._reset_state()
        self.locked, self.f0, self._gain, self.snr_db, self.pulse_value = False, None, 0.0, -99.0, 0.0

    def band_state(self) -> np.ndarray | None:
        return self._hw.band_state()

    # -- core ---------------------------------------------------------------------
    def process(self, small_bgr: np.ndarray) -> np.ndarray:
        cfg, fs = self.cfg, self.params.fs
        x = small_bgr.astype(np.float32)
        hw = self._hw.process(small_bgr) / np.float32(cfg.alpha_hw)   # contract band, pixel units
        self.frames_since_init = self._hw.frames_since_init
        if self.frames_since_init == 1:
            self._reset_state()
            self.frames_since_init = 1
            self._rgb.clear()

        h, w = x.shape[:2]
        m = self._mask if (self._mask is not None and self._mask.shape == (h, w)
                           and self._mask.sum() > 4) else _centre_weights(h, w)
        msum = float(m.sum())

        # slow per-pixel mean colour (normalisation) and skin-mean colour
        k = np.float32(1.0 / max(1.0, cfg.mean_tau_s * fs))
        if self._mean_px is None:
            self._mean_px = x.copy()
        else:
            self._mean_px += k * (x - self._mean_px)
        skin_rgb = (x * m[..., None]).reshape(-1, 3).sum(0) / msum
        self._rgb.append(skin_rgb)
        self._g_mean = skin_rgb.copy() if self._g_mean is None else self._g_mean + k * (skin_rgb - self._g_mean)

        self._n += 1
        if self._n % cfg.update_every == 0:
            self._update_lock()

        step = 1.0 / max(1, cfg.ramp_frames)
        self._gain += float(np.clip((1.0 if self.locked else 0.0) - self._gain, -step, step))
        if self.f0 is None:
            return np.zeros_like(x)

        # local: normalised contract band -> chrominance -> mask-normalised pooling
        s_px = (hw / np.maximum(self._mean_px, 4.0)) @ _CHROMA_W
        num, den = _pool(s_px * m, cfg.pool_levels), _pool(m, cfg.pool_levels)
        s_pool = num / np.maximum(den, 1e-3)
        if self._narrow_px is None or self._narrow_px.z is None or self._narrow_px.z[0][0].shape != s_pool.shape:
            self._narrow_px = _SOS(self._sos(self.f0))
            self._narrow_px.init(np.zeros_like(s_pool))
        local = self._narrow_px.step(s_pool) / np.float32(max(contract_gain(self.f0, self.params), 1e-3))

        # global: chrominance of the skin-mean colour (no contract filter in this path)
        s_g = float(((skin_rgb - self._g_mean) / np.maximum(self._g_mean, 4.0)) @ _CHROMA_W)
        if self._narrow_g is None or self._narrow_g.z is None:
            self._narrow_g = _SOS(self._sos(self.f0))
            self._narrow_g.init(np.zeros((1,), np.float32))
        glob = float(self._narrow_g.step(np.array([s_g], np.float32))[0])
        self.pulse_value = glob

        s_mix = cfg.mix * glob + (1 - cfg.mix) * upscale(local, cfg.pool_levels, (h, w))
        # render along the blood-volume colour direction, scaled by pixel colour
        amp = np.float32(self.params.alpha * self._gain / _PBV_S) * np.asarray(s_mix, np.float32)
        delta = amp[..., None] * PBV_BGR * self._mean_px if np.ndim(amp) else amp * PBV_BGR * self._mean_px
        delta *= m[..., None]
        c = np.float32(cfg.soft_clip)
        return (c * np.tanh(delta / c)).astype(np.float32)

    def _sos(self, f0: float) -> np.ndarray:
        lo, hi = max(0.2, f0 - self.cfg.bw), min(self.params.fs / 2 - 0.5, f0 + self.cfg.bw)
        return sps.butter(self.cfg.order, [lo, hi], btype="band", fs=self.params.fs, output="sos")

    def _update_lock(self) -> None:
        cfg, fs = self.cfg, self.params.fs
        if len(self._rgb) < cfg.min_window_s * fs:
            return
        pk = spectrum_peak(pos_signal(np.asarray(self._rgb), fs, cfg.pos_win_s), fs, cfg.band)
        if pk is None:
            self.snr_db = -99.0
            return
        f_peak, self.snr_db = pk
        if not self.locked:
            if self.snr_db >= cfg.snr_lock_db:
                self.locked, self.f0 = True, f_peak
            elif self.f0 is None:
                self.f0 = f_peak  # pre-settle the narrow filters, output stays off
            self._retune()
            return
        if self.snr_db < cfg.snr_hold_db:
            self.locked = False
            return
        if self.snr_db >= cfg.snr_lock_db:
            new = self.f0 + float(np.clip(f_peak - self.f0, -cfg.max_jump_hz, cfg.max_jump_hz))
            self.f0 = 0.7 * self.f0 + 0.3 * new
            self._retune()

    def _retune(self) -> None:
        if self.f0 is None:
            return
        for f in (self._narrow_px, self._narrow_g):
            if f is not None:
                f.set_sos(self._sos(self.f0))


def make_locked(params: FilterParams | None = None, **kw) -> PulseLockedEVM:
    return PulseLockedEVM(params, LockConfig(**kw))


def main(argv: list[str] | None = None) -> int:
    import sys

    from dashboard import run_dashboard

    return run_dashboard(FloatEVM, ["--preset", "locked", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
