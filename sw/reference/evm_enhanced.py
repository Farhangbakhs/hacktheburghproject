"""Enhanced (experimental) variants of the EVM colour-magnification processor.

Drop-in replacement for evm_float.FloatEVM: same interface
(`name`, `process`, `reset`, `set_params`, `band_state`, `frames_since_init`),
so the dashboard and evaluation harness can swap them. Pick a variant with
`PRESETS[name](FilterParams())` or build `EnhancedEVM(params, EnhancedConfig(...))`.

Every option is labelled:

* CONTRACT-SAFE  - works around an unchanged FPGA kernel: laptop-side
                   processing of the int8 output, or per-frame writes of the
                   existing AXI-Lite registers (alpha, init).
* NON-CONTRACT   - would need a change to docs/PLAN.md section 3 (FPGA filter
                   or frame format). Flag to the team before relying on it.

Pipeline per frame (float32):

    x    = small_bgr (H x W x 3 uint8)
    x'   = pyrDown^extra_levels(x)                         NON-CONTRACT (pre-pool)
    band = temporal_bandpass(x')                           'iir1' = the contract filter
    d    = clip(alpha_hw * gate * band, +-stage_clip)      == FPGA int8 output
    --- everything below is laptop-side post-processing of d: CONTRACT-SAFE ---
    d    = pyrUp^extra_levels(d)
    d    = T @ d             3x3 colour render (bgr / yiq / ycrcb / pulse_dir / green)
    d    = pyrUp^k pyrDown^k(d), k = post_blur_levels      spatial pooling
    d    = post_temporal(d)  'lp' (1st-order LP) or 'butter' (Butterworth band)
    d    = post_mult * d     split gain + gain normalisation
    d    = soft_clip * tanh(d / soft_clip)                 soft limiter
    out  = clip(d, +-clip)

Why the post options are exact equivalents of doing the same thing before the
FPGA: the contract filter is linear and shift-invariant per pixel and channel,
pyrDown/pyrUp and the colour matrix are linear, so they commute with it
(including the init rule), up to the +-127 clip and int8 rounding. The test
suite checks  filter(pyrDown(x)) == pyrDown(filter(x))  numerically.
Pre-pooling the uint8 input on the laptop and re-quantising to uint8 would kill
the sensor-noise dither and lose the sub-LSB pulse; if anyone does that for real
add +-0.5 LSB triangular dither first. `extra_levels` here pools in float (no
re-quantisation), i.e. it models an FPGA running the filter at a coarser size.

Gain structure ("split gain"): the FPGA register alpha_hw stays moderate
(20-30) so motion edges do not saturate +-127; the rest of the gain is applied
on the laptop after pooling/filtering. With `hw_alpha` set, FilterParams.alpha
still means the total gain, so the dashboard slider keeps working:
    total     = params.alpha * post_gain * (1/peak_gain if gain_normalise)
    alpha_hw  = hw_alpha or params.alpha     (Q8.8 register: must be < 256)
    post_mult = total / alpha_hw
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np
from scipy import signal as sps

from evm_common import OUT_CLIP, FilterParams, upscale

FILTERS = ("iir1", "iir1x2", "butter2")
RENDERS = ("bgr", "yiq", "ycrcb", "pulse_dir")
POST_TEMPORAL = (None, "lp", "butter")

# Rows map a BGR column vector to (luma, chroma1, chroma2).
_BGR_TO = {
    "bgr": np.eye(3),
    # YIQ (NTSC) from RGB, columns reordered to B, G, R.
    "yiq": np.array(
        [
            [0.114, 0.587, 0.299],
            [-0.322, -0.274, 0.596],
            [0.312, -0.523, 0.211],
        ]
    ),
    # OpenCV YCrCb (offsets dropped: no temporal AC content).
    "ycrcb": np.array(
        [
            [0.114, 0.587, 0.299],
            [-0.114 * 0.713, -0.587 * 0.713, (1 - 0.299) * 0.713],
            [(1 - 0.114) * 0.564, -0.587 * 0.564, -0.299 * 0.564],
        ]
    ),
}
# pulse_dir: scalar s = G - (R+B)/2 (BGR weights); blood arrival lowers G most.
PULSE_DIR_W = np.array([-0.5, 1.0, -0.5])


# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------
@dataclass
class EnhancedConfig:
    """Independently switchable enhancements. Defaults == contract behaviour."""

    # temporal filter (in the FPGA position)
    filter: str = "iir1"             # 'iir1' contract | 'iir1x2','butter2' NON-CONTRACT
    extra_levels: int = 0            # NON-CONTRACT pre-pool (float, no requantise), 0..3
    stage_clip: float = OUT_CLIP     # clip after hw gain (= FPGA int8 clip)
    hw_alpha: float | None = None    # CONTRACT-SAFE split gain: fixed alpha register
    # colour render (3x3 on the output)                  CONTRACT-SAFE
    color_space: str = "bgr"         # 'bgr' | 'yiq' | 'ycrcb' | 'pulse_dir'
    render: str | None = None        # alias for color_space (overrides it if set)
    luma_gain: float = 1.0           # yiq/ycrcb: weight on Y (0.3-1)
    chroma_attenuation: float = 1.0  # yiq/ycrcb: weight on chroma (Wu et al.; 1-2 ok)
    pulse_colour: tuple[float, float, float] = (-0.3, -0.2, 1.0)  # BGR, "redder on beat"
    channels: str = "all"            # 'all' | 'green' (amplify only G)
    # spatial pooling of the output                       CONTRACT-SAFE
    post_blur_levels: int = 0        # 0..3; 2 levels ~ 4x less noise std
    # extra laptop temporal filter on the output          CONTRACT-SAFE
    post_temporal: str | None = None  # None | 'lp' | 'butter'
    post_lp_hz: float = 2.5
    post_band: tuple[float, float] = (0.75, 2.2)
    temporal_smooth_hz: float | None = None  # shortcut: post_temporal='lp' at this cutoff
    # gain                                                CONTRACT-SAFE
    post_gain: float = 1.0           # laptop multiplier (2-5x with hw_alpha 20-30)
    gain_normalise: bool = False     # total gain / peak |H| of the whole cascade
    soft_clip: float | None = None   # tanh limiter level T (output units)
    clip: float = OUT_CLIP           # final hard clip (30-40 is less garish)
    # motion gate                                         CONTRACT-SAFE (see reset)
    motion_gate: bool = False
    motion_threshold: float = 1.5    # mean |dLuma| per pixel/frame at 1/4 small size
    motion_hold_frames: int = 5      # still frames before re-init + ramp start
    motion_ramp_frames: int = 20     # linear 0 -> 1 gain ramp (~0.7 s)
    soft_reset: bool = True          # re-init filter state on motion
    reset_strength: float = 1.0      # 1.0 = INIT register (safe); <1 blend: NON-CONTRACT

    def __post_init__(self) -> None:
        if self.render is not None:
            self.color_space = self.render
        if self.temporal_smooth_hz and self.post_temporal is None:
            self.post_temporal, self.post_lp_hz = "lp", float(self.temporal_smooth_hz)

    def validate(self) -> None:
        if self.filter not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}")
        if not 0 <= self.extra_levels <= 3 or not 0 <= self.post_blur_levels <= 3:
            raise ValueError("extra_levels / post_blur_levels must be 0..3")
        if self.color_space not in RENDERS:
            raise ValueError(f"color_space/render must be one of {RENDERS}")
        if self.channels not in ("all", "green"):
            raise ValueError("channels must be 'all' or 'green'")
        if self.post_temporal not in POST_TEMPORAL:
            raise ValueError(f"post_temporal must be one of {POST_TEMPORAL}")
        if not 0 < self.reset_strength <= 1:
            raise ValueError("reset_strength must be in (0, 1]")
        if self.clip <= 0 or self.stage_clip <= 0 or (self.soft_clip is not None and self.soft_clip <= 0):
            raise ValueError("clip levels must be positive")
        if self.hw_alpha is not None and not 0 < self.hw_alpha < 256:
            raise ValueError("hw_alpha must be in (0, 256) (Q8.8 register)")
        if self.motion_ramp_frames < 1 or self.motion_hold_frames < 0:
            raise ValueError("bad motion ramp/hold")

    @property
    def contract_safe(self) -> bool:
        """True if every enabled option works around an unchanged FPGA kernel."""
        return (
            self.filter == "iir1"
            and self.extra_levels == 0
            and self.stage_clip <= OUT_CLIP
            and not (self.motion_gate and self.soft_reset and self.reset_strength < 1)
        )

    def short(self) -> str:
        p = [self.filter]
        if self.extra_levels:
            p.append(f"pre{self.extra_levels}")
        if self.hw_alpha:
            p.append(f"hw{self.hw_alpha:g}")
        if self.color_space in ("yiq", "ycrcb"):
            p.append(f"{self.color_space}(Y{self.luma_gain:g},C{self.chroma_attenuation:g})")
        elif self.color_space == "pulse_dir":
            p.append("pulse_dir")
        if self.channels != "all":
            p.append(self.channels)
        if self.post_blur_levels:
            p.append(f"blur{self.post_blur_levels}")
        if self.post_temporal == "lp":
            p.append(f"lp{self.post_lp_hz:g}Hz")
        elif self.post_temporal == "butter":
            p.append(f"bw{self.post_band[0]:g}-{self.post_band[1]:g}Hz")
        if self.post_gain != 1:
            p.append(f"x{self.post_gain:g}")
        if self.gain_normalise:
            p.append("norm")
        if self.soft_clip:
            p.append(f"tanh{self.soft_clip:g}")
        if self.motion_gate:
            p.append(f"gate{self.motion_threshold:g}" + ("+init" if self.soft_reset else ""))
        if self.clip != OUT_CLIP:
            p.append(f"clip{self.clip:g}")
        return ",".join(p)


# --------------------------------------------------------------------------------
# Filter design helpers (also used by tests for analytic responses)
# --------------------------------------------------------------------------------
def dolp_sos(params: FilterParams) -> np.ndarray:
    """Contract difference-of-lowpasses as one biquad (sos row).

    H = r_h/(1-a_h z^-1) - r_l/(1-a_l z^-1),  a = 1 - r.
    """
    rh, rl = params.r_hi, params.r_lo
    ah, al = 1 - rh, 1 - rl
    return np.array([[rh - rl, rl * ah - rh * al, 0.0, 1.0, -(ah + al), ah * al]])


def lp1_sos(fc: float, fs: float) -> np.ndarray:
    r = 1 - math.exp(-2 * math.pi * fc / fs)
    return np.array([[r, 0.0, 0.0, 1.0, -(1 - r), 0.0]])


def filter_sos(kind: str, params: FilterParams) -> np.ndarray:
    """Second-order-sections form of each temporal filter in the FPGA position."""
    if kind == "iir1":
        return dolp_sos(params)
    if kind == "iir1x2":
        return np.vstack([dolp_sos(params)] * 2)
    if kind == "butter2":
        return sps.butter(2, [params.fc_lo, params.fc_hi], btype="band", fs=params.fs, output="sos")
    raise ValueError(kind)


def post_sos(cfg: EnhancedConfig, fs: float) -> np.ndarray | None:
    if cfg.post_temporal == "lp":
        return lp1_sos(cfg.post_lp_hz, fs)
    if cfg.post_temporal == "butter":
        lo, hi = cfg.post_band
        return sps.butter(2, [lo, hi], btype="band", fs=fs, output="sos")
    return None


def cascade_sos(cfg: EnhancedConfig, params: FilterParams) -> np.ndarray:
    s, ps = filter_sos(cfg.filter, params), post_sos(cfg, params.fs)
    return s if ps is None else np.vstack([s, ps])


def sos_peak(sos: np.ndarray, fs: float) -> tuple[float, float]:
    """(peak |H|, frequency of the peak in Hz) on a dense grid."""
    f, h = sps.sosfreqz(sos, worN=8192, fs=fs)
    i = int(np.argmax(np.abs(h)))
    return float(np.abs(h[i])), float(f[i])


def peak_gain(kind: str, params: FilterParams) -> tuple[float, float]:
    return sos_peak(filter_sos(kind, params), params.fs)


def color_matrix(cfg: EnhancedConfig) -> np.ndarray | None:
    """3x3 BGR->BGR render matrix (None if identity)."""
    if cfg.color_space == "pulse_dir":
        # s = w.d ; render -s along the display colour (blood in -> G down -> redder)
        t = -np.outer(np.asarray(cfg.pulse_colour, float), PULSE_DIR_W)
    elif cfg.color_space == "bgr":
        t = np.eye(3)
    else:
        m = _BGR_TO[cfg.color_space]
        d = np.diag([cfg.luma_gain, cfg.chroma_attenuation, cfg.chroma_attenuation])
        t = np.linalg.inv(m) @ d @ m
    if cfg.channels == "green":
        t = np.diag([0.0, 1.0, 0.0]) @ t
    if np.allclose(t, np.eye(3), atol=1e-12):
        return None
    return t.astype(np.float32)


# --------------------------------------------------------------------------------
# Temporal filters (per pixel, per channel, any array shape)
# --------------------------------------------------------------------------------
class _IIR1:
    """Contract filter, same float32 arithmetic as FloatEVM."""

    def __init__(self, params: FilterParams):
        self.set_params(params)
        self.l_hi = self.l_lo = None

    def set_params(self, p: FilterParams) -> None:
        self.r_hi, self.r_lo = np.float32(p.r_hi), np.float32(p.r_lo)

    def init(self, x: np.ndarray) -> None:
        self.l_hi, self.l_lo = x.copy(), x.copy()

    def step(self, x: np.ndarray) -> np.ndarray:
        self.l_hi += self.r_hi * (x - self.l_hi)
        self.l_lo += self.r_lo * (x - self.l_lo)
        return self.l_hi - self.l_lo

    def band(self) -> np.ndarray:
        return self.l_hi - self.l_lo

    def pull(self, x: np.ndarray, k: float) -> None:
        self.l_hi += np.float32(k) * (x - self.l_hi)
        self.l_lo += np.float32(k) * (x - self.l_lo)


class _IIR1x2:
    """Two cascaded difference-of-lowpass stages (sharper skirts)."""

    def __init__(self, params: FilterParams):
        self.a, self.b = _IIR1(params), _IIR1(params)
        self._band = None

    def set_params(self, p: FilterParams) -> None:
        self.a.set_params(p)
        self.b.set_params(p)

    def init(self, x: np.ndarray) -> None:
        self.a.init(x)
        self.b.init(np.zeros_like(x))
        self._band = np.zeros_like(x)

    def step(self, x: np.ndarray) -> np.ndarray:
        self._band = self.b.step(self.a.step(x))
        return self._band

    def band(self) -> np.ndarray:
        return self._band

    def pull(self, x: np.ndarray, k: float) -> None:
        self.a.pull(x, k)
        self.b.pull(np.zeros_like(x), k)
        self._band = self.b.band()


class _SOS:
    """Per-pixel cascade of biquads (transposed direct form II, float32 state).

    init() sets the steady state for the current input (scipy sosfilt_zi), so
    a constant input yields exactly zero output from the first frame.
    """

    def __init__(self, sos: np.ndarray):
        self.set_sos(sos)
        self.z = None
        self._out = None

    def set_sos(self, sos: np.ndarray) -> None:
        self.sos = np.asarray(sos, float)
        self.coef = [tuple(np.float32(c) for c in (s[0], s[1], s[2], s[4], s[5])) for s in self.sos]
        self.zi = sps.sosfilt_zi(self.sos).astype(np.float32)  # unit-step steady state

    def init(self, x: np.ndarray) -> None:
        self.z = [[self.zi[i, 0] * x, self.zi[i, 1] * x] for i in range(len(self.coef))]
        self._out = np.zeros_like(x)

    def step(self, x: np.ndarray) -> np.ndarray:
        y = x
        for (b0, b1, b2, a1, a2), z in zip(self.coef, self.z):
            xin = y
            y = b0 * xin + z[0]
            z[0] = b1 * xin - a1 * y + z[1]
            z[1] = b2 * xin - a2 * y
        self._out = y
        return y

    def band(self) -> np.ndarray:
        return self._out

    def pull(self, x: np.ndarray, k: float) -> None:
        kf = np.float32(k)
        for i, z in enumerate(self.z):
            for j in range(2):
                z[j] += kf * (self.zi[i, j] * x - z[j])
        self._out = self._out * np.float32(1 - k)


class _Butter2(_SOS):
    def __init__(self, params: FilterParams):
        super().__init__(filter_sos("butter2", params))

    def set_params(self, p: FilterParams) -> None:
        self.set_sos(filter_sos("butter2", p))


def _make_filter(kind: str, params: FilterParams):
    return {"iir1": _IIR1, "iir1x2": _IIR1x2, "butter2": _Butter2}[kind](params)


def _blur(img: np.ndarray, levels: int) -> np.ndarray:
    small = img
    for _ in range(levels):
        small = cv2.pyrDown(small)
    return upscale(small, levels, img.shape[:2])


# --------------------------------------------------------------------------------
# Processor
# --------------------------------------------------------------------------------
class EnhancedEVM:
    """FloatEVM-compatible processor with switchable enhancements."""

    def __init__(self, params: FilterParams | None = None, config: EnhancedConfig | None = None):
        self.params = params or FilterParams()
        self.params.validate()
        self.config = config or EnhancedConfig()
        self.config.validate()
        cfg = self.config
        self._filt = _make_filter(cfg.filter, self.params)
        self._post = None
        ps = post_sos(cfg, self.params.fs)
        if ps is not None:
            self._post = _SOS(ps)
        self._cmat = color_matrix(cfg)
        self._shape = None
        self._post_state = None
        self._prev_luma: np.ndarray | None = None
        self._motion_hint: float | None = None
        self._gate = 1.0
        self._still = 10**9  # frames since last motion
        self.last_motion = 0.0
        self.frames_since_init = 0
        self._update_gain()

    # -- interface ----------------------------------------------------------------
    def set_params(self, params: FilterParams) -> None:
        """Change cutoffs/gain live; filter state kept, coefficients recomputed."""
        params.validate()
        self.params = params
        self._filt.set_params(params)
        ps = post_sos(self.config, params.fs)
        if ps is not None:
            self._post.set_sos(ps)
        self._update_gain()

    def reset(self) -> None:
        """INIT flag: re-initialise from the next frame."""
        self._shape = None

    def set_motion(self, value: float) -> None:
        """External motion hint for the next frame (e.g. mean |dLuma| on the face
        mask from the face tracker), used instead of the whole-frame estimate.
        Same units as motion_threshold."""
        self._motion_hint = float(value)

    @property
    def gate(self) -> float:
        return self._gate

    @property
    def alpha_hw(self) -> float:
        """Value to write to the FPGA alpha register (before the motion gate)."""
        return self._alpha_hw

    @property
    def total_peak_gain(self) -> float:
        """End-to-end amplification at the cascade's passband peak (no clipping)."""
        return self._alpha_hw * self._post_mult * self._peak

    def band_state(self) -> np.ndarray | None:
        """Un-amplified temporal band (FPGA-position filter), BGR float32, input size."""
        if self._shape is None:
            return None
        b = self._filt.band()
        if self.config.extra_levels:
            b = upscale(b, self.config.extra_levels, self._shape[:2])
        return b

    def process(self, small_bgr: np.ndarray) -> np.ndarray:
        if small_bgr.dtype != np.uint8 or small_bgr.ndim != 3 or small_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8, got {small_bgr.shape} {small_bgr.dtype}")
        cfg = self.config
        x = small_bgr.astype(np.float32)
        xf = x
        for _ in range(cfg.extra_levels):
            xf = cv2.pyrDown(xf)

        motion = self._measure_motion(small_bgr) if cfg.motion_gate else 0.0
        if self._shape != small_bgr.shape:
            self._shape = small_bgr.shape
            self._filt.init(xf)
            self._post_state = None
            self._gate, self._still = 1.0, 10**9
            self.frames_since_init = 0
        elif cfg.motion_gate:
            self._update_gate(motion, xf)

        # ---- FPGA position ----
        band = self._filt.step(xf)
        self.frames_since_init += 1
        d = np.clip(np.float32(self._alpha_hw * self._gate) * band, -cfg.stage_clip, cfg.stage_clip)
        if not self._has_post:
            return d if cfg.clip >= cfg.stage_clip else np.clip(d, -cfg.clip, cfg.clip)

        # ---- laptop post-processing ----
        if cfg.extra_levels:
            d = upscale(d, cfg.extra_levels, small_bgr.shape[:2])
        if self._cmat is not None:
            d = cv2.transform(d, self._cmat)
        if cfg.post_blur_levels:
            d = _blur(d, cfg.post_blur_levels)
        if self._post is not None:
            if self._post_state is None:
                self._post.init(np.zeros_like(d))  # input is already band-limited
                self._post_state = True
            d = self._post.step(d)
        if self._post_mult != 1.0:
            d = d * np.float32(self._post_mult)
        if cfg.soft_clip:
            t = np.float32(cfg.soft_clip)
            d = t * np.tanh(d / t)
        return np.clip(d, -cfg.clip, cfg.clip).astype(np.float32, copy=False)

    # -- internals ----------------------------------------------------------------
    def _update_gain(self) -> None:
        cfg, p = self.config, self.params
        self._peak, self._peak_f = sos_peak(cascade_sos(cfg, p), p.fs)
        total = p.alpha * cfg.post_gain / (self._peak if cfg.gain_normalise else 1.0)
        self._alpha_hw = cfg.hw_alpha if cfg.hw_alpha else p.alpha
        self._post_mult = total / self._alpha_hw if self._alpha_hw > 0 else 0.0
        if math.isclose(self._post_mult, 1.0, rel_tol=1e-12):
            self._post_mult = 1.0
        self._has_post = bool(
            cfg.extra_levels or self._cmat is not None or cfg.post_blur_levels
            or self._post is not None or self._post_mult != 1.0 or cfg.soft_clip
        )
        self.name = f"enh[{cfg.short()}]"

    def _measure_motion(self, small_bgr: np.ndarray) -> float:
        luma = cv2.cvtColor(cv2.pyrDown(small_bgr), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._motion_hint is not None:
            m, self._motion_hint = self._motion_hint, None
        elif self._prev_luma is None or self._prev_luma.shape != luma.shape:
            m = 0.0
        else:
            m = float(np.mean(np.abs(luma - self._prev_luma)))
        self._prev_luma = luma
        self.last_motion = m
        return m

    def _reinit(self, xf: np.ndarray) -> None:
        cfg = self.config
        if cfg.reset_strength >= 1:
            self._filt.init(xf)  # == contract INIT register
        else:
            self._filt.pull(xf, cfg.reset_strength)  # NON-CONTRACT partial reset
        self._post_state = None

    def _update_gate(self, motion: float, xf: np.ndarray) -> None:
        cfg = self.config
        if motion > cfg.motion_threshold:
            self._gate, self._still = 0.0, 0  # gain -> 0 at once
            if cfg.soft_reset:
                self._reinit(xf)
            return
        self._still += 1
        if self._still == cfg.motion_hold_frames and cfg.soft_reset:
            self._reinit(xf)  # one more INIT once motion has settled
        self._gate = float(np.clip((self._still - cfg.motion_hold_frames) / cfg.motion_ramp_frames, 0.0, 1.0))


# --------------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------------
def _preset(**kw) -> Callable[[FilterParams], EnhancedEVM]:
    cfg = EnhancedConfig(**kw)
    cfg.validate()

    def make(params: FilterParams | None = None) -> EnhancedEVM:
        return EnhancedEVM(params, replace(cfg))

    make.config = cfg  # type: ignore[attr-defined]
    return make


PRESETS: dict[str, Callable[[FilterParams], EnhancedEVM]] = {
    # exactly FloatEVM (bit-for-bit, float)
    "contract": _preset(),
    # CONTRACT-SAFE: blur the FPGA output 2 pyramid levels (== filtering at 40x30)
    "pooled": _preset(post_blur_levels=2),
    # CONTRACT-SAFE: + YIQ render, chroma kept, luma partly suppressed
    "pooled_yiq": _preset(post_blur_levels=2, color_space="yiq", luma_gain=0.6, chroma_attenuation=1.0),
    # NON-CONTRACT: 4th-order Butterworth in the FPGA position + output pooling
    "butter_pooled": _preset(filter="butter2", post_blur_levels=2),
    # CONTRACT-SAFE best combination (synthetic: +25 dB pixel SNR vs contract):
    # split gain (alpha_hw 25 register, rest on laptop), pool 2 levels, laptop
    # Butterworth 0.75-2.2 Hz post-filter, YIQ render with chroma halved,
    # normalised gain, tanh limiter, motion gate with INIT + 20-frame ramp.
    "best_safe": _preset(
        hw_alpha=25.0,
        post_blur_levels=2,
        post_temporal="butter",
        color_space="yiq",
        luma_gain=1.0,
        chroma_attenuation=0.5,
        gain_normalise=True,
        soft_clip=30.0,
        clip=40.0,
        motion_gate=True,
    ),
    # NON-CONTRACT best overall: the only deviation is the 4th-order Butterworth
    # in the FPGA filter position; 3-level output pooling + the safe extras.
    "best": _preset(
        filter="butter2",
        post_blur_levels=3,
        color_space="yiq",
        luma_gain=1.0,
        chroma_attenuation=0.5,
        gain_normalise=True,
        soft_clip=30.0,
        clip=40.0,
        motion_gate=True,
    ),
}


def main(argv: list[str] | None = None) -> int:
    import sys

    from dashboard import run_dashboard

    argv = list(sys.argv[1:] if argv is None else argv)
    preset = argv.pop(0) if argv and argv[0] in PRESETS else "best_safe"
    from evm_float import FloatEVM

    return run_dashboard(FloatEVM, ["--preset", preset, *argv])


if __name__ == "__main__":
    raise SystemExit(main())
