"""Live test dashboard for the EVM reference implementations (OpenCV window).

Layout:  [ original | magnified ]  +  pulse-trace strip  +  status bar

Keys:
  q / Esc  quit                 r  reset filter state (INIT)
  v        cycle view           e  toggle exposure lock
  d        driver settings (DirectShow only; untick auto exposure/WB there)
  f        toggle fs = measured fps (default: fixed 30 fps per contract)
  m        toggle mirror        s  save screenshot to recordings/
  [ / ]    fewer / more pyramid levels (coarser = less noise; contract is 160x120)
  p        next processor preset (contract, pooled, ..., best_safe, best)
  k        toggle skin mask on the amplified signal (amplify face skin only)
  o        toggle skin-mask overlay        x  re-meter exposure on face
  h        toggle help

Trackbars: alpha, low cutoff, high cutoff (Hz x100).
The trace is the un-amplified bandpass green channel averaged over the face
skin mask (centre box if no face). A pulse shows as a regular ~1 Hz wave.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from camera import CameraConfig, FrameSource
from face_tracker import FaceResult, FaceTracker
from face_tracker import draw as draw_face
from evm_common import (
    ALPHA_DEFAULT,
    FC_HI_DEFAULT,
    FC_LO_DEFAULT,
    FS_DEFAULT,
    FilterParams,
    add_signal,
    downscale,
    pyramid_levels_for,
    upscale,
)

WIN = "EVM pulse - reference"
VIEWS = ("side-by-side", "magnified", "signal")
TRACE_SECONDS = 10.0
STRIP_H = 150
BAR_H = 54
PANEL_W = 640

GREEN = (90, 220, 90)
AMBER = (0, 190, 255)
RED = (60, 60, 240)
GREY = (170, 170, 170)
WHITE = (240, 240, 240)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EVM pulse magnification test dashboard")
    p.add_argument("--source", default="0", help="camera index or video file (default 0)")
    p.add_argument("--backend", default=None, help="dshow | msmf | auto (default dshow on Windows)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=FS_DEFAULT)
    p.add_argument("--levels", type=int, default=None, help="pyramid levels (default: auto to ~160 wide)")
    p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    p.add_argument("--fc-lo", type=float, default=FC_LO_DEFAULT)
    p.add_argument("--fc-hi", type=float, default=FC_HI_DEFAULT)
    p.add_argument("--no-lock", action="store_true", help="leave auto exposure / WB on")
    p.add_argument("--exposure", type=float, default=None, help="manual exposure (driver units)")
    p.add_argument("--wb", type=float, default=None, help="manual white balance (K)")
    p.add_argument("--no-mjpg", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--preset", default=None, help="processor preset from evm_enhanced.PRESETS")
    p.add_argument("--no-mask", action="store_true", help="amplify whole frame, not just skin")
    p.add_argument("--no-meter", action="store_true", help="skip face-metered exposure at start")
    p.add_argument("--headless", type=int, default=0, metavar="N",
                   help="process N frames with no window and print stats (tests/benchmarks)")
    return p


class PulseTrace:
    """Rolling (t, value) buffer drawn as an auto-scaled line plot."""

    def __init__(self, seconds: float = TRACE_SECONDS):
        self.seconds = seconds
        self.buf: deque[tuple[float, float]] = deque()

    def add(self, t: float, v: float) -> None:
        self.buf.append((t, v))
        while self.buf and t - self.buf[0][0] > self.seconds:
            self.buf.popleft()

    def clear(self) -> None:
        self.buf.clear()

    def values(self) -> np.ndarray:
        return np.array([v for _, v in self.buf], dtype=np.float32)

    def draw(self, canvas: np.ndarray, label: str) -> None:
        h, w = canvas.shape[:2]
        canvas[:] = (24, 24, 24)
        for s in range(1, int(self.seconds)):
            x = int(w * s / self.seconds)
            cv2.line(canvas, (x, 0), (x, h), (45, 45, 45), 1)
        cv2.line(canvas, (0, h // 2), (w, h // 2), (60, 60, 60), 1)
        cv2.putText(canvas, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
        if len(self.buf) < 2:
            return
        t = np.array([p[0] for p in self.buf])
        v = np.array([p[1] for p in self.buf])
        v = v - v.mean()
        scale = max(np.abs(v).max(), 1e-3)
        xs = (w - 1) - (t[-1] - t) / self.seconds * (w - 1)
        ys = h / 2 - v / scale * (h / 2 - 22)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, GREEN, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"+/-{scale:.3f} px", (w - 110, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)


def roi_box(h: int, w: int) -> tuple[int, int, int, int]:
    """Central box (x0, y0, x1, y1) where the face is expected; step 5 swaps in a detector."""
    return int(w * 0.35), int(h * 0.2), int(w * 0.65), int(h * 0.65)


def _fit(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def _text(img, s, org, color=WHITE, scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class Dashboard:
    def __init__(self, processor_factory: Callable, args: argparse.Namespace):
        self.args = args
        cfg = CameraConfig(
            source=args.source, width=args.width, height=args.height, fps=args.fps,
            mjpg=not args.no_mjpg, lock_exposure=not args.no_lock, exposure=args.exposure,
            lock_wb=not args.no_lock, wb_temperature=args.wb,
        )
        if args.backend:
            cfg.backend = args.backend
        self.src = FrameSource(cfg)
        print(self.src.report)
        self.params = FilterParams(args.fc_lo, args.fc_hi, args.alpha, args.fps)
        self.presets = load_presets(processor_factory)
        self.preset_name = args.preset or getattr(processor_factory, "preset_name", None) or next(iter(self.presets))
        if self.preset_name not in self.presets:
            raise SystemExit(f"unknown preset {self.preset_name!r}; choose from {list(self.presets)}")
        self.evm = self.presets[self.preset_name](self.params)
        self.levels = args.levels
        self.view = 0
        self.mirror = not args.no_mirror
        self.show_help = False
        self.adapt_fs = False
        self.exposure_locked = not args.no_lock
        self.trace = PulseTrace()
        self.face = FaceTracker()
        self.face_res = FaceResult(found=False)
        self.use_mask = not args.no_mask
        self.show_mask = False
        self._prev_face_luma: np.ndarray | None = None
        if (not args.no_meter and args.exposure is None and not args.no_lock
                and not self.src.is_file and not args.headless):
            self.meter()
        self.proc_ms: deque[float] = deque(maxlen=30)
        self.warning = ""
        self.t_init = time.perf_counter()

    # -- controls -----------------------------------------------------------
    def create_window(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.createTrackbar("alpha", WIN, int(self.params.alpha), 200, lambda _: None)
        cv2.createTrackbar("lo Hz x100", WIN, int(self.params.fc_lo * 100), 300, lambda _: None)
        cv2.createTrackbar("hi Hz x100", WIN, int(self.params.fc_hi * 100), 400, lambda _: None)

    def poll_trackbars(self) -> None:
        alpha = cv2.getTrackbarPos("alpha", WIN)
        lo = cv2.getTrackbarPos("lo Hz x100", WIN) / 100
        hi = cv2.getTrackbarPos("hi Hz x100", WIN) / 100
        fs = self.src.measured_fps() if self.adapt_fs and self.src.measured_fps() > 5 else self.args.fps
        new = FilterParams(lo, hi, float(alpha), fs)
        try:
            new.validate()
        except ValueError:
            self.warning = "invalid band: need 0 < lo < hi"
            return
        self.warning = ""
        if (new.fc_lo, new.fc_hi, new.alpha, round(new.fs, 1)) != (
            self.params.fc_lo, self.params.fc_hi, self.params.alpha, round(self.params.fs, 1)
        ):
            self.params = new
            self.evm.set_params(new)

    def reset(self) -> None:
        self.evm.reset()
        self.trace.clear()
        self.t_init = time.perf_counter()

    def handle_key(self, key: int) -> bool:
        """Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord("r"):
            self.reset()
        elif key == ord("v"):
            self.view = (self.view + 1) % len(VIEWS)
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key in (ord("["), ord("]")) and self.levels is not None:
            self.levels = max(0, min(6, self.levels + (1 if key == ord("]") else -1)))
            self.reset()
        elif key == ord("p"):
            names = list(self.presets)
            self.preset_name = names[(names.index(self.preset_name) + 1) % len(names)]
            self.evm = self.presets[self.preset_name](self.params)
            self.reset()
        elif key == ord("k"):
            self.use_mask = not self.use_mask
        elif key == ord("o"):
            self.show_mask = not self.show_mask
        elif key == ord("x") and not self.src.is_file:
            self.meter()
            self.reset()
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("f"):
            self.adapt_fs = not self.adapt_fs
        elif key == ord("e") and not self.src.is_file:
            if self.exposure_locked:
                self.src.unlock_exposure()
            else:
                self.src.lock_exposure(self.args.exposure)
            self.exposure_locked = not self.exposure_locked
            self.reset()
        elif key == ord("d") and not self.src.is_file:
            if not self.src.open_driver_dialog():
                self.warning = "driver dialog needs --backend dshow"
        elif key == ord("s"):
            out = Path(__file__).parent / "recordings"
            out.mkdir(exist_ok=True)
            path = out / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), self._last_canvas)
            print(f"[dashboard] saved {path}")
        return True

    # -- per frame ------------------------------------------------------------
    def step(self, frame: np.ndarray, t: float) -> dict:
        if self.levels is None:
            self.levels = pyramid_levels_for(frame.shape[1])
        was_found = self.face_res.found
        self.face_res = res = self.face.update(frame, t)
        if res.found and not was_found and getattr(self.evm, "full_res", False):
            self.evm.reset()  # restart (and fade in) the filter once the face is acquired
        t0 = time.perf_counter()
        small = downscale(frame, self.levels)
        sh, sw = small.shape[:2]
        mask_small = res.mask_resized((sh, sw)) if res.found else None
        if hasattr(self.evm, "set_mask"):
            self.evm.set_mask(mask_small)
        if hasattr(self.evm, "set_motion") and mask_small is not None:
            self.evm.set_motion(self._face_motion(small, mask_small))
        if hasattr(self.evm, "set_mask_full"):
            self.evm.set_mask_full(res.skin_mask if (self.use_mask and res.found) else None)
        if getattr(self.evm, "full_res", False):   # e.g. Laplacian motion EVM
            signal_full = self.evm.process(frame).astype(np.float32, copy=True)
            signal = downscale(signal_full, self.levels)
        else:
            signal = self.evm.process(small)
            signal_full = upscale(signal.astype(np.float32), self.levels, frame.shape[:2])
        masked_inside = getattr(self.evm, "applies_mask", False)
        if self.use_mask and res.found and res.skin_mask is not None and not masked_inside:
            signal_full *= res.skin_mask[..., None]
        magnified = add_signal(frame, signal_full)
        self.proc_ms.append((time.perf_counter() - t0) * 1000)

        band = self.evm.band_state()
        if band is None:
            band = np.zeros_like(small, np.float32)
        elif band.shape[:2] != (sh, sw):
            band = downscale(band.astype(np.float32), self.levels)
        if mask_small is not None and mask_small.sum() > 1:
            self.trace.add(t, float((band[..., 1] * mask_small).sum() / mask_small.sum()))
        else:
            x0, y0, x1, y1 = roi_box(sh, sw)
            self.trace.add(t, float(band[y0:y1, x0:x1, 1].mean()))
        return {"small": small, "signal": signal, "signal_full": signal_full, "magnified": magnified}

    def _face_motion(self, small: np.ndarray, mask_small: np.ndarray) -> float:
        """Mean |delta luma| over face skin (levels) - motion hint for the gate."""
        luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self._prev_face_luma = self._prev_face_luma, luma
        if prev is None or prev.shape != luma.shape:
            return 0.0
        w = mask_small.sum()
        return float((np.abs(luma - prev) * mask_small).sum() / w) if w > 1 else 0.0

    def render(self, frame: np.ndarray, out: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        orig = frame.copy()
        if self.face_res.found:
            draw_face(orig, self.face_res, show_mask=self.show_mask)
        else:
            x0, y0, x1, y1 = roi_box(h, w)
            cv2.rectangle(orig, (x0, y0), (x1, y1), AMBER, 1)
        mag = out["magnified"]
        sig_vis = np.clip(out["signal_full"] + 128, 0, 255).astype(np.uint8)
        if self.mirror:
            orig, mag, sig_vis = (cv2.flip(i, 1) for i in (orig, mag, sig_vis))

        if VIEWS[self.view] == "side-by-side":
            top = np.hstack([_fit(orig, PANEL_W), _fit(mag, PANEL_W)])
            _text(top, "original", (10, 22))
            _text(top, f"magnified x{self.params.alpha:.0f}", (PANEL_W + 10, 22))
        elif VIEWS[self.view] == "magnified":
            top = _fit(mag, 2 * PANEL_W)
            _text(top, f"magnified x{self.params.alpha:.0f}", (10, 22))
        else:
            top = np.hstack([_fit(mag, PANEL_W), _fit(sig_vis, PANEL_W)])
            _text(top, "magnified", (10, 22))
            _text(top, "amplified signal (+128)", (PANEL_W + 10, 22))
        width = top.shape[1]

        strip = np.zeros((STRIP_H, width, 3), np.uint8)
        where = "face skin" if self.face_res.found else "centre box (no face found)"
        self.trace.draw(strip, f"bandpassed green, {where} (last 10 s)")

        bar = np.full((BAR_H, width, 3), 16, np.uint8)
        mfps = self.src.measured_fps()
        fps_ok = abs(mfps - self.params.fs) < 0.1 * self.params.fs
        ms = np.mean(self.proc_ms) if self.proc_ms else 0.0
        sh, sw = out["small"].shape[:2]
        settling = time.perf_counter() - self.t_init < 2.0
        lum = self._face_luma(frame)
        row1 = [
            (f"camera {mfps:4.1f} fps", GREEN if fps_ok else RED),
            (f"fs {self.params.fs:.1f}{' auto' if self.adapt_fs else ''}", WHITE),
            (f"band {self.params.fc_lo:.2f}-{self.params.fc_hi:.2f} Hz "
             f"({self.params.fc_lo * 60:.0f}-{self.params.fc_hi * 60:.0f} bpm)", WHITE),
            (f"alpha {self.params.alpha:.0f} (true gain)"
             if getattr(getattr(self.evm, "config", None), "gain_normalise", False)
             else f"alpha {self.params.alpha:.0f} (~x{0.41 * self.params.alpha:.0f} peak)", WHITE),
            (f"proc {ms:4.1f} ms", WHITE),
            (f"{w}x{h}->{sw}x{sh} L{self.levels}", GREY),
        ]
        row2 = [
            ("exposure LOCKED" if self.exposure_locked else "exposure AUTO",
             GREEN if self.exposure_locked else AMBER),
            ("settling..." if settling else "running", AMBER if settling else GREEN),
            (f"face {self.face.backend} {self.face_res.timing_ms:.0f}ms" if self.face_res.found else "no face",
             GREEN if self.face_res.found else AMBER),
            (f"preset {self.preset_name} [p]" + (f" | {self.evm.name}" if "locked" in self.preset_name else ""),
             GREEN if getattr(self.evm, "locked", True) else AMBER),
            ("mask ON [k]" if self.use_mask else "mask off [k]", GREEN if self.use_mask else GREY),
        ]
        gate = getattr(self.evm, "gate", None)
        if gate is not None and gate < 0.99:
            row2.append((f"output fading in {gate:.2f}" if getattr(self.evm, "full_res", False) else f"motion gate {gate:.2f}", AMBER))
        if lum is not None:
            row2.append((f"skin brightness {lum:.0f}" + (" - add light!" if lum < 70 else ""),
                         GREEN if lum >= 70 else RED))
        if mfps and not fps_ok and not self.adapt_fs:
            row2.append(("fps != fs: band shifted (press f or add light)", RED))
        if self.warning:
            row2.append((self.warning, RED))
        for y, row in ((20, row1), (44, row2)):
            x = 10
            for txt, col in row:
                _text(bar, txt, (x, y), col)
                x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 22
        _text(bar, "h: help", (width - 70, 44), GREY)

        canvas = np.vstack([top, strip, bar])
        if self.show_help:
            for i, line in enumerate((__doc__ or "").strip().splitlines()[2:17]):
                _text(canvas, line, (20, 60 + 20 * i), WHITE, 0.5)
        self._last_canvas = canvas
        return canvas

    def meter(self) -> None:
        """Expose for the face (brightest exposure keeping 30 fps, no skin clipping)."""
        def stats(f):
            r = self.face.update(f)
            m = r.skin_mask if r.found else None
            if m is None or m.sum() < 1:
                return None, 0.0
            skin = m > 0.5
            luma = float((cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) * m).sum() / m.sum())
            return luma, float((f[skin].max(axis=1) >= 250).mean())
        print("[dashboard] metering exposure on face...")
        res = self.src.meter_on_face(stats)
        print(f"[dashboard] exposure {res['exposure']} -> skin luma {res['luma']}, {res['fps']} fps")
        self.exposure_locked = True

    def _face_luma(self, frame: np.ndarray) -> float | None:
        m = self.face_res.skin_mask if self.face_res.found else None
        if m is None or m.sum() < 1:
            return None
        return float((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * m).sum() / m.sum())

    # -- loops -------------------------------------------------------------------
    def run(self) -> int:
        self.create_window()
        try:
            while True:
                ok, frame, t = self.src.read()
                if not ok:
                    print("[dashboard] source ended / read failed")
                    break
                self.poll_trackbars()
                out = self.step(frame, t)
                cv2.imshow(WIN, self.render(frame, out))
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    print("[dashboard] quit key pressed")
                    break
                # -1 means "property unsupported" on some builds; only 0 means closed.
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) == 0:
                    print("[dashboard] window closed")
                    break
        finally:
            self.src.release()
            cv2.destroyAllWindows()
        return 0

    def run_headless(self, n: int) -> int:
        t_start = time.perf_counter()
        frames = 0
        try:
            for _ in range(n):
                ok, frame, t = self.src.read()
                if not ok:
                    break
                out = self.step(frame, t)
                self.render(frame, out)
                frames += 1
        finally:
            self.src.release()
        wall = time.perf_counter() - t_start
        vals = self.trace.values()
        print(f"[headless] frames={frames} wall={wall:.2f}s "
              f"proc_ms={np.mean(self.proc_ms):.2f} source_fps={self.src.measured_fps():.1f} "
              f"trace_std={vals.std() if vals.size else 0:.4f}")
        return 0 if frames else 1


def load_presets(default_factory: Callable) -> dict[str, Callable]:
    """'contract' (the given factory, e.g. FloatEVM) + evm_enhanced.PRESETS."""
    presets: dict[str, Callable] = {"contract": default_factory}
    try:
        from evm_enhanced import PRESETS
    except Exception as e:  # keep the dashboard usable without it
        print(f"[dashboard] enhanced presets unavailable: {e}")
        return presets
    for name, fac in PRESETS.items():
        if name != "contract":
            presets[name] = fac
    try:
        from evm_locked import make_locked
        presets["locked"] = make_locked  # MIT-like: pulse-locked narrowband, pooled
        presets["locked_face"] = lambda p: make_locked(p, mix=1.0)   # face-wide pulse only
        presets["locked_pixel"] = lambda p: make_locked(p, mix=0.0)  # per-pixel FPGA path only
    except Exception as e:
        print(f"[dashboard] locked presets unavailable: {e}")
    try:
        from evm_laplacian import make_laplacian
        # saracen/magnification motion EVM (laptop-only, non-contract); face-masked by the dashboard
        presets["laplacian"] = lambda p: make_laplacian(p, levels=4, work_scale=1)        # 320x240, real-time
        presets["laplacian_full"] = make_laplacian                                       # 640x480, ~30 ms
        presets["laplacian_colour"] = lambda p: make_laplacian(p, levels=4, work_scale=1, chrom_attenuation=1.0)
    except Exception as e:
        print(f"[dashboard] laplacian presets unavailable: {e}")
    try:
        from evm_saracen_cpp import load_library, make_saracen
        load_library()
        # saracen/magnification C++ (embedded DLL), face-masked by the dashboard
        presets["saracen_cpp"] = make_saracen
        presets["saracen_cpp_colour"] = lambda p: make_saracen(p, chrom_attenuation=1.0)
        presets["saracen_cpp_smooth"] = lambda p: make_saracen(p, lambda_c=80.0)      # least grain
        presets["saracen_cpp_original"] = lambda p: make_saracen(p, levels=5, lambda_c=20.0, work_scale=0)
    except Exception as e:
        print(f"[dashboard] saracen C++ presets unavailable: {e}")
    try:
        from evm_fixed import FixedEVM
        from evm_saracen_fixed import make_saracen_fixed
        presets["fixed"] = FixedEVM                     # v1 contract, bit-exact (what the FPGA computes)
        presets["saracen_fixed"] = make_saracen_fixed   # v2 proposal: laptop pre/post + bit-exact kernel
    except Exception as e:
        print(f"[dashboard] fixed-point presets unavailable: {e}")
    return presets


def run_dashboard(processor_factory: Callable, argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dash = Dashboard(processor_factory, args)
    return dash.run_headless(args.headless) if args.headless else dash.run()
