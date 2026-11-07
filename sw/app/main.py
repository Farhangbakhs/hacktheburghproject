"""Pulse magnification app: original | magnified | BPM.

    python main.py                         # webcam 0, local backend
    python main.py --backend pynq --host 192.168.2.99
    python main.py --video clip.mp4        # file instead of webcam (loops)
    python main.py --record demo.mp4       # also save what's on screen

Keys:  1 local   2 mock server   3 PYNQ   4 simulated FPGA   r reset filter + BPM
       m face-only magnification   n noise smoothing
       e toggle auto-exposure    [ ] exposure down/up    s camera driver dialog
       v start/stop recording    f fullscreen
       q / Esc quit
Sliders: alpha (gain), low / high cutoff in centi-Hz.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from bpm_live import BpmEstimator
from processors import PYNQ_DEFAULT_IP, make_processor
from protocol import DEFAULT_ALPHA, DEFAULT_FC_HI, DEFAULT_FC_LO, FPS, PORT

WIN = "Pulse"
BACKEND_KEYS = {ord("1"): "local", ord("2"): "mock", ord("3"): "pynq", ord("4"): "mockfpga"}


# ------------------------------------------------------------------ pipeline

def downscale(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Gaussian pyramid down while it stays >= target, then resize to exactly (w, h)."""
    small = frame
    while small.shape[1] // 2 >= w and small.shape[0] // 2 >= h:
        small = cv2.pyrDown(small)
    if small.shape[1] != w or small.shape[0] != h:
        small = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
    return small


def face_mask(shape, face, feather: float = 0.2) -> np.ndarray:
    """Soft elliptical weight (0..1, float32, size of `shape`) covering the face box.

    Built at quarter resolution and upscaled, so it's cheap enough per frame.
    """
    h, w = shape[:2]
    s = 4
    small = np.zeros((h // s, w // s), np.float32)
    x, y, fw, fh = (v / s for v in face)
    centre = (int(x + fw / 2), int(y + fh * 0.55))
    axes = (int(fw * 0.6), int(fh * 0.8))
    cv2.ellipse(small, centre, axes, 0, 0, 360, 1.0, -1)
    k = int(fw * feather) | 1
    small = cv2.GaussianBlur(small, (k * 2 + 1, k * 2 + 1), k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def magnify(frame: np.ndarray, signal: np.ndarray, mask: np.ndarray = None,
            smooth: bool = True) -> np.ndarray:
    """Upscale the int8 signal to frame size and add it.

    smooth: blur the small signal first. Camera grain differs pixel to pixel
            while the pulse is spatially smooth, so this removes grain only.
    mask:   per-pixel weight (from face_mask) to fade magnification off the face.
    """
    sig = signal.astype(np.float32)
    if smooth:
        sig = cv2.GaussianBlur(sig, (5, 5), 1.2)
    up = cv2.resize(sig, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    if mask is not None:
        up *= mask[:, :, None]
    return np.clip(frame.astype(np.float32) + up, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ sources

class Camera:
    """Webcam with everything that could drift locked to fixed values.

    Probed on the team laptop's webcam (DirectShow): manual exposure and manual
    white balance work; gain is fixed at 0 (no auto-gain); no autofocus. Media
    Foundation ignored the white-balance lock, so DirectShow is used on Windows.

    Exposure is log2 seconds on DirectShow: -5 = 1/32 s, -6 = 1/64 s. Anything
    longer than one frame (-4 = 1/16 s) silently drops the camera to ~16 fps,
    which breaks the 30 fps the filter assumes, so -5 is the brightest usable.
    These settings persist in the driver after exit, so every one is set
    explicitly on open rather than trusting whatever the last program left.
    """

    MAX_EXPOSURE_30FPS = -5

    def __init__(self, index: int, width: int = 640, height: int = 480, lock: bool = True,
                 exposure: float = -5, wb_kelvin: float = 4500) -> None:
        self.windows = sys.platform == "win32"
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if self.windows else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open camera %d" % index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       # ignored if unsupported
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)       # backlight compensation is another auto-adjust
        if exposure > self.MAX_EXPOSURE_30FPS:
            print("[camera] warning: exposure %g is longer than a frame; camera will drop below 30 fps"
                  % exposure)
        self.exposure = exposure
        self.wb_kelvin = wb_kelvin
        self.auto_exposure = True
        self.set_auto_exposure(not lock)
        self.report()

    def set_auto_exposure(self, on: bool) -> None:
        """on: let the camera adapt (exposure + WB). off: lock to self.exposure / self.wb_kelvin."""
        if self.windows:  # DirectShow: 0.75 = auto, 0.25 = manual
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if on else 0.25)
        else:             # V4L2: 3 = aperture priority (auto), 1 = manual
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if on else 1)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if on else 0)
        if not on:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.wb_kelvin)
        self.auto_exposure = on

    def step_exposure(self, delta: float) -> None:
        self.exposure = min(self.exposure + delta, self.MAX_EXPOSURE_30FPS)
        if not self.auto_exposure:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def open_driver_settings(self) -> None:
        """DirectShow's own settings dialog: has power-line frequency (set 50 Hz in the UK)."""
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def report(self) -> None:
        g = self.cap.get
        for _ in range(15):  # first frames after a settings change come slowly; skip them
            self.cap.read()
        t = time.perf_counter()
        n = 0
        while time.perf_counter() - t < 1.0:  # measure what the camera really delivers
            if self.cap.read()[0]:
                n += 1
        fps = n / (time.perf_counter() - t)
        print("[camera] %dx%d, measured %.1f fps, exposure %s (%g), WB %s (%gK), gain %g"
              % (g(cv2.CAP_PROP_FRAME_WIDTH), g(cv2.CAP_PROP_FRAME_HEIGHT), fps,
                 "auto" if self.auto_exposure else "locked", g(cv2.CAP_PROP_EXPOSURE),
                 "auto" if self.auto_exposure else "locked", g(cv2.CAP_PROP_WB_TEMPERATURE),
                 g(cv2.CAP_PROP_GAIN)))
        if fps < FPS - 3:
            print("[camera] warning: below 30 fps. Shorten exposure (--exposure -6) or check USB.")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class VideoFile:
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path, self.loop = path, loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open video %s" % path)
        self.auto_exposure = None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and abs(fps - FPS) > 1:
            print("[video] warning: file is %.1f fps, filter assumes %.0f" % (fps, FPS))

    def set_auto_exposure(self, on: bool) -> None:
        pass

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


# ------------------------------------------------------------------ app

class App:
    def __init__(self, source, backend: str, host: str, size, headless: bool = False,
                 record: str = None, realtime: bool = True, port: int = PORT) -> None:
        self.port = port
        self.source = source
        self.host = host
        self.w, self.h = size
        self.headless = headless
        self.realtime = realtime
        self.alpha, self.fc_lo, self.fc_hi = DEFAULT_ALPHA, DEFAULT_FC_LO, DEFAULT_FC_HI
        self.bpm = BpmEstimator()
        self.proc = None
        self.backend = None
        self.message, self.message_until = "", 0.0
        self.fps = 0.0
        self.record_path = record
        self.writer = None
        self.fullscreen = False
        self.mask_face = True     # m: only magnify the face
        self.smooth = True        # n: spatially smooth the signal
        self._face_smooth = None
        self.frames = 0
        self.errors = 0
        if not self.switch_backend(backend) and backend != "local":
            self.switch_backend("local")

    # -- control

    def flash(self, msg: str, seconds: float = 3.0) -> None:
        print("[app] " + msg)
        self.message, self.message_until = msg, time.perf_counter() + seconds

    def switch_backend(self, kind: str) -> bool:
        try:
            new = make_processor(kind, host=self.host, port=self.port, alpha=self.alpha,
                                 fc_hi=self.fc_hi, fc_lo=self.fc_lo)
        except (OSError, ConnectionError) as e:
            self.flash("%s unavailable: %s" % (kind, e))
            return False
        if self.proc is not None:
            self.proc.close()
        self.proc, self.backend = new, kind
        self.flash("backend: %s" % kind)
        return True

    def reset(self) -> None:
        self.proc.reset()
        self.bpm.reset()
        self.flash("reset")

    def _make_ui(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        # Win32 backend: draw something first so the window exists before trackbars.
        cv2.imshow(WIN, np.zeros((480, 1540, 3), np.uint8))
        cv2.waitKey(1)
        cv2.createTrackbar("alpha", WIN, int(self.alpha), 200, lambda v: None)
        cv2.createTrackbar("low cHz", WIN, int(self.fc_lo * 100), 300, lambda v: None)
        cv2.createTrackbar("high cHz", WIN, int(self.fc_hi * 100), 500, lambda v: None)

    @staticmethod
    def _window_open() -> bool:
        try:
            return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def _read_sliders(self) -> None:
        try:
            alpha = cv2.getTrackbarPos("alpha", WIN)
            lo = max(cv2.getTrackbarPos("low cHz", WIN), 10) / 100.0
            hi = max(cv2.getTrackbarPos("high cHz", WIN), 20) / 100.0
        except cv2.error:
            return
        if hi <= lo + 0.1:
            hi = lo + 0.1
        if (alpha, lo, hi) != (self.alpha, self.fc_lo, self.fc_hi):
            self.alpha, self.fc_lo, self.fc_hi = alpha, lo, hi
            self.proc.set_params(alpha=alpha, fc_lo=lo, fc_hi=hi)

    def _toggle_record(self, size) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.flash("recording saved: %s" % self.record_path)
            return
        if not self.record_path:
            self.record_path = time.strftime("pulse_%H%M%S.mp4")
        self.writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        self.flash("recording to %s" % self.record_path)

    # -- drawing

    def _compose(self, frame, magnified, small, signal) -> np.ndarray:
        dh = 480
        scale = dh / frame.shape[0]
        dw = int(round(frame.shape[1] * scale))
        left = cv2.resize(frame, (dw, dh))
        right = cv2.resize(magnified, (dw, dh))

        colour = (0, 220, 0) if self.bpm.face is not None else (0, 200, 255)
        for roi in self.bpm.rois:
            x, y, rw, rh = (int(v * scale) for v in roi)
            cv2.rectangle(left, (x, y), (x + rw, y + rh), colour, 1)

        self._label(left, "original", (10, 28))
        self._label(right, "magnified x%d" % self.alpha, (10, 28))

        panel = np.full((dh, 260, 3), 24, np.uint8)
        y = 50
        if self.bpm.bpm is not None:
            # red when the latest estimate is confident; grey "?" when it's stale
            ok = self.bpm.reliable
            cv2.putText(panel, "%d%s" % (round(self.bpm.bpm), "" if ok else "?"), (20, y + 60),
                        cv2.FONT_HERSHEY_DUPLEX, 2.6, (80, 80, 255) if ok else (120, 120, 120),
                        3, cv2.LINE_AA)
            cv2.putText(panel, "BPM", (180, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "BPM ...", (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (200, 200, 200), 2, cv2.LINE_AA)
            cv2.rectangle(panel, (20, y + 60), (20 + int(220 * self.bpm.fill), y + 66),
                          (80, 80, 255), -1)
        lines = [
            "backend  %s" % self.backend,
            "fps      %.1f" % self.fps,
            "process  %.1f ms" % self.proc.last_latency_ms,
            "size     %dx%d" % (self.w, self.h),
            "band     %.2f-%.2f Hz" % (self.fc_lo, self.fc_hi),
            "face     %s" % ("yes" if self.bpm.face is not None else "no"),
            "signal   %.1f" % self.bpm.confidence,
            "mask %s  smooth %s" % ("on" if self.mask_face else "off", "on" if self.smooth else "off"),
        ]
        if getattr(self.source, "auto_exposure", None) is not None:
            lines.append("exposure %s" % ("auto" if self.source.auto_exposure
                                          else "locked %g" % self.source.exposure))
        if self.writer is not None:
            lines.append("REC")
        for i, s in enumerate(lines):
            cv2.putText(panel, s, (20, 200 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255) if s == "REC" else (220, 220, 220), 1, cv2.LINE_AA)
        if self.fps and self.fps < FPS - 5:
            cv2.putText(panel, "below 30 fps: band shifts", (20, dh - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        canvas = np.hstack([left, right, panel])
        if time.perf_counter() < self.message_until:
            self._label(canvas, self.message, (10, dh - 16))
        return canvas

    @staticmethod
    def _label(img, text, org) -> None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # -- main loop

    def step(self, frame: np.ndarray, t: float) -> np.ndarray:
        small = downscale(frame, self.w, self.h)
        try:
            signal = self.proc.process(small)
        except (OSError, ConnectionError) as e:
            self.errors += 1
            self.flash("%s failed (%s), falling back to local" % (self.backend, e))
            self.switch_backend("local")
            signal = self.proc.process(small)
        self.bpm.update(frame, t)
        mask = None
        if self.mask_face and self.bpm.face is not None:
            # smooth the face box so the mask doesn't jump each time detection reruns
            f = np.asarray(self.bpm.face, np.float32)
            self._face_smooth = f if self._face_smooth is None else 0.8 * self._face_smooth + 0.2 * f
            mask = face_mask(frame.shape, self._face_smooth)
        return self._compose(frame, magnify(frame, signal, mask, self.smooth), small, signal)

    def run(self, max_frames: int = None) -> None:
        if not self.headless:
            self._make_ui()
        last = time.perf_counter()
        t_media = 0.0
        try:
            while max_frames is None or self.frames < max_frames:
                frame = self.source.read()
                if frame is None:
                    break
                now = time.perf_counter()
                dt, last = now - last, now
                if dt > 0:
                    self.fps = 1.0 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
                # BPM timestamps: wall clock live, nominal 30 fps for files/headless
                t_media = now if self.realtime else t_media + 1.0 / FPS

                if not self.headless:
                    self._read_sliders()
                canvas = self.step(frame, t_media)
                self.frames += 1

                if self.record_path and self.writer is None and self.frames == 1:
                    self._toggle_record((canvas.shape[1], canvas.shape[0]))
                if self.writer is not None:
                    self.writer.write(canvas)
                if self.headless:
                    continue

                cv2.imshow(WIN, canvas)
                wait_ms = 1
                if not self.realtime:  # video file: play at 30 fps, not as fast as possible
                    wait_ms = max(1, int(1000 * (1.0 / FPS - (time.perf_counter() - now))))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27) or not self._window_open():
                    break
                if key in BACKEND_KEYS:
                    self.switch_backend(BACKEND_KEYS[key])
                elif key == ord("r"):
                    self.reset()
                elif key == ord("e"):
                    self.source.set_auto_exposure(not self.source.auto_exposure)
                    self.reset()
                elif key in (ord("["), ord("]")) and hasattr(self.source, "step_exposure"):
                    self.source.step_exposure(-1 if key == ord("[") else 1)
                    self.flash("exposure %g%s" % (self.source.exposure,
                               " (max for 30 fps)" if self.source.exposure >= Camera.MAX_EXPOSURE_30FPS else ""))
                    self.reset()
                elif key == ord("s") and hasattr(self.source, "open_driver_settings"):
                    self.source.open_driver_settings()
                elif key == ord("m"):
                    self.mask_face = not self.mask_face
                    self.flash("face-only magnification %s" % ("on" if self.mask_face else "off"))
                elif key == ord("n"):
                    self.smooth = not self.smooth
                    self.flash("noise smoothing %s" % ("on" if self.smooth else "off"))
                elif key == ord("v"):
                    self._toggle_record((canvas.shape[1], canvas.shape[0]))
                elif key == ord("f"):
                    self.fullscreen = not self.fullscreen
                    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        finally:
            if self.writer is not None:
                self.writer.release()
            self.proc.close()
            self.source.release()
            if not self.headless:
                cv2.destroyAllWindows()


def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["local", "mock", "pynq", "mockfpga"], default="local")
    ap.add_argument("--host", default=PYNQ_DEFAULT_IP, help="PYNQ IP address")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-size", type=parse_size, default=(640, 480))
    ap.add_argument("--video", help="video file instead of the webcam")
    ap.add_argument("--size", type=parse_size, default=(160, 120), help="processing size, WxH")
    ap.add_argument("--no-lock", action="store_true", help="leave auto-exposure/WB on")
    ap.add_argument("--exposure", type=float, default=-5,
                    help="locked exposure, log2 s (-5 = 1/32 s, brightest that keeps 30 fps)")
    ap.add_argument("--wb", type=float, default=4500, help="locked white balance, kelvin")
    ap.add_argument("--record", help="save the display to this .mp4")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--frames", type=int, help="stop after N frames")
    args = ap.parse_args(argv)

    if args.video:
        source = VideoFile(args.video, loop=not args.headless)
    else:
        source = Camera(args.camera, *args.cam_size, lock=not args.no_lock,
                        exposure=args.exposure, wb_kelvin=args.wb)
    app = App(source, args.backend, args.host, args.size, headless=args.headless,
              record=args.record, realtime=not (args.video or args.headless))
    app.run(args.frames)
    bpm = app.bpm.bpm
    print("[app] %d frames, last fps %.1f, BPM %s, backend errors %d"
          % (app.frames, app.fps, "%.1f" % bpm if bpm else "-", app.errors))


if __name__ == "__main__":
    main()
