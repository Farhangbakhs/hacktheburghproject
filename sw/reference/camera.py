"""Webcam / video source with best-effort locking of fps, exposure and white balance.

Pulse magnification needs a steady 30 fps and *no* automatic exposure / white
balance changes (those swamp the ~0.5% colour change of a heartbeat).
Drivers differ wildly, so every setting is read back and reported; check the
printout and record what worked in TUNING.md.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

# MSMF opens much faster with this off; must be set before cv2 import.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


@dataclass
class CameraConfig:
    source: str = "0"  # camera index or video file path
    backend: str = "dshow" if sys.platform == "win32" else "auto"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    mjpg: bool = True
    lock_exposure: bool = True
    exposure: float | None = None  # driver units; DSHOW uses log2 seconds e.g. -6
    lock_wb: bool = True
    wb_temperature: float | None = None  # Kelvin, e.g. 4500
    loop_video: bool = True


@dataclass
class CameraReport:
    backend: str = ""
    is_file: bool = False
    requested: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"[camera] backend={self.backend} file={self.is_file}"]
        for k, v in self.actual.items():
            req = self.requested.get(k, "")
            lines.append(f"  {k:<14} actual={v!s:<10} requested={req}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


class FrameSource:
    """Wraps cv2.VideoCapture. read() -> (ok, frame_bgr, timestamp_s)."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.is_file = not cfg.source.isdigit()
        self.report = CameraReport(is_file=self.is_file)
        self._t_hist: deque[float] = deque(maxlen=60)
        self._file_frame_idx = 0

        if self.is_file:
            if not os.path.exists(cfg.source):
                raise FileNotFoundError(cfg.source)
            self.cap = cv2.VideoCapture(cfg.source)
            self.report.backend = "file"
        else:
            api = BACKENDS.get(cfg.backend, cv2.CAP_ANY)
            self.cap = cv2.VideoCapture(int(cfg.source), api)
            self.report.backend = cfg.backend
            if not self.cap.isOpened() and api != cv2.CAP_ANY:
                self.report.notes.append(f"{cfg.backend} failed, falling back to auto")
                self.cap = cv2.VideoCapture(int(cfg.source), cv2.CAP_ANY)
                self.report.backend = "auto"
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video source {cfg.source!r}")
        if not self.is_file:
            self._configure()
        self._fill_actual()

    # -- configuration -----------------------------------------------------
    def _set(self, prop: int, value: float, name: str) -> bool:
        ok = self.cap.set(prop, value)
        self.report.requested[name] = value
        if not ok:
            self.report.notes.append(f"driver rejected {name}={value}")
        return ok

    def _configure(self) -> None:
        c = self.cfg
        if c.mjpg:
            self._set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"), "fourcc")
        self._set(cv2.CAP_PROP_FRAME_WIDTH, c.width, "width")
        self._set(cv2.CAP_PROP_FRAME_HEIGHT, c.height, "height")
        self._set(cv2.CAP_PROP_FPS, c.fps, "fps")
        self._set(cv2.CAP_PROP_BUFFERSIZE, 1, "buffersize")
        # Let auto-exposure settle on the scene before freezing it.
        if c.lock_exposure or c.lock_wb:
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self.cap.read()
        if c.lock_exposure:
            self.lock_exposure(c.exposure)
        if c.lock_wb:
            self._set(cv2.CAP_PROP_AUTO_WB, 0, "auto_wb")
            if c.wb_temperature is not None:
                self._set(cv2.CAP_PROP_WB_TEMPERATURE, c.wb_temperature, "wb_temperature")
        self._set(cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus")

    def lock_exposure(self, exposure: float | None = None) -> None:
        """Disable auto-exposure. Value conventions differ by backend:
        DSHOW/V4L2 use 0.25 = manual, 0.75 = auto; MSMF often uses 0/1.
        If no explicit exposure is given we re-write the current value, which
        on many DSHOW drivers is what actually freezes it."""
        manual = 0.25 if self.report.backend in ("dshow", "v4l2") else 0
        self._set(cv2.CAP_PROP_AUTO_EXPOSURE, manual, "auto_exposure")
        value = exposure if exposure is not None else self.cap.get(cv2.CAP_PROP_EXPOSURE)
        self._set(cv2.CAP_PROP_EXPOSURE, value, "exposure")

    def meter_on_face(self, skin_stats_fn, target_max: float = 190.0, fps_floor: float = 27.0,
                      max_saturated: float = 0.02, candidates: list[float] | None = None) -> dict:
        """Expose for the FACE, not the scene: pick the brightest manual exposure
        that keeps the frame rate and does not clip skin.

        Webcam auto-exposure meters the whole frame, so a bright window or
        ceiling lights behind the subject leave the face dark (we measured
        skin luma 53/255 that way). This tries candidates from brightest to
        darkest and keeps the first whose measured fps >= fps_floor, skin
        saturation <= max_saturated and skin luma <= target_max.

        skin_stats_fn(frame_bgr) -> (mean skin luma or None, saturated fraction).
        Returns {"exposure", "luma", "fps", "tried": [...]}. Blocks ~1 s per candidate.
        """
        if candidates is None:
            if self.report.backend in ("dshow", "v4l2"):
                # DSHOW: log2 seconds. -5 = 31 ms is the longest that still gives 30 fps.
                candidates = [-5, -6, -7, -8]
            else:
                cur = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                candidates = [cur]
        manual = 0.25 if self.report.backend in ("dshow", "v4l2") else 0
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
        tried, chosen = [], None
        for exp in candidates:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            for _ in range(8):  # let the new setting take effect
                self.cap.read()
            lumas, sats, t0, n = [], [], time.perf_counter(), 0
            while n < 20:
                ok, f = self.cap.read()
                if not ok:
                    break
                n += 1
                luma, sat = skin_stats_fn(f)
                if luma is not None:
                    lumas.append(luma)
                    sats.append(sat)
            fps = n / max(time.perf_counter() - t0, 1e-6)
            luma = float(np.mean(lumas)) if lumas else None
            sat = float(np.mean(sats)) if sats else 0.0
            tried.append({"exposure": exp, "luma": luma, "fps": round(fps, 1), "saturated": round(sat, 4)})
            ok_fps = fps >= fps_floor
            ok_clip = sat <= max_saturated and (luma is None or luma <= target_max)
            if ok_fps and ok_clip:
                chosen = exp
                break
        if chosen is None:  # nothing acceptable: fall back to the darkest tried
            chosen = candidates[-1]
            self.cap.set(cv2.CAP_PROP_EXPOSURE, chosen)
        self.report.requested["auto_exposure"] = manual
        self.report.requested["exposure"] = chosen
        self.report.notes.append(
            f"face-metered exposure={chosen} " + ", ".join(
                f"[{t['exposure']}: luma {t['luma'] if t['luma'] is None else round(t['luma'])}, "
                f"{t['fps']} fps, sat {t['saturated']:.1%}]" for t in tried))
        self._fill_actual()
        res = next(t for t in tried if t["exposure"] == chosen) if any(t["exposure"] == chosen for t in tried) else {}
        return {"exposure": chosen, "luma": res.get("luma"), "fps": res.get("fps"), "tried": tried}

    def unlock_exposure(self) -> None:
        auto = 0.75 if self.report.backend in ("dshow", "v4l2") else 1
        self._set(cv2.CAP_PROP_AUTO_EXPOSURE, auto, "auto_exposure")

    def open_driver_dialog(self) -> bool:
        """DSHOW only: opens the driver's own settings window (often the most
        reliable way to untick auto-exposure / auto-WB on Windows)."""
        return bool(self.cap.set(cv2.CAP_PROP_SETTINGS, 1))

    def _fill_actual(self) -> None:
        props = {
            "width": cv2.CAP_PROP_FRAME_WIDTH,
            "height": cv2.CAP_PROP_FRAME_HEIGHT,
            "fps": cv2.CAP_PROP_FPS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_wb": cv2.CAP_PROP_AUTO_WB,
            "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
            "gain": cv2.CAP_PROP_GAIN,
            "autofocus": cv2.CAP_PROP_AUTOFOCUS,
        }
        self.report.actual = {k: round(self.cap.get(p), 3) for k, p in props.items()}
        fcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.report.actual["fourcc"] = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4))

    # -- reading -------------------------------------------------------------
    @property
    def nominal_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else self.cfg.fps

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.cfg.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return False, None, time.perf_counter()
        if self.is_file:
            # Files: use the nominal frame clock so fs is exact.
            t = self._file_frame_idx / self.nominal_fps
            self._file_frame_idx += 1
        else:
            t = time.perf_counter()
        self._t_hist.append(t)
        return True, frame, t

    def measured_fps(self) -> float:
        """Frame rate actually delivered over the last ~2 s (0 if unknown)."""
        if len(self._t_hist) < 2:
            return 0.0
        span = self._t_hist[-1] - self._t_hist[0]
        return (len(self._t_hist) - 1) / span if span > 0 else 0.0

    def release(self) -> None:
        self.cap.release()
