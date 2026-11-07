"""Record a test clip yourself, then see the MIT-style magnification of it.

    .venv\\Scripts\\python test_clip.py                 # 15 s clip, auto name
    .venv\\Scripts\\python test_clip.py --name kitchen --seconds 20 --exposure -5

1. A preview window opens: check "face found" and that skin brightness is
   green (>= 80, ideally 100-180). Move to better light / adjust until it is.
   Exposure is metered on your FACE first (brightest setting that keeps
   30 fps without clipping skin); webcam auto-exposure meters the room.
2. Press SPACE to start recording (3 s countdown, then stay still, look at the
   camera, breathe normally). ESC/q aborts.
3. When done, the clip is saved to recordings/<name>.npz and the offline MIT
   replica (evm_mit.py, auto band around your measured pulse) plays back
   original | magnified in a loop. Pulse BPM + SNR are printed (SNR > ~3 dB =
   clean). The mp4 is saved next to the clip. Press q to close.
4. Optional: --live afterwards opens the live dashboard on the 'locked' preset.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from camera import CameraConfig, FrameSource
from face_tracker import FaceTracker
from face_tracker import draw as draw_face

HERE = Path(__file__).parent
WIN = "Test clip - SPACE: record   q: quit"


def _text(img, s, org, col=(240, 240, 240), scale=0.7):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def skin_stats(frame, tracker) -> tuple[float | None, float]:
    """(mean skin luma, fraction of skin pixels with any channel >= 250)."""
    res = tracker.update(frame)
    m = res.skin_mask if res.found else None
    if m is None or m.sum() < 1:
        return None, 0.0
    skin = m > 0.5
    sat = float((frame[skin].max(axis=1) >= 250).mean()) if skin.any() else 0.0
    return skin_luma(frame, res), sat


def skin_luma(frame, res) -> float | None:
    m = res.skin_mask if res.found else None
    if m is None or m.sum() < 1:
        return None
    return float((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * m).sum() / m.sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=time.strftime("clip_%Y%m%d_%H%M%S"))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--source", default="0")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--exposure", type=float, default=None, help="manual exposure (dshow: -5 = 31 ms)")
    ap.add_argument("--no-lock", action="store_true", help="leave auto exposure/WB on (not advised)")
    ap.add_argument("--no-meter", action="store_true",
                    help="skip face-metered exposure (default: expose for the face, not the room)")
    ap.add_argument("--alpha", type=float, default=100.0, help="MIT replica gain")
    ap.add_argument("--level", type=int, default=4, help="MIT replica pyramid level")
    ap.add_argument("--live", action="store_true", help="open the live 'locked' dashboard afterwards")
    args = ap.parse_args(argv)

    cfg = CameraConfig(source=args.source, exposure=args.exposure,
                       lock_exposure=not args.no_lock, lock_wb=not args.no_lock)
    if args.backend:
        cfg.backend = args.backend
    src = FrameSource(cfg)
    print(src.report)
    tracker = FaceTracker()
    if not args.no_meter and args.exposure is None and not args.no_lock and not src.is_file:
        print("[test_clip] metering exposure on your face (look at the camera)...")
        res = src.meter_on_face(lambda f: skin_stats(f, tracker))
        print(f"[test_clip] exposure {res['exposure']} -> skin luma {res['luma']}, {res['fps']} fps")
    n_target = int(round(args.seconds * 30))
    frames: list[np.ndarray] = []
    stamps: list[float] = []
    state, t_count = "preview", 0.0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame, t = src.read()
            if not ok:
                print("camera read failed")
                return 1
            if state == "recording":
                frames.append(frame.copy())
                stamps.append(t)
            res = tracker.update(frame, t)
            view = frame.copy()
            draw_face(view, res, show_mask=state == "preview")
            view = cv2.flip(view, 1)
            lum = skin_luma(frame, res)
            fps = src.measured_fps()
            good = lum is not None and lum >= 80
            _text(view, "face found" if res.found else "NO FACE", (10, 30), (90, 220, 90) if res.found else (60, 60, 240))
            _text(view, f"skin brightness {lum:.0f}" if lum is not None else "skin brightness -",
                  (10, 60), (90, 220, 90) if good else (60, 60, 240))
            _text(view, f"{fps:4.1f} fps", (10, 90), (90, 220, 90) if abs(fps - 30) < 3 else (60, 60, 240))
            if state == "preview":
                _text(view, "SPACE = start recording" + ("" if good else "  (more light!)"), (10, 465))
            elif state == "countdown":
                left = 3 - (time.perf_counter() - t_count)
                _text(view, f"recording in {left:.0f}... stay still", (140, 250), (0, 190, 255), 1.0)
                if left <= 0:
                    state = "recording"
            else:
                done = len(frames) / n_target
                cv2.rectangle(view, (0, 470), (int(640 * done), 480), (60, 60, 240), -1)
                _text(view, f"REC {len(frames) / 30:4.1f}/{args.seconds:.0f}s - stay still", (10, 455), (60, 60, 240))
                if len(frames) >= n_target:
                    break
            cv2.imshow(WIN, view)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) == 0:
                print("aborted")
                return 1
            if k == ord(" ") and state == "preview":
                state, t_count = "countdown", time.perf_counter()
    finally:
        src.release()
        cv2.destroyAllWindows()

    out_dir = HERE / "recordings"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{args.name}.npz"
    ts = np.array(stamps) - stamps[0]
    fs_meas = (len(ts) - 1) / ts[-1] if ts[-1] > 0 else 30.0
    np.savez_compressed(path, frames=np.stack(frames), fs=np.float64(30.0), timestamps=ts)
    print(f"[test_clip] saved {len(frames)} frames -> {path} (measured {fs_meas:.2f} fps)")
    if abs(fs_meas - 30) > 1.5:
        print("[test_clip] WARNING: camera was not at 30 fps; add light or set --exposure")

    from evm_mit import main as mit_main

    rc = mit_main([str(path), "--auto-band", "--alpha", str(args.alpha), "--level", str(args.level)])
    if args.live:
        from evm_locked import main as live_main

        live_main([] if args.exposure is None else ["--exposure", str(args.exposure)])
    return rc


if __name__ == "__main__":
    sys.exit(main())
