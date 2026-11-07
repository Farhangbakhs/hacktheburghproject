"""Record raw webcam frames losslessly, for tuning BPM / magnification offline.

    python capture_raw.py out.avi --seconds 30 --bpm 88

Writes out.avi (FFV1, lossless) and out.json (per-frame timestamps, camera
settings, and the reference BPM you read off a watch/phone). Compressed video
(mp4) destroys the sub-1-level colour changes the pulse makes, so use this.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2

from main import Camera, parse_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .avi")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-size", type=parse_size, default=(640, 480))
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--exposure", type=float, default=-5)
    ap.add_argument("--wb", type=float, default=4500)
    ap.add_argument("--bpm", type=float, help="reference heart rate from a watch/phone")
    ap.add_argument("--countdown", type=float, default=3.0)
    args = ap.parse_args()

    cam = Camera(args.camera, *args.cam_size, lock=not args.no_lock,
                 exposure=args.exposure, wb_kelvin=args.wb)
    frame = cam.read()  # warm up and get the real size
    if frame is None:
        raise SystemExit("camera gave no frame")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (w, h))
    if not writer.isOpened():
        raise SystemExit("FFV1 writer unavailable in this OpenCV build")

    print("sit still... recording in %.0f s" % args.countdown, flush=True)
    t_end = time.perf_counter() + args.countdown
    while time.perf_counter() < t_end:
        cam.read()  # keep draining so exposure settles and the buffer stays fresh
    print("recording %.0f s" % args.seconds, flush=True)

    stamps = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        frame = cam.read()
        if frame is None:
            break
        stamps.append(time.perf_counter() - t0)
        writer.write(frame)
    writer.release()
    cam.release()

    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0
    meta = {"frames": len(stamps), "fps": fps, "size": [w, h], "timestamps": stamps,
            "exposure_locked": not args.no_lock, "reference_bpm": args.bpm}
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(meta, f)
    print("saved %s: %d frames, %.1f fps" % (args.out, len(stamps), fps))


if __name__ == "__main__":
    main()
