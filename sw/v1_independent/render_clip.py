"""Offline v1 renderer: prerecorded clip -> side-by-side (original | magnified) video.
    .venv/bin/python render_clip.py CLIP.mp4 --out out/clip_v1.mp4 [--alpha 50] [--impl fixed|float]
    .venv/bin/python render_clip.py ../../test_vectors/v1_contract --out out/v1.mp4      # 160x120 recorded frames
    ... --png 150  also saves frame 150 of the comparison as a PNG
Pipeline (PLAN.md section 2): downscale (pyrDown) -> filter -> upscale (pyrUp) -> add to full-res frame.
"""
import argparse, math
from pathlib import Path
import cv2
import numpy as np
from evm_v1 import FixedV1, FloatV1, coeff_q016, alpha_q88, words_to_frames, FC_HI, FC_LO


def read_small_frames(source, nmax):
    """Return (small BGR frames N,H,W,3 uint8 at ~160 wide, full-res frames list or None)."""
    src = Path(source)
    if src.is_dir():  # v1 vector dir
        import json
        p = json.loads((src / "params.json").read_text())
        w = np.fromfile(src / "input_frames.bin", "<u4")[: min(nmax, p["N"]) * p["h"] * p["w"]]
        return words_to_frames(w.reshape(-1, p["h"], p["w"])), None
    cap = cv2.VideoCapture(str(src))
    full = []
    while len(full) < nmax:
        ok, f = cap.read()
        if not ok:
            break
        full.append(f)
    if not full:
        raise SystemExit(f"could not read frames from {src}")
    levels = max(0, round(math.log2(full[0].shape[1] / 160)))
    small = []
    for f in full:
        s = f
        for _ in range(levels):
            s = cv2.pyrDown(s)
        small.append(s)
    return np.stack(small), full


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--impl", choices=["fixed", "float"], default="fixed")
    ap.add_argument("--frames", type=int, default=300); ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--display-width", type=int, default=None, help="resize full-res frames to this width for output")
    ap.add_argument("--png", type=int, default=None); ap.add_argument("--scale", type=float, default=3.0,
                    help="display scale for 160x120 vector input")
    a = ap.parse_args()

    small, full = read_small_frames(a.source, a.frames)
    model = (FixedV1(coeff_q016(FC_HI), coeff_q016(FC_LO), alpha_q88(a.alpha)) if a.impl == "fixed"
             else FloatV1(alpha=a.alpha))
    writer = None
    for i in range(len(small)):
        sig = model.process(small[i], init=(i == 0)).astype(np.float32)
        if full is not None:
            base = full[i]
            if a.display_width:
                base = cv2.resize(base, (a.display_width, round(base.shape[0] * a.display_width / base.shape[1])), interpolation=cv2.INTER_AREA)
            up = cv2.resize(sig, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            base = cv2.resize(small[i], None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
            up = cv2.resize(sig, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
        mag = cv2.add(base, up, dtype=cv2.CV_8U)
        frame = np.hstack([label(base.copy(), "original"), label(mag, f"v1 {a.impl} alpha={a.alpha:g}")])
        if writer is None:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if a.png == i:
            cv2.imwrite(str(Path(a.out).with_suffix(f".f{i}.png")), frame)
    writer.release()
    print(f"wrote {a.out} ({len(small)} frames)")


if __name__ == "__main__":
    main()
