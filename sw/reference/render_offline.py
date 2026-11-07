"""Render a recorded clip through a dashboard preset (offline, same code path as live).

    .venv\Scripts\python render_offline.py recordings/brightroom.npz --preset locked --alpha 100
Writes recordings/<clip>_<preset>.mp4 (original | magnified) and prints the lock timeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from dashboard import load_presets
from evm_common import FilterParams, add_signal, downscale, pyramid_levels_for, upscale
from evm_float import FloatEVM
from face_tracker import FaceTracker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--preset", default="locked")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = np.load(a.clip)
    frames, fs = d["frames"], float(d["fs"]) if "fs" in d else 30.0
    presets = load_presets(FloatEVM)
    evm = presets[a.preset](FilterParams(alpha=a.alpha, fs=fs))
    tr = FaceTracker()
    lv = pyramid_levels_for(frames.shape[2])
    out = a.out or str(Path(a.clip).with_suffix("")) + f"_{a.preset}.mp4"
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fs, (2 * w, h))
    for i, f in enumerate(frames):
        r = tr.update(f, i / fs)
        small = downscale(f, lv)
        mfull = r.skin_mask if (r.found and not a.no_mask) else None
        if hasattr(evm, "set_mask"):
            evm.set_mask(r.mask_resized(small.shape[:2]) if mfull is not None else None)
        sig = (evm.process(f).astype(np.float32) if getattr(evm, "full_res", False)
               else upscale(evm.process(small).astype(np.float32), lv, (h, w)))
        if mfull is not None:
            sig *= mfull[..., None]
        vw.write(np.hstack([f, add_signal(f, sig)]))
        if i % 30 == 29:
            print(f"t={(i + 1) / fs:4.1f}s  {getattr(evm, 'name', a.preset)}")
    vw.release()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
