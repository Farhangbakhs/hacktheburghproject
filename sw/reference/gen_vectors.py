"""Golden test vectors for the HLS kernel (docs/PLAN.md section 3), from evm_fixed.py.

    .venv\\Scripts\\python gen_vectors.py                        # real face clip -> test_vectors/v1_contract/
    .venv\\Scripts\\python gen_vectors.py --clip recordings/me.npz --n 300
    .venv\\Scripts\\python gen_vectors.py verify --got path/to/hls_output.bin
    .venv\\Scripts\\python gen_vectors.py verify --dir test_vectors/edge --got out_edge.bin

Writes (main set, test_vectors/v1_contract/):
  input_frames.bin     N frames x H x W words, word = B, G, R, 0 (uint8)
  expected_output.bin  N frames x H x W words, word = B, G, R, 0 (int8)
  params.json          w, h, r_hi, r_lo, alpha, N (+ init_frames, fs, cutoffs, sha256, layout)
  expected_state.bin   L_hi then L_lo (int32 LE, HxWx3 B,G,R interleaved) after each of the
                       first `state_frames` frames - for stage-by-stage debugging
and an edge-case set in test_vectors/edge/ (same files) with extreme registers
(alpha = 0xFFFF, r_hi = 0xFFFF, r_lo = 1), full-swing 0/255 flicker, clipping in
both signs and an INIT mid-stream (init_frames in params.json).

Frame order is frame-major, then row-major pixels; all multi-byte values little-endian.
A frame with index in init_frames is sent with init = 1 (state re-initialised; its output is 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from evm_common import FC_HI_DEFAULT, FC_LO_DEFAULT, FS_DEFAULT, SMALL_H, SMALL_W, FilterParams
from evm_fixed import FixedEVM, pack_input, pack_output, unpack_input, unpack_output

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent.parent / "test_vectors" / "v1_contract"
DEFAULT_CLIP = HERE / "recordings" / "brightroom.npz"


# -- sources ------------------------------------------------------------------------------
def to_small(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Gaussian-pyramid downscale to exactly w x h (as the laptop does before sending)."""
    f = frame
    while f.shape[1] // 2 >= w and f.shape[0] // 2 >= h:
        f = cv2.pyrDown(f)
    if f.shape[:2] != (h, w):
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    return f


def load_clip_frames(path: Path, n: int, start: int, w: int, h: int) -> np.ndarray:
    if path.suffix == ".npz":
        frames = np.load(path)["frames"]
    else:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while len(frames) < start + n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        frames = np.stack(frames)
    if len(frames) < start + n:
        raise SystemExit(f"{path} has {len(frames)} frames, need start+n = {start + n}")
    return np.stack([to_small(f, w, h) for f in frames[start:start + n]])


def edge_frames(n: int, w: int, h: int, seed: int = 7) -> np.ndarray:
    """Synthetic stress frames: random, full-swing flicker, ramps, constants, steps."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for i in range(n):
        f = np.empty((h, w, 3), np.uint8)
        q = (xx * 4 // w) + 4 * (yy * 2 // h)  # 8 regions
        flick = 255 if i % 2 else 0
        regions = [
            rng.integers(0, 256, (h, w)),                       # 0 random noise
            np.full((h, w), flick),                             # 1 0/255 flicker (max band)
            np.full((h, w), 255 - flick),                       # 2 opposite phase
            (xx * 255 // max(1, w - 1) + i * 7) % 256,          # 3 moving ramp
            np.full((h, w), 0),                                 # 4 constant 0
            np.full((h, w), 255),                               # 5 constant 255
            np.full((h, w), 255 if (i // 8) % 2 else 0),        # 6 slow square wave
            np.full((h, w), 128 + int(round(100 * np.sin(2 * np.pi * 1.2 * i / 30)))),  # 7 sine
        ]
        for ch in range(3):
            plane = np.zeros((h, w), np.int64)
            for k, r in enumerate(regions):
                plane[q == k] = (r[q == k] + 40 * ch * (k == 0)) % 256
            f[..., ch] = plane
        out.append(f)
    return np.stack(out)


# -- generation ------------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_model(frames: np.ndarray, regs: dict, init_frames: list[int], state_frames: int):
    evm = FixedEVM()
    evm.set_registers(regs["r_hi"], regs["r_lo"], regs["alpha"])
    outs, states = [], []
    for i, f in enumerate(frames):
        outs.append(evm.process_int8(f, init=(i in init_frames)))
        if i < state_frames:
            states.append(evm.state())
    return np.stack(outs), states


def write_set(out_dir: Path, frames: np.ndarray, regs: dict, init_frames: list[int], meta: dict,
              state_frames: int = 4) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    outputs, states = run_model(frames, regs, init_frames, state_frames)
    files = {
        "input_frames.bin": b"".join(pack_input(f) for f in frames),
        "expected_output.bin": b"".join(pack_output(o) for o in outputs),
        "expected_state.bin": b"".join(lh.astype("<i4").tobytes() + ll.astype("<i4").tobytes()
                                       for lh, ll in states),
    }
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
    params = {
        "w": w, "h": h, "r_hi": regs["r_hi"], "r_lo": regs["r_lo"], "alpha": regs["alpha"], "N": n,
        "init_frames": init_frames,
        "state_frames": len(states),
        **meta,
        "layout": {
            "input_frames.bin": "N*h*w little-endian uint32 words; byte0=B byte1=G byte2=R byte3=0 (uint8)",
            "expected_output.bin": "N*h*w words; byte0=B byte1=G byte2=R (int8 two's complement), byte3=0",
            "expected_state.bin": "state_frames x [L_hi, L_lo], each h*w*3 int32 LE (Q16.16), "
                                  "pixel-interleaved B,G,R, row-major; state AFTER that frame",
            "order": "frame-major, then row-major pixels",
            "init": "frames listed in init_frames are sent with init=1; their output is all zeros",
        },
        "contract": "docs/PLAN.md section 3 (bit-exact)",
        "stats": {
            "nonzero_output_fraction": float((outputs != 0).mean()),
            "clipped_fraction": float((np.abs(outputs.astype(int)) == 127).mean()),
            "output_min": int(outputs.min()), "output_max": int(outputs.max()),
        },
        "sha256": {name: _sha(out_dir / name) for name in files},
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))
    return params


def generate(args) -> int:
    out = Path(args.out)
    p = FilterParams(args.fc_lo, args.fc_hi, args.alpha, args.fs)
    p.validate()
    evm = FixedEVM(p)
    regs = evm.registers

    clip = Path(args.clip)
    if clip.exists():
        frames = load_clip_frames(clip, args.n, args.start, args.w, args.h)
        source = f"{clip.name} frames {args.start}..{args.start + args.n - 1}, pyrDown to {args.w}x{args.h}"
    else:
        from evm_eval import synthetic_clip
        print(f"[gen_vectors] {clip} not found - using a synthetic face clip (record a real one!)")
        fr, _ = synthetic_clip(n=args.n, h=args.h * 4, w=args.w * 4)
        frames = np.stack([to_small(f, args.w, args.h) for f in fr])
        source = "synthetic_clip (evm_eval.py)"

    meta = {"fs": p.fs, "fc_hi": p.fc_hi, "fc_lo": p.fc_lo, "alpha_float": p.alpha, "source": source}
    main = write_set(out, frames, regs, [0], meta, args.state_frames)
    print(f"[gen_vectors] main set -> {out}: N={main['N']} {main['w']}x{main['h']} regs "
          f"r_hi={regs['r_hi']} r_lo={regs['r_lo']} alpha={regs['alpha']}; "
          f"nonzero {main['stats']['nonzero_output_fraction']:.1%}, clipped {main['stats']['clipped_fraction']:.2%}, "
          f"range [{main['stats']['output_min']}, {main['stats']['output_max']}]")

    if not args.no_edge:
        ne = 64
        e_regs = {"r_hi": 0xFFFF, "r_lo": 1, "alpha": 0xFFFF}
        e_meta = {"fs": p.fs, "source": "edge_frames (synthetic stress patterns)",
                  "note": "extreme registers; not physically meaningful cutoffs"}
        edge = write_set(out / "edge", edge_frames(ne, args.w, args.h), e_regs, [0, ne // 2], e_meta,
                         args.state_frames)
        print(f"[gen_vectors] edge set -> {out / 'edge'}: N={ne}, init_frames {edge['init_frames']}, "
              f"clipped {edge['stats']['clipped_fraction']:.1%}, range "
              f"[{edge['stats']['output_min']}, {edge['stats']['output_max']}]")
    return 0


def verify(args) -> int:
    d = Path(args.dir)
    params = json.loads((d / "params.json").read_text())
    n, h, w = params["N"], params["h"], params["w"]
    exp = (d / "expected_output.bin").read_bytes()
    got = Path(args.got).read_bytes()
    fsz = h * w * 4
    if len(got) != len(exp):
        print(f"[verify] size mismatch: got {len(got)} bytes, expected {len(exp)} ({n} frames x {fsz})")
        if len(got) % fsz == 0:
            print(f"[verify] got holds {len(got) // fsz} whole frames; comparing those")
        n = min(n, len(got) // fsz)
    bad_total = 0
    shown = 0
    for i in range(n):
        e = unpack_output(exp[i * fsz:(i + 1) * fsz], h, w)
        g = unpack_output(got[i * fsz:(i + 1) * fsz], h, w)
        pad = np.frombuffer(got[i * fsz:(i + 1) * fsz], np.uint8).reshape(h, w, 4)[..., 3]
        diff = np.argwhere(e != g)
        if pad.any():
            print(f"[verify] frame {i}: byte3 not zero in {int((pad != 0).sum())} words")
        if len(diff):
            bad_total += len(diff)
            for y, x, c in diff[: max(0, 10 - shown)]:
                print(f"[verify] frame {i} y={y} x={x} ch={'BGR'[c]}: expected {e[y, x, c]}, got {g[y, x, c]}")
                shown += 1
    total = n * h * w * 3
    if bad_total == 0 and len(got) == len(exp):
        print(f"[verify] PASS: {n} frames, {total} values bit-exact")
        return 0
    print(f"[verify] FAIL: {bad_total} of {total} values differ")
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify":
        ap = argparse.ArgumentParser(prog="gen_vectors.py verify")
        ap.add_argument("--dir", default=str(DEFAULT_OUT))
        ap.add_argument("--got", required=True, help="kernel output, same format as expected_output.bin")
        return verify(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=str(DEFAULT_CLIP))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--w", type=int, default=SMALL_W)
    ap.add_argument("--h", type=int, default=SMALL_H)
    ap.add_argument("--fs", type=float, default=FS_DEFAULT)
    ap.add_argument("--fc-lo", type=float, default=FC_LO_DEFAULT)
    ap.add_argument("--fc-hi", type=float, default=FC_HI_DEFAULT)
    ap.add_argument("--alpha", type=float, default=50.0)
    ap.add_argument("--state-frames", type=int, default=4)
    ap.add_argument("--no-edge", action="store_true")
    return generate(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
