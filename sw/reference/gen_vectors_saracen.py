"""Golden vectors for the PROPOSED section 3 v2 (saracen Laplacian EVM) kernel.

    .venv\\Scripts\\python gen_vectors_saracen.py                 # face clip -> test_vectors/v2_saracen/
    .venv\\Scripts\\python gen_vectors_saracen.py verify --got hls_out.bin
    .venv\\Scripts\\python gen_vectors_saracen.py verify --dir test_vectors/v2_saracen/edge --got out.bin

Files per set (all little-endian, frame-major):
  input_words.bin      N x total_words 32-bit words; low 16 bits = int16 Q8.8 Laplacian band
                       value (L channel), high 16 bits = 0. Order: level 0 (= Laplacian L1,
                       160x120) row-major, then level 1 (80x60), then level 2 (40x30)
  expected_output.bin  N x total_words words, same layout, int16 Q8.8 filtered+amplified
  params.json          registers (r_hi, r_lo, alpha_i, count_i), N, level shapes, init_frames, ...
  expected_state.bin   state_frames x [L_hi, L_lo], each total_words int32 (Q16.16) - state
                       AFTER that frame, for stage-by-stage debugging
Model: evm_saracen_fixed.SaracenKernel (bit-exact spec in its module docstring).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from evm_common import FilterParams
from evm_saracen_fixed import (MAX_HW_LEVELS, Registers, SaracenHW, SaracenKernel, laptop_pre, pack_words,
                               registers_for, unpack_words)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent.parent / "test_vectors" / "v2_saracen"
DEFAULT_CLIP = HERE / "recordings" / "brightroom.npz"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_set(out: Path, words: np.ndarray, regs: Registers, init_frames: list[int], meta: dict,
              state_frames: int = 4) -> dict:
    """words: int16 [N, total]. Runs the kernel model and writes the vector files."""
    out.mkdir(parents=True, exist_ok=True)
    k = SaracenKernel(regs)
    outs, states = [], []
    for i, w in enumerate(words):
        outs.append(k.process(w, init=i in init_frames))
        if i < state_frames:
            states.append(k.state())
    outs = np.stack(outs)
    files = {
        "input_words.bin": b"".join(pack_words(w) for w in words),
        "expected_output.bin": b"".join(pack_words(o) for o in outs),
        "expected_state.bin": b"".join(a.astype("<i4").tobytes() + b.astype("<i4").tobytes() for a, b in states),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)
    params = {
        "contract": "PROPOSED section 3 v2 (saracen Laplacian EVM) - see sw/reference/HANDOFF_JACK.md",
        **regs.as_dict(),
        "N": int(len(words)),
        "level_shapes_hw": [list(s) for s in regs.shapes] + [[0, 0]] * (MAX_HW_LEVELS - len(regs.shapes)),
        "init_frames": init_frames,
        "state_frames": len(states),
        **meta,
        "formats": {
            "input": "int16 Q8.8 (value*256) in low 16 bits of 32-bit LE word, high 16 bits 0",
            "output": "int16 Q8.8 in low 16 bits, high 16 bits 0; clipped to [-32767, 32767]",
            "state": "int32 Q16.16, L_hi block then L_lo block per frame",
            "r_hi/r_lo": "uint16 Q0.16", "alpha_i": "uint16 Q8.8, gain for level i",
        },
        "stats": {
            "input_min": int(words.min()), "input_max": int(words.max()),
            "output_min": int(outs.min()), "output_max": int(outs.max()),
            "nonzero_output_fraction": float((outs != 0).mean()),
            "clipped_fraction": float((np.abs(outs.astype(np.int32)) == 32767).mean()),
        },
        "sha256": {n: _sha(out / n) for n in files},
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))
    return params


def generate(a) -> int:
    from gen_vectors import load_clip_frames  # same clip loader as v1

    p = FilterParams(a.fc_lo, a.fc_hi, a.alpha, a.fs)
    p.validate()
    cfg = SaracenHW(levels=a.levels, lambda_c=a.lambda_c)
    clip = Path(a.clip)
    if not clip.exists():
        raise SystemExit(f"{clip} not found - record one with test_clip.py")
    frames = load_clip_frames(clip, a.n, a.start, 640, 480)  # full res; laptop_pre does the rest
    pre = [laptop_pre(f, cfg) for f in frames]
    words = np.stack([w for _, _, w in pre])
    work_hw = pre[0][0].shape[:2]
    regs = registers_for(p, cfg, work_hw)
    meta = {"fs": p.fs, "fc_hi": p.fc_hi, "fc_lo": p.fc_lo, "alpha_float": p.alpha,
            "lambda_c": cfg.lambda_c, "exaggeration": cfg.exaggeration, "levels": cfg.levels,
            "work_size_hw": list(work_hw),
            "source": f"{clip.name} frames {a.start}..{a.start + a.n - 1} (640x480 -> 320x240 Lab L-channel Laplacian)"}
    main = write_set(Path(a.out), words, regs, [0], meta, a.state_frames)
    print(f"[v2 vectors] main -> {a.out}: N={main['N']} total_words={main['total_words']} "
          f"regs r_hi={regs.r_hi} r_lo={regs.r_lo} alpha={regs.alpha} count={regs.count}; "
          f"in [{main['stats']['input_min']}, {main['stats']['input_max']}] "
          f"out [{main['stats']['output_min']}, {main['stats']['output_max']}] "
          f"clipped {main['stats']['clipped_fraction']:.3%}")

    if not a.no_edge:
        ne = 64
        rng = np.random.default_rng(11)
        e_regs = Registers(0xFFFF, 1, [0xFFFF, 0x8000, 0x0100, 0], list(regs.count), list(regs.shapes))
        ew = []
        for i in range(ne):
            v = np.empty(regs.total, np.int16)
            q = regs.total // 4
            v[:q] = rng.integers(-32767, 32768, q)                 # random full range
            v[q:2 * q] = 32767 if i % 2 else -32767                # full-swing flicker (overflow / clip)
            v[2 * q:3 * q] = 0                                     # zero
            v[3 * q:] = np.int16(int(20000 * np.sin(2 * np.pi * 1.2 * i / 30)))  # sine
            ew.append(v)
        edge = write_set(Path(a.out) / "edge", np.stack(ew), e_regs, [0, ne // 2],
                         {"source": "synthetic stress words", "note": "extreme registers, not physical"},
                         a.state_frames)
        print(f"[v2 vectors] edge -> {Path(a.out) / 'edge'}: init_frames {edge['init_frames']}, "
              f"out [{edge['stats']['output_min']}, {edge['stats']['output_max']}], "
              f"clipped {edge['stats']['clipped_fraction']:.1%}")
    return 0


def verify(a) -> int:
    d = Path(a.dir)
    params = json.loads((d / "params.json").read_text())
    n, tot = params["N"], params["total_words"]
    counts = [params[f"count_{i}"] for i in range(MAX_HW_LEVELS)]
    exp = unpack_words((d / "expected_output.bin").read_bytes()).reshape(n, tot)
    raw = Path(a.got).read_bytes()
    if len(raw) % (tot * 4):
        print(f"[verify] got {len(raw)} bytes, not a whole number of {tot * 4}-byte frames")
        return 1
    try:
        got = unpack_words(raw).reshape(-1, tot)
    except ValueError as e:
        print(f"[verify] {e}")
        return 1
    m = min(n, len(got))
    bad = np.argwhere(exp[:m] != got[:m])
    edges = np.cumsum([0] + counts)
    for f, i in bad[:10]:
        lvl = int(np.searchsorted(edges, i, side="right") - 1)
        print(f"[verify] frame {f} word {i} (level {lvl}, index {i - edges[lvl]}): "
              f"expected {exp[f, i]}, got {got[f, i]}")
    if len(bad) == 0 and len(got) == n:
        print(f"[verify] PASS: {n} frames x {tot} words bit-exact")
        return 0
    print(f"[verify] FAIL: {len(bad)} mismatches over {m} frames" + (f"; got {len(got)} of {n} frames" if len(got) != n else ""))
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "verify":
        ap = argparse.ArgumentParser(prog="gen_vectors_saracen.py verify")
        ap.add_argument("--dir", default=str(DEFAULT_OUT))
        ap.add_argument("--got", required=True)
        return verify(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=str(DEFAULT_CLIP))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--fc-lo", type=float, default=0.25)
    ap.add_argument("--fc-hi", type=float, default=2.44)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--lambda-c", type=float, default=40.0)
    ap.add_argument("--state-frames", type=int, default=4)
    ap.add_argument("--no-edge", action="store_true")
    return generate(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
