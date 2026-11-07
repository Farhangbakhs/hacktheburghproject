"""Generate v1 golden vectors from any clip, independently of sw/reference/gen_vectors.py.
    .venv/bin/python gen_vectors.py --source clip.mp4 --out /path/to/outdir [--frames 300]
    .venv/bin/python gen_vectors.py --source ../../test_vectors/v1_contract --out out/regen   # re-derive from recorded frames
Same files/layout as test_vectors/v1_contract (never writes into test_vectors/ unless you point --out there).
"""
import argparse, json
from pathlib import Path
import numpy as np
from evm_v1 import FixedV1, coeff_q016, alpha_q88, frames_to_words, FC_HI, FC_LO, ALPHA, FS
from render_clip import read_small_frames

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="video file, or a v1 vector dir (reads input_frames.bin)")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=300)
a = ap.parse_args()

frames, _ = read_small_frames(a.source, a.frames)
n, h, w, _ = frames.shape
r_hi, r_lo, alpha = coeff_q016(FC_HI), coeff_q016(FC_LO), alpha_q88(ALPHA)
m = FixedV1(r_hi, r_lo, alpha)
outs, states = [], []
for i in range(n):
    outs.append(m.process(frames[i], init=(i == 0)))
    if i < 4:
        lh, ll = m.state_int32(); states.append(np.stack([lh, ll]))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
frames_to_words(frames).astype("<u4").tofile(out / "input_frames.bin")
frames_to_words(np.stack(outs)).astype("<u4").tofile(out / "expected_output.bin")
np.stack(states).astype("<i4").tofile(out / "expected_state.bin")
(out / "params.json").write_text(json.dumps(dict(
    w=w, h=h, r_hi=r_hi, r_lo=r_lo, alpha=alpha, N=n, init_frames=[0], state_frames=min(4, n),
    fs=FS, fc_hi=FC_HI, fc_lo=FC_LO, alpha_float=ALPHA, source=str(a.source),
    contract="docs/PLAN.md section 3 (bit-exact)"), indent=2))
print(f"wrote {n} frames of {w}x{h} to {out}")
