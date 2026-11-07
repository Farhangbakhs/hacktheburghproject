# Independent v1 implementation (Jack)

A second, independent implementation of the PLAN.md section 3 filter ("v1", colour EVM), written from the
contract text only, to cross-check `sw/reference/` and the golden vectors. Not owned by Person 2 or 3; it
lives outside `sw/reference/` on purpose. Nothing else depends on it.

```
cd sw/v1_independent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # 8 tests, incl. the golden vectors
.venv/bin/python verify_vectors.py           # bit-exact check on test_vectors/v1_contract (+ /edge)
```

| File | What |
|---|---|
| `evm_v1.py` | `FixedV1` (bit-exact int64 model) and `FloatV1`; coefficient helpers; DMA word packing |
| `verify_vectors.py` | Runs `FixedV1` on Person 2's v1 vectors: output and filter state, bit for bit |
| `gen_vectors.py` | Regenerates vectors from a clip or a vector dir (writes to `--out`, never into `test_vectors/` by default) |
| `render_clip.py` | Prerecorded clip -> original \| magnified side-by-side mp4 (`--alpha`, `--impl fixed|float`) |
| `analyze.py`, `analyze_pulse.py` | Is there a pulse? Per-ROI spectrum, control regions, drift-corrected face signal |

## Results so far (2026-11-07)
- `FixedV1` reproduces Person 2's v1 vectors exactly (300 frames + 64-frame stress set); regenerated
  `input_frames.bin`, `expected_output.bin`, `expected_state.bin` are byte-identical. Coefficients match
  (r_hi 22427, r_lo 10457, alpha 12800).
- Pixel 9a clips (exposure locked, 1080p30): a face-localised peak at 68-78 bpm, stable across the clip, but no
  ground-truth heart rate was recorded. Visually, v1 amplifies head/edge motion; judged too motion-heavy to demo
  as is. See `analyze_pulse.py` for the method and its caveats (hand-picked boxes, one subject, short clips).
- Clips and renders are not committed (`out/` is git-ignored; ~100+ MB each).
