# Person 2 build plan: reference algorithm (`sw/reference/`)

Follows docs/PLAN.md §5 "Person 2". Section 3 is binding; the only deviation
proposed is flagged below. One step at a time, each with tests, each checked
on the real webcam before moving on.

| Step | Deliverable | Tests | Hardware check (you) | Status |
|---|---|---|---|---|
| 1 | `evm_float.py` + `evm_common.py` + `camera.py` + `dashboard.py` (live original \| magnified, pulse trace, sliders, fps, exposure lock) | `tests/test_evm_float.py` (28): coeffs, analytic frequency response, init/reset, clip, pyramid, synthetic end-to-end pulse, headless dashboard | Face visibly pulses; trace shows ~1 Hz wave (**go/no-go**) | **built, awaiting webcam check** |
| 2 | `TUNING.md`: which camera/backend/exposure/lighting settings worked | n/a | Fill in from your runs | template ready |
| 3 | `evm_fixed.py`: bit-exact §3 model (int64 maths, arithmetic shifts, Q formats); dashboard `--impl fixed` | fixed vs float within ±2 LSB on synthetic + recorded clip; hand-computed vectors; overflow edge cases | Same visual result as float | next |
| 4 | `gen_vectors.py` + clip recorder (`c` key: save 10 s of 160x120 frames): writes `test_vectors/input_frames.bin`, `expected_output.bin`, `params.json` | Round-trip read/write; byte layout (B,G,R,0 / int8 x3,0); re-run determinism | Record a real 10 s face clip; send vectors to Jack | |
| 5 | `bpm.py`: face ROI + **POS** rPPG + Welch/FFT peak + SNR confidence; BPM overlay on dashboard | Synthetic pulse -> exact BPM; harmonic/half-rate rejection; SNR gating; resampling with jittery timestamps | Compare to phone/watch on 3 people; log errors | |
| 6 | Help Person 3 wrap `FloatEVM`/`FixedEVM` behind `process(small_bgr) -> int8` | Interface test | | |

## Deviation flagged (needs team agreement, not yet applied)

- **Step 5 method:** PLAN.md says "mean green over ROI". Research (rPPG-Toolbox,
  NeurIPS 2023) shows mean green MAE ~20 bpm vs **POS ~4 bpm** on UBFC-rPPG.
  Proposal: implement POS as default, keep mean-green as an option. This does not
  touch the §3 interface (BPM is laptop-side only).
- Face ROI: MediaPipe Face Landmarker (Tasks API; mediapipe 1.2.x has Win wheels),
  fallback OpenCV YuNet/Haar. Adds a dependency; laptop only.

## Design notes (step 1)

- Filter maths exactly §3 in float32; `process()` returns `alpha*band` clipped
  to ±127 (pixel units), matching the fixed-point output scale.
- `fs` fixed at 30 per contract. Dashboard flags red if the camera delivers a
  different rate (dim light often halves it); `f` key optionally retunes
  coefficients to measured fps for laptop experiments only.
- Pyramid levels auto-chosen to reach ~160 wide (640x480 -> 160x120, L2).
- Processor interface (`process`, `reset`, `set_params`, `band_state`, `name`)
  is what `evm_fixed.FixedEVM` will also implement, so the dashboard swaps them.

## Run

```
cd sw/reference
uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt
.venv\Scripts\python -m pytest -q tests
.venv\Scripts\python evm_float.py                # live webcam
.venv\Scripts\python evm_float.py --backend msmf # if dshow misbehaves
.venv\Scripts\python evm_float.py --source clip.mp4
```
