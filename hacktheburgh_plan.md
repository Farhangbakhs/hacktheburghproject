# Pulse magnification (Eulerian video magnification) build plan

HackTheBurgh 2026. Deadline Sunday 12:00. Submit by 11:00.
Team: Jack (hardware, PYNQ-Z2 + Vivado), Person 2 (reference algorithm), Person 3 (app + networking + pitch).

Give this whole file to your coding agent, then point it at your workstream section. The interface contract (section 3) is the single source of truth. Change it only by agreement, and update this file when you do.

---

## 1. Goal

Live webcam video where a person's face visibly pulses with their heartbeat, plus a BPM readout. The per-pixel temporal filtering and amplification runs on the PYNQ-Z2 FPGA, with a software version as baseline and fallback.

Track: G-Research ("make the invisible visible").

## 2. Architecture

```
Laptop webcam (full res)
  -> downscale to 160x120 (Gaussian pyramid)
  -> TCP to PYNQ ARM (Python server)
  -> AXI DMA -> FPGA kernel (temporal bandpass + amplify) -> AXI DMA
  -> TCP back to laptop
  -> upscale amplified signal, add to full-res frame
  -> display: original | magnified | BPM
```

The same app can swap the PYNQ for a local Python processor (fallback and benchmark baseline).

## 3. Interface contract (agree in first 30 min)

### Frame format

- Resolution: W=160, H=120 (configurable; FPGA max set at build time).
- Frame rate: 30 fps, fixed. Lock the webcam to it.
- Input pixels: uint8, BGR order (OpenCV native), row-major.
- Input DMA stream: 32-bit words, one pixel per word: byte0=B, byte1=G, byte2=R, byte3=0.
- Output DMA stream: 32-bit words, one pixel per word: int8 amplified signal per channel, same byte order, clipped to [-127, 127].

### Filter (bit-exact, fixed point)

Bandpass = difference of two first-order low-pass filters, per pixel, per channel.

- Coefficients: `r = 1 - exp(-2*pi*fc/fs)`, stored as unsigned Q0.16 (`round(r * 65536)`).
  - `r_hi` for the high cutoff (default 2.0 Hz), `r_lo` for the low cutoff (default 0.83 Hz).
- State: L_hi, L_lo as int32 Q16.16, one pair per pixel per channel.
- Init (first frame, or when INIT flag set): `L_hi = L_lo = p << 16`.
- Update per frame (64-bit intermediates, arithmetic shift):

  ```
  x     = p << 16
  L_hi += (r_hi * (x - L_hi)) >> 16
  L_lo += (r_lo * (x - L_lo)) >> 16
  band  = L_hi - L_lo                      # Q16.16
  out   = clip((alpha * band) >> 24, -127, 127)  # alpha is Q8.8
  ```

- Alpha: unsigned Q8.8 (`round(alpha * 256)`), default alpha = 50.

### Control registers (AXI-Lite, so changes need no rebuild)

| Name          | Type         | Meaning                             |
| ------------- | ------------ | ----------------------------------- |
| width, height | uint16       | Frame size                          |
| r_hi, r_lo    | uint16 Q0.16 | Filter coefficients                 |
| alpha         | uint16 Q8.8  | Gain                                |
| init          | 1 bit        | Re-initialise state from next frame |

Exact offsets come from the HLS-generated driver; Jack publishes them once the kernel builds.

### Network protocol (laptop <-> PYNQ)

- TCP, PYNQ listens on port 5005.
- Request: 16-byte header (little-endian `uint32 frame_id, uint16 w, uint16 h, uint16 r_hi, uint16 r_lo, uint16 alpha, uint8 init, uint8 pad`) + `w*h*4` bytes input.
- Response: `uint32 frame_id` + `w*h*4` bytes output.

### Golden test vectors

Person 2 produces, from the fixed-point reference:

- `test_vectors/input_frames.bin` (N frames, input format above)
- `test_vectors/expected_output.bin` (N frames, output format above)
- `test_vectors/params.json` (w, h, r_hi, r_lo, alpha, N)
  Use a real 10 s face clip, N=300. Jack's HLS testbench must match bit-for-bit.

## 4. Repo layout

```
docs/PLAN.md                  (this file)
sw/reference/evm_float.py     float reference + visual check
sw/reference/evm_fixed.py     bit-exact fixed-point model of section 3
sw/reference/gen_vectors.py   writes test_vectors/
sw/reference/bpm.py           BPM estimation
sw/app/main.py                capture, display, UI
sw/app/processors.py          LocalPython, MockServer, PynqRemote
sw/pynq_server/server.py      runs on PYNQ ARM
hw/hls/evm_kernel.cpp/.h      HLS kernel
hw/hls/tb.cpp                 C-sim testbench using test_vectors/
hw/vivado/build.tcl           block design + bitstream
hw/overlay/                   evm.bit + evm.hwh for PYNQ
test_vectors/
```

## 5. Workstreams

### Person 2: reference algorithm

1. `evm_float.py`: webcam -> Gaussian pyramid downscale (cv2.pyrDown x2-3 to ~160x120) -> float IIR bandpass (section 3 maths in float) -> amplify -> pyrUp to full res -> add -> display. **Go/no-go: visible pulse on a face within 2 hours.**
2. Tuning notes: steady lighting, still subject, disable auto-exposure/white balance if possible (`cv2.CAP_PROP_AUTO_EXPOSURE` etc., platform dependent). Record which settings worked.
3. `evm_fixed.py`: bit-exact version of section 3. Check it visually matches the float version.
4. `gen_vectors.py`: produce the golden vectors. Send to Jack early, even a quick first version.
5. `bpm.py`: face ROI (OpenCV face detector), mean green over ROI, FFT over a rolling 10-15 s window, peak in 0.8-2 Hz -> BPM. Validate against a phone/watch reading on 3 people; record errors.
6. Help Person 3 integrate into the app.

### Person 3: app, networking, pitch

1. `processors.py`: one interface, `process(small_frame_bgr) -> int8 signal`, with:
   - `LocalPython` (wraps `evm_fixed.py`; use a stub until it exists)
   - `MockServer` client talking to a local copy of the server
   - `PynqRemote` (same client, PYNQ IP address)
2. `server.py`: TCP server per section 3. Start with a CPU backend (fixed-point model in NumPy) so it runs on the laptop and on the PYNQ ARM before the FPGA is ready. Jack swaps in the DMA backend later.
3. `main.py`: side-by-side original vs magnified, BPM, FPS counter, sliders for alpha and cutoffs, backend selector (Local / PYNQ). Keyboard shortcut to reset state.
4. Benchmark mode: frames/sec and latency per backend at 160x120 and larger sizes.
5. Pitch (3 min), demo script, and a **recorded fallback video by Sunday 10:00**.

### Jack: hardware

No dependency on software for steps 1-2. Start immediately.

1. **Board bring-up:** PYNQ image on SD, board on network, Jupyter reachable. Note PYNQ version and its recommended Vivado version.
2. **DMA loopback (hardware go/no-go):** Zynq PS + AXI DMA + AXI-Stream FIFO loopback. Build, load as overlay from Python, send a NumPy array, get it back. **Target: within ~3 hours.**
3. **HLS kernel** (`evm_kernel.cpp`):
   - AXI-Stream in/out (32-bit), AXI-Lite control per section 3.
   - State arrays in BRAM: 160x120x3x2 int32 ≈ 460 KB. Zynq-7020 has ~630 KB BRAM, so check the utilisation report. If too big, process at a smaller size or use int16 state (agree the change with Person 2).
   - Pipeline II=1 over pixels; the state read-modify-write per pixel has no cross-pixel dependency.
   - C-sim against golden vectors (until they arrive, generate a quick test with a short Python snippet).
4. **Integrate:** replace FIFO with kernel, build bitstream, test on board with golden vectors via Jupyter.
5. **DMA backend for `server.py`:** allocate buffers once (`pynq.allocate`), reuse per frame.
6. **Benchmark:** kernel time vs CPU fixed-point on PYNQ ARM vs laptop, at several resolutions.
7. Stretch: larger frames (state in DDR), higher frame rate.

## 6. Checkpoints

| When      | Checkpoint                                  | If missed                                                |
| --------- | ------------------------------------------- | -------------------------------------------------------- |
| +0:30     | Section 3 agreed                            | Use the defaults written here                            |
| +2:00     | Float prototype shows a pulse               | Fix lighting/exposure; if still nothing, rethink         |
| +3:00     | DMA loopback works on board                 | Debug toolchain; time-box to +5:00                       |
| +6:00     | HLS kernel passes C-sim with golden vectors | Simplify: one channel, smaller frame                     |
| Sat late  | Kernel on board matches golden vectors      | Keep going overnight; software path is demo-ready        |
| Sun 09:00 | Feature freeze, end-to-end demo works       | Demo with LocalPython, show FPGA in C-sim/on-board tests |
| Sun 10:00 | Fallback video recorded                     | Non-negotiable                                           |
| Sun 11:00 | Submitted                                   |                                                          |

## 7. Demo plan

1. Hook: "Your heartbeat is visible on your face. You just can't see it."
2. Judge sits in front of the camera; magnified view pulses; BPM shown next to their watch/phone reading.
3. Slide alpha up and down live.
4. Backend switch: same output from FPGA and software; show speed comparison.
5. Close: how it works (4 steps), honest limits, uses (contactless monitoring, structural vibration).
   Fun demo, not a medical device. Ask consent before filming anyone.

## 8. Risks

| Risk                                                    | Mitigation                                            |
| ------------------------------------------------------- | ----------------------------------------------------- |
| No visible pulse (lighting, compression, auto-exposure) | Steady lamp, lock exposure, sit still, check early    |
| Vivado/DMA integration eats time                        | Loopback first, time-box, software fallback           |
| BRAM too small                                          | Smaller frame, int16 state, or one channel            |
| Fixed-point mismatch HLS vs Python                      | Bit-exact contract, golden vectors, compare per stage |
| "Why an FPGA?" from judges                              | Benchmark at higher resolutions; low-power edge story |
| Network flakiness at venue                              | Direct Ethernet cable laptop <-> PYNQ, static IPs     |

## 9. Working with coding agents

- Give the agent this file plus your section. Ask it to implement exactly section 3; any deviation must be flagged.
- Agents can't see the board, camera or screen. Paste real outputs, errors and reports back in.
- Ask for tests first: fixed-point model vs float, HLS C-sim vs golden vectors, server round-trip with the mock.
- Small commits, one shared repo. Don't edit someone else's directory without telling them.
