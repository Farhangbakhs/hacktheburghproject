# Hardware status (Jack), 2026-11-07 night

| Step | State |
|---|---|
| Board bring-up | Done. PYNQ 3.1.1 (Carlisle, kernel 6.6, Python 3.10, NumPy 1.21), board at 192.168.2.99 |
| DMA loopback | **PASS on board**: 16 / 1024 / 19200 words, ~1 ms. `hw/overlay/loopback.bit/.hwh` |
| HLS kernel | **Not started.** Waiting on the v1 vs v2 contract decision (see below) |

## Rebuild the loopback
Vivado 2021.1 (+ Vitis HLS 2021.1) in the vivado-on-silicon-mac docker image on Jack's Mac. Inside the container
terminal: `bash ~/repo/hw/vivado/run_loopback.sh` (or `... resume`). Needs the PYNQ-Z2 board files and the
`LD_PRELOAD` the script sets. Details in `hw/README.md`. The docker scripts themselves are not in git.
Test on board: copy `loopback.bit`, `loopback.hwh`, `hw/bringup/loopback_test.py` to `/home/xilinx/`, then
`python3 loopback_test.py` (Jupyter terminal at http://192.168.2.99, password `xilinx`).

## Decision needed from the team: v1 or v2 contract
- Person 2 proposes v2 (`sw/reference/HANDOFF_JACK.md`): saracen motion EVM, 25,200 int16 words/frame, 4 gains.
  Not yet reflected in `docs/PLAN.md`; needs sign-off, plus a protocol change from Person 3.
- Jack independently verified the v2 vectors against a scalar model of the handoff spec (300 + 64 frames: PASS),
  and the v1 vectors against `sw/v1_independent/` (PASS).
- On phone clips (Pixel 9a, locked exposure) v1 showed a face-localised ~70-78 bpm component but looked too
  motion-heavy to demo. Suggested: ask Person 2 to run v2 on the same clips and compare by eye, and re-record a
  still, breath-hold clip (head against a wall, phone fixed, steady light).
- Either way the FPGA filter is the same IIR; kernel + C-sim + bitstream is roughly 4-6 h of work once decided.

## Risks / notes
- FPGA share of the work is small in v2 (~25k values/frame at ~0.25 ms); a CPU does it as fast. The pitch should
  lean on bit-exact verification, power/edge story, and benchmarks at larger resolutions.
- `test_vectors/` is ~128 MB of binaries in git; avoid regenerating and recommitting them.
