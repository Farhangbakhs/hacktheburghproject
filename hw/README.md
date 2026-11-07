# hw/ (Jack)

## Step 1: board bring-up (manual)
1. Flash the PYNQ-Z2 image (pynq.io) to SD, jumper JP4 to SD, power on.
2. Ethernet cable laptop <-> board. Set laptop to static 192.168.2.1/24; board is 192.168.2.99 by default.
3. Open http://192.168.2.99 (Jupyter, password `xilinx`). Open a terminal there.
4. Copy `hw/bringup/board_info.py` over (Jupyter upload or `scp`) and run `python3 board_info.py`.
   Paste the output to Claude. It tells us the PYNQ version and therefore which Vivado version to use
   (PYNQ v3.0.x -> Vivado 2022.1, v2.7 -> 2020.2).

## Step 2: DMA loopback
1. Install the PYNQ-Z2 board files in Vivado (needed by `board_part`).
2. `cd hw/vivado && vivado -mode batch -source loopback.tcl` (about 10-20 min).
   Produces `hw/overlay/loopback.bit` and `loopback.hwh`.
3. Copy both plus `hw/bringup/loopback_test.py` to the board (same folder, same base name).
4. `sudo python3 loopback_test.py`. Expect `LOOPBACK PASS`.
Time-box: debug until +5:00, then fall back to the software path.

`loopback.tcl` and the test are untested; paste any Vivado/Python errors back.

## Vivado in Docker (Apple Silicon)
`hw/vivado-docker/` is a copy of vivado-on-silicon-mac (Vivado + Vitis HLS 2021.1 already installed in
`hw/vivado-docker/Xilinx/`, git-ignored, ~48 GB). Only its scripts are tracked. It needs the `x64-linux`
Docker image, so Docker Desktop must be running; if the image is missing, run `scripts/gen_image.sh`.
- Start: `hw/vivado-docker/scripts/start_container.sh` (opens a VNC desktop).
- The whole repo is mounted in the container at `/home/user/repo` (edited start_container.sh).
- Board files: unzip the TUL PYNQ-Z2 files so `pynq-z2/` is in `hw/vivado-docker/board_files/` (git-ignored).
- Easiest: in the container terminal run `bash ~/repo/hw/vivado/run_loopback.sh` (or `... resume` after a
  crash). It sets the preload, runs Vivado and writes `hw/vivado/build.log`. Manual equivalent:
  ```
  # IMPORTANT: same preload as scripts/de_start.sh, otherwise Vivado aborts (libudev realloc crash,
  # hs_err_pid*.log) as soon as a run finishes. Never run `vivado` without it.
  export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libselinux.so.1 /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libgdk-x11-2.0.so.0"
  source /home/user/Xilinx/Vivado/2021.1/settings64.sh
  cd /home/user/repo/hw/vivado
  BOARD_REPO=/home/user/board_files vivado -mode batch -source loopback.tcl 2>&1 | tee build.log
  ```
  If it crashed after synthesis, `vivado -mode batch -source loopback_resume.tcl` continues from the
  existing project. If you move/copy the docker folder, use `rsync -a` so hidden files (.config) come along.
  Outputs land in `hw/overlay/` of the repo directly.
