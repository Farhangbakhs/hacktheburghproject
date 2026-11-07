#!/bin/bash
# Run INSIDE the vivado docker container (no copy/paste needed):
#   bash ~/repo/hw/vivado/run_loopback.sh          # full build
#   bash ~/repo/hw/vivado/run_loopback.sh resume   # continue after a crash (synthesis already done)
# Output is also saved to hw/vivado/build.log (readable from the Mac side).
set -o pipefail

# Same preload as scripts/de_start.sh. Without it Vivado aborts in libudev (hs_err_pid*.log).
export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1 /lib/x86_64-linux-gnu/libselinux.so.1 /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libgdk-x11-2.0.so.0"
source /home/user/Xilinx/Vivado/2021.1/settings64.sh

cd /home/user/repo/hw/vivado || exit 1
export BOARD_REPO=/home/user/board_files

case "$1" in
  resume) TCL=loopback_resume.tcl ;;
  *)      TCL=loopback.tcl ;;
esac

echo "Running $TCL (log: hw/vivado/build.log)"
vivado -mode batch -source "$TCL" 2>&1 | tee build.log
status=${PIPESTATUS[0]}
echo
if ls ../overlay/loopback.bit ../overlay/loopback.hwh >/dev/null 2>&1; then
  echo "SUCCESS: outputs in hw/overlay/:"; ls -la ../overlay
else
  echo "FAILED (vivado exit $status). Check: grep -n ERROR build.log"
fi
