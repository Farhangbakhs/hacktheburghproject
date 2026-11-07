"""Run ON the PYNQ board (Jupyter terminal or ssh): python3 board_info.py
Prints the versions Jack needs to record (PLAN.md step 1)."""
import platform, socket, subprocess

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"<failed: {e}>"

print("hostname      :", socket.gethostname())
print("ip addresses  :", sh("hostname -I"))
print("kernel        :", platform.platform())
print("python        :", platform.python_version())
try:
    import pynq
    print("pynq version  :", pynq.__version__)
    print("pynq device   :", pynq.Device.active_device.name)
    print("PL clocks MHz :", {k: getattr(pynq.Clocks, k) for k in ("fclk0_mhz", "fclk1_mhz")})
except Exception as e:
    print("pynq          : <import failed>", e)
print("numpy         :", sh("python3 -c 'import numpy;print(numpy.__version__)'"))
print("cpu           :", sh("grep -m1 'model name' /proc/cpuinfo"))
print("mem           :", sh("grep MemTotal /proc/meminfo"))
# /home/xilinx/REVISION can embed a git URL with credentials; print only the first line, URL stripped.
print("image release :", sh("head -n1 /home/xilinx/REVISION 2>/dev/null | sed 's#https\\?://[^ ]*##' || true"))
