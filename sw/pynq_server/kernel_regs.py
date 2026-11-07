"""AXI-Lite register map of the HLS kernel (PLAN.md section 3, control registers).

PLACEHOLDER OFFSETS: these follow the usual Vitis HLS layout (CTRL at 0x00,
scalar arguments from 0x10 in 8-byte steps) but the real values come from the
generated driver (xevm_kernel_hw.h) once Jack's kernel builds. Update them here
only; dma_backend.py and mock_pynq.py both read this file.
"""

REG_CTRL = 0x00
REG_WIDTH = 0x10
REG_HEIGHT = 0x18
REG_R_HI = 0x20
REG_R_LO = 0x28
REG_ALPHA = 0x30
REG_INIT = 0x38

# CTRL bits (ap_ctrl_hs)
AP_START = 0x01
AP_DONE = 0x02
AP_IDLE = 0x04
