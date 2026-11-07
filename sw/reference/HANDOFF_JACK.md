# Handoff to Jack: switching the FPGA kernel to the saracen (Laplacian) EVM

From: Person 2 (reference algorithm), 2026-11-07
Status: **proposal, needs team sign-off.** I have **not** edited `docs/PLAN.md` (shared folder). If we agree, section 3 needs updating to "v2" as described here. The v1 golden vectors are still provided, so nothing you've built against v1 is wasted.

---

## TL;DR

- We're switching the algorithm from MIT's **colour** EVM (v1, PLAN.md section 3) to MIT's **motion** EVM, as implemented in [saracen/magnification](https://github.com/saracen/magnification) (C++, MIT licence).
- **The FPGA does the same maths as before:**
  - per value: `L_hi += r_hi*(x-L_hi)>>16; L_lo += r_lo*(x-L_lo)>>16; out = alpha*(L_hi-L_lo)`
  - same Q0.16 coefficients, same INIT semantics, same 32-bit AXI-Stream words.
- **What changes:**
  - *Input:* signed int16 Laplacian band values, brightness (L) channel only, instead of uint8 BGR pixels.
  - *Frame:* 25,200 values per frame (three pyramid levels), instead of 19,200 pixels x 3 channels.
  - *Gain:* one alpha register per level instead of one global alpha.
- **BRAM for state drops from 460 KB to 202 KB.**
- Golden vectors: `test_vectors/v2_saracen/` (face clip, N=300) and `test_vectors/v2_saracen/edge/` (stress cases). Verify with `gen_vectors_saracen.py verify` (below).

---

## 1. Why we switched

1. **v1 colour EVM didn't look good enough on our webcam.** The heartbeat colour change is about 0.5% of skin brightness. With venue lighting the face reads 23–61 / 255 in brightness, and our pulse SNR on the face was 0–4 dB. The magnified output looked like random flicker rather than a pulse.
   - It only looks like MIT's video offline: an ideal narrow band picked around the person's heart rate, heavy pooling, and a brightly lit subject.
   - We built that (`evm_mit.py`, `evm_locked.py`), and it needs about 8 s to lock onto the pulse plus good light.
2. **The motion method looks clearly better live, and copes with this lighting.** On the live dashboard it shows head and skin micro-motion from pulse and breathing.
   - We ported saracen's C++ unchanged as an embedded DLL (`cpp/`) and compared side by side. The team judged it better on our face.
   - We tuned it for faces (half resolution, `lambda_c` 40). That cut skin grain by 64% compared with saracen's defaults, measured on a recorded clip, and runs at 30 fps.
3. **It's still FPGA-friendly.** saracen's temporal filter is *identical* to our section 3 IIR, so your kernel design, pipeline (II=1), DMA and AXI-Lite plumbing all carry over. Only the stream contents and a few registers change.
4. **It needs less BRAM.** saracen discards the finest Laplacian band and the coarsest residual, so the FPGA filters only the three middle bands: 160x120 + 80x60 + 40x30 = 25,200 values × 2 states × int32 = **201.6 KB**. v1 needed 460 KB for 160x120x3.

## 2. Partition (who does what)

```
Laptop                                      FPGA (your kernel)                    Laptop
------                                      ------------------                    ------
640x480 BGR
 -> pyrDown -> 320x240
 -> BGR->Lab (float)
 -> Laplacian pyramid of L, 4 bands
 -> send bands 1..3 as int16 Q8.8  ----->   per value: IIR bandpass   ----->   bands -> /256
    (25,200 words, 32-bit each)             x alpha[level], clip int16          rebuild pyramid (band 0, residual = 0)
                                            state 25,200 x 2 int32              L += motion; Lab->BGR
                                                                                mask to face skin, add to frame
```

The PYNQ ARM only forwards words between TCP and DMA; no image processing runs there.

## 3. Kernel spec (bit-exact)

The reference model is `sw/reference/evm_saracen_fixed.py` → class `SaracenKernel`.

Per frame, for `i = 0 .. total-1`, where `total = count_0 + count_1 + count_2 + count_3`:

```
v      = int16 from low 16 bits of input word i           (Q8.8; high 16 bits are 0)
x      = (int64)v << 8                                    (Q16.16)
if INIT or first frame:  L_hi[i] = L_lo[i] = x
L_hi[i] += ((int64)r_hi * (x - L_hi[i])) >> 16           (ARITHMETIC shift = floor)
L_lo[i] += ((int64)r_lo * (x - L_lo[i])) >> 16
band   = L_hi[i] - L_lo[i]                                (Q16.16)
lvl    = 0 if i < count_0, 1 if i < count_0+count_1, 2 if i < count_0+count_1+count_2, else 3
out    = clip(((int64)alpha_lvl * band) >> 16, -32767, 32767)   (Q8.8)
output word i = (uint16)out in low 16 bits, high 16 bits = 0;  TLAST on i == total-1
```

- State fits int32: |L| ≤ 32767 << 8 = 2^23. Products need 64 bits: r × diff ≤ 2^16 × 2^24, alpha × band ≤ 2^16 × 2^24.
- On an INIT frame the update changes nothing (x == L), so its output is all zeros. Same as v1.
- `>>` on negative values must floor: `(-1) >> 16 == -1`. `ap_int` does this. Cast the unsigned registers to a signed 64-bit type **before** multiplying.

### Registers (AXI-Lite; values for our defaults)

| Name | Type | Default | Meaning |
|---|---|---|---|
| r_hi | uint16 Q0.16 | 26223 | high cutoff 2.44 Hz @ 30 fps (saracen's 0.4) |
| r_lo | uint16 Q0.16 | 3343 | low cutoff 0.25 Hz @ 30 fps (saracen's 0.05) |
| alpha_0 | uint16 Q8.8 | 3968 (=15.5) | gain, level 0 (160x120) |
| alpha_1 | uint16 Q8.8 | 5120 (=20) | gain, level 1 (80x60) |
| alpha_2 | uint16 Q8.8 | 5120 (=20) | gain, level 2 (40x30) |
| alpha_3 | uint16 Q8.8 | 0 | spare level (unused by default) |
| count_0..3 | uint16 | 19200, 4800, 1200, 0 | words per level, in stream order |
| init | 1 bit | | re-initialise state from this frame |

The per-level alphas come from saracen's spatial-wavelength clamp. The laptop computes them and you just apply them. Default BRAM sizing: `MAX_WORDS = 25200` (or 25600 for headroom).

### HLS sketch (not compiled; for orientation)

```cpp
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#define MAX_WORDS 25200
typedef ap_axiu<32, 0, 0, 0> word_t;

void evm_v2(hls::stream<word_t>& in, hls::stream<word_t>& out,
            ap_uint<16> r_hi, ap_uint<16> r_lo,
            ap_uint<16> alpha_0, ap_uint<16> alpha_1, ap_uint<16> alpha_2, ap_uint<16> alpha_3,
            ap_uint<16> count_0, ap_uint<16> count_1, ap_uint<16> count_2, ap_uint<16> count_3,
            ap_uint<1> init) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=r_hi   // ... same for every scalar, plus return
#pragma HLS INTERFACE s_axilite port=return
  static ap_int<32> L_hi[MAX_WORDS], L_lo[MAX_WORDS];
  static bool first = true;
  const ap_uint<17> e0 = count_0, e1 = e0 + count_1, e2 = e1 + count_2, total = e2 + count_3;
  const bool do_init = init || first;
  for (ap_uint<17> i = 0; i < total; i++) {
#pragma HLS PIPELINE II=1
    word_t w = in.read();
    ap_int<16> v = w.data.range(15, 0);
    ap_int<64> x = ((ap_int<64>)v) << 8;
    ap_int<64> lh = do_init ? x : (ap_int<64>)L_hi[i];
    ap_int<64> ll = do_init ? x : (ap_int<64>)L_lo[i];
    lh += (((ap_int<64>)r_hi) * (x - lh)) >> 16;
    ll += (((ap_int<64>)r_lo) * (x - ll)) >> 16;
    L_hi[i] = lh; L_lo[i] = ll;
    ap_uint<16> a = i < e0 ? alpha_0 : i < e1 ? alpha_1 : i < e2 ? alpha_2 : alpha_3;
    ap_int<64> o = (((ap_int<64>)a) * (lh - ll)) >> 16;
    if (o > 32767) o = 32767;
    if (o < -32767) o = -32767;
    word_t r; r.data = 0; r.data.range(15, 0) = (ap_int<16>)o;
    r.keep = -1; r.strb = -1; r.last = (i == total - 1);
    out.write(r);
  }
  first = false;
}
```

## 4. Golden vectors

Everything is in `test_vectors/`. Each set has a `params.json` holding the registers, `N`, level shapes, `init_frames` and sha256 checksums.

| Folder | What | Size |
|---|---|---|
| `v2_saracen/` | **use this.** Real face clip (`recordings/brightroom.npz`, frames 0–299), default registers, INIT on frame 0 | 300 × 25,200 words |
| `v2_saracen/edge/` | stress: r_hi=0xFFFF, r_lo=1, alphas 0xFFFF/0x8000/0x0100/0; random full range, ±32767 flicker, zeros, sine; INIT on frames 0 and 32 | 64 × 25,200 |
| `v1_contract/`, `v1_contract/edge/` | the old section 3 (160x120 BGR) vectors, kept as fallback | 300 / 64 × 19,200 |

Files in each set:
- `input_words.bin` (v2) or `input_frames.bin` (v1): the stream you feed in, frame after frame.
- `expected_output.bin`: what must come out, bit for bit.
- `expected_state.bin`: `L_hi` then `L_lo` (int32 LE) **after** each of the first 4 frames. If C-sim fails, compare this to find the first stage that diverges.

Check your C-sim or on-board output (write it in the same layout as `expected_output.bin`):

```
cd sw/reference
.venv\Scripts\python gen_vectors_saracen.py verify --got <your_output.bin>
.venv\Scripts\python gen_vectors_saracen.py verify --dir ../../test_vectors/v2_saracen/edge --got <edge_out.bin>
```

It prints `PASS`, or the first 10 mismatches with frame, word index, level, and expected vs got. To regenerate: `python gen_vectors_saracen.py` (deterministic).

## 5. Honest limits / things to agree

- **Brightness only.** The FPGA filters only the Lab L channel; colour (a, b) is not magnified (chroma attenuation 0). saracen's default keeps 10% of colour.
  - Measured on our face clip, the v2 fixed-point pipeline differs from the C++ by a mean of **0.15 grey levels** (chroma 0) and 0.24 (chroma 0.1), against a magnified signal of about 2.3 levels.
  - Filtering colour too would triple BRAM to about 605 KB, so I'd keep it brightness-only.
- **Quantisation.** Bands are sent as Q8.8 int16. On our clip inputs span −5830..10862, well inside ±32767, and output clipping is 0.005%.
- **Fixed sizes.** 640x480 camera → 320x240 working size → levels of 160x120 / 80x60 / 40x30. Other sizes only change `count_i`.
- **Network (Person 3).** The request header needs `r_hi, r_lo, alpha_0..3, count_0..3, init` in place of `w, h, alpha`, and the payload becomes `total_words × 4` bytes, about 101 KB per frame each way (v1: 77 KB). I've only suggested this; the protocol is Person 3's to define.
- **Bandwidth/compute.** 25,200 values per frame at II=1 is about 0.25 ms at 100 MHz, so the bottleneck is DMA/TCP, not the kernel.

## 6. What Person 2 delivered (all in `sw/reference/`, tests in `tests/`)

| File | What |
|---|---|
| `evm_saracen_fixed.py` | **v2 bit-exact kernel model** + laptop pre/post + dashboard processor |
| `gen_vectors_saracen.py` | v2 golden vectors + `verify` |
| `evm_fixed.py`, `gen_vectors.py` | v1 bit-exact model + vectors + `verify` (fallback) |
| `cpp/` | saracen C++: original `main.cpp` (unmodified, MIT licence, commit pinned) + `saracen_evm.cpp` DLL wrapper; `build.cmd` (MSVC 2022 + OpenCV 4.14) |
| `evm_saracen_cpp.py` | ctypes wrapper for the DLL (float reference the v2 model is checked against) |
| `dashboard.py` | live webcam dashboard; presets include `saracen_cpp` (C++ float) and `saracen_fixed` (v2 fixed-point split), so you can compare live what your kernel will produce |
| `face_tracker.py` | YuNet + 68-point LBF face landmarks → feathered skin mask (magnification only on the face) |

Tests: `cd sw/reference && .venv\Scripts\python -m pytest -q tests`. 142 pass and 4 are skipped (the real-face tests need an image in `tests/data/`). The v2 tests include:
- a literal scalar implementation of the spec, matched on every value,
- overflow and clipping extremes,
- the word layout,
- agreement with the C++ DLL,
- vector round-trip and `verify` localisation.
