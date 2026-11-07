# EVM visual quality upgrades for a 30 fps laptop webcam, under the fixed PLAN.md section 3 FPGA contract

Builds on `../Webcam pulse and bluff tells/evm_realtime.md`. That file's findings (MIT parameters, DoLP frequency response, DSHOW camera control, basic masking and clipping) are not repeated here.

"Computed" means I calculated it with NumPy/SciPy from the filter equations (`.venv` Python, fs = 30, r_hi = 0.3422, r_lo = 0.1596). It is not from an external source. "Inference" means engineering reasoning that should be checked on the camera.

**Key structural insight (inference, but it follows directly from the PLAN.md section 3 maths):** the contract kernel is linear, per pixel and per channel, and uses the same filter on B, G and R. The only nonlinearities are the ±127 clip and int8 rounding. So **any linear spatial filter (blur, pyrDown/pyrUp pooling, masking by a fixed weight), any linear colour transform (BGR→YIQ, chroma scaling, channel selection) and any extra linear temporal filter applied to the int8 output on the laptop is mathematically identical to applying it before the FPGA**, as long as the output does not clip. With alpha = 50, int8 LSB = 0.02 input levels of band and the clip sits at |band| = 2.54 levels (computed). The pulse band at level 2 is roughly 0.1–0.5 levels, so normal skin pixels do not clip, and only motion edges do. Most upgrades below can therefore be done after the FPGA on the laptop with **zero contract change**. Pooling after the FPGA also avoids a real problem with pooling before it: re-quantising a heavily blurred image to uint8 (see Q1).

## 1. Spatial pooling

### Takeaway
160x120 from a 640x480 webcam is only pyramid level 2. MIT used level 4–6 for faces, and Wu et al. showed a size-20 binomial hid the pulse while size 80 revealed it. Add 2 more levels of pooling (effective level 4, about 40x30 over the frame, or about 10–20 px across the face). **Do it on the int8 output on the laptop (pyrDown x2 then pyrUp x2 of the FPGA output), not on the input.** Use skin-masked normalised averaging so background and hair do not bleed into the face.

### Cited Findings
- MIT released code: face = Gaussian level 4, face2 = level 6. A noise experiment showed a binomial filter of size 20 failed to reveal the pulse and size 80 succeeded. "Spatial low pass filtering reduces the variance of the noise according to the area of the low pass filter." — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf); [MIT EVM page](https://people.csail.mit.edu/mrub/evm/)
- Fraunhofer HHI's local rPPG analysis runs "on a Gaussian pyramid level of about 10,000 px (≈100x100)" specifically "to limit noise", with POS per pixel position. That precedent is heavier than 160x120 = 19,200 px over the whole frame, because their 10k px covers the face/tissue region. — [Kossack et al., CVPRW 2022 (arXiv 2208.13840)](https://ar5iv.labs.arxiv.org/html/2208.13840)
- rPPG ROI literature: numerical error rises when the number of effective skin pixels is too small. Non-skin pixels degrade the signal. Some face regions (forehead, cheeks) carry more pulse than others, and SNR-weighted ROIs beat the whole face. — [Model-based ROI segmentation, SciTePress 2019](https://www.scitepress.org/Papers/2019/73898/73898.pdf); [Po et al. 2017 MTA ROI VPPG](https://www.ee.cityu.edu.hk/~lmpo/publications/2017_MTA_ROI_VPPG.pdf); [Wong et al. CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/CVPM/papers/Wong_Optimising_rPPG_Signal_Extraction_by_Exploiting_Facial_Surface_Orientation_CVPRW_2022_paper.pdf)

### Inferences
- **Why post-pool rather than pre-pool.** If the laptop pyrDowns to 40x30, pyrUps back to 160x120 and rounds to uint8, the pooled noise falls well below 1 LSB. The input then loses its natural dither, and the sub-LSB pulse (~0.1–0.5 levels) gets quantised into stair-steps or vanishes. At level 2 the sensor noise (several levels std at your dim luma 23–61) still dithers the uint8 input, so the FPGA's Q16.16 state can recover the sub-LSB mean. If you must pre-pool, add ±0.5 LSB triangular dither before rounding (`np.random.triangular(-1,0,1,shape)`). That is still contract-compliant (160x120 BGR uint8).
- **Recommended:** keep sending plain pyrDown x2 (level 2). On the int8 output: `d = out.astype(np.float32)` → `cv2.pyrDown` x2 → `cv2.pyrUp` x2 (to 160x120) → existing pyrUp x2 to 640x480 → add. Expose a "pool levels" slider with values 0–3 (effective level 2–5). Default 2 (effective level 4). Noise std drops by about 2x per level (≈4x at +2 levels, from the area law above).
- **Masked normalised pooling** (better than plain blur at the face edge): `num = blur(d*m)`, `den = blur(m)`, `d_pool = num/(den+1e-3) * m`, with m = feathered skin mask and `blur = cv2.GaussianBlur(sigma≈4–6 px at 160x120)` or pyramid. This averages only skin. Hair, background and eyes are not mixed into the cheek, and the edge halo is not spread inward.
- **Bilateral / guided filter** (`cv2.ximgproc.guidedFilter`, which contrib has): edge-preserving pooling keeps skin from mixing with background. But it *preserves* strong temporal edges (motion halos), which is the opposite of what you want, and it gives less noise reduction than a mask-normalised Gaussian at equal radius. Masked normalised Gaussian is simpler and better here. Use guided filtering only to feather the mask edges (guide = grayscale frame).
- **Skin-region averaging to one value per region** (e.g. a 4x4 grid of face patches) is the extreme version: very clean, but blocky. Pyramid level 4–5 of the masked output looks like MIT's smooth "blush".

### Gaps
- No source gives an optimal pooling radius in "pixels across the face" for webcams. MIT's level 4 on a ~960x544 video with a large face is the best anchor. Tune by eye with the slider and the Q7 metrics.

## 2. Colour space and which channel to amplify

### Takeaway
The pulse lives mainly in green and in a specific chroma direction. The POS projection (S1 = G−B, S2 = G+B−2R) is the best-studied way to extract it under lighting changes. Because the FPGA applies the same linear filter to each of B, G and R, you can **apply any 3x3 colour matrix to the int8 output on the laptop** (exact, no contract change). Recommended render: project the output delta onto the pulse direction (or YIQ with luma attenuated or kept and chroma scaled), then map back to BGR. Converting to YCrCb *before* sending would also work numerically, but it violates the "BGR" label in section 3. Flag it and don't do it.

### Cited Findings
- MIT processes in YIQ and attenuates chroma (I, Q) to avoid colour clipping artefacts. For the pulse results chromAtn = 1 (no attenuation); for motion results it is 0.1. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf); [MIT EVM page](https://people.csail.mit.edu/mrub/evm/)
- POS (Wang, den Brinker, Stuijk, de Haan, IEEE TBME 2017) defines a plane orthogonal to the skin tone in temporally normalised RGB. Projecting onto it cancels intensity/specular variation. — [Algorithmic Principles of Remote PPG (summary)](https://typeset.io/pdf/algorithmic-principles-of-remote-ppg-300xdjrz7d.pdf)
- Concrete POS form used per pixel by HHI: S1 = g − b, S2 = g + b − 2r, h = S1 + (σ(S1)/σ(S2))·S2 on temporally normalised channels, with a 10 s window, 1 s step and 5th-order Butterworth 0.6–4.0 Hz. POS was chosen because it "performed best under different lighting". — [Kossack et al. CVPRW 2022](https://ar5iv.labs.arxiv.org/html/2208.13840)

### Inferences
- **Laptop-side colour rendering of the int8 output `d` (B,G,R planes):**
  1. *Noise-suppression option:* convert d to YIQ (linear 3x3), multiply Y by `k_Y` ≈ 0.3–1 (luma carries most motion/illumination flicker) and I/Q by `k_C` ≈ 1–2, then convert back. This is the MIT knob, applied afterwards.
  2. *Pulse-direction option (cleanest):* per-pixel scalar `s = w·d` with w = normalised POS-like direction (e.g. w ∝ [B,G,R] weights of −1, +2, −1 in "G − (R+B)/2" form, or the POS h). Then render `delta = s * c`, where c is a fixed display colour vector (e.g. BGR = [−0.3, −0.2, +1] for "redder when blood arrives"). Noise orthogonal to w (much of the luma/illumination noise) is discarded. Be aware that this is a stylised colour, not the true colour change. Say so in the pitch.
  3. *Green-only:* amplify only the G output channel (or G − mean(R,B)). Simple, and about 1/3 the noise energy of all three channels, but it shows as green/magenta flicker rather than "blush".
- Pre-FPGA YCrCb conversion gives no benefit over post-FPGA conversion (linearity) and it breaks the BGR semantics in section 3. **Do not change the contract for this.**
- Normalise d by the local skin mean (d / lowpass(frame)) before rendering, so dark and light skin regions get comparable relative amplification. This is the "temporally normalised" step of POS/CHROM.

### Gaps
- I found no paper that compares YIQ-EVM vs POS-direction rendering for *visual* quality. That choice is aesthetic plus Q7 metrics.

## 3. Temporal filter

### Takeaway
The contract DoLP is broad: -3 dB band 0.50–3.42 Hz, peak gain 0.41. Only 26% of white-noise output power falls in 0.83–2 Hz (computed). A 2nd-order (4-pole) Butterworth 0.83–2 Hz would put 79% in-band. **You can cascade a laptop-side temporal filter on the int8 output** (exact, by linearity). Adding a first-order low-pass at 2.5 Hz raises the in-band fraction to 43%. Adding a laptop-side biquad or high-pass that turns the cascade into a proper band-pass gets close to Butterworth. An ideal FFT window filter is cleanest but needs 5–10 s buffers and lag. Not worth it live.

### Cited Findings
- MIT: "for color amplification of blood flow, a narrow passband produces a more noise-free result". Ideal filters were used for colour. The real-time app used a 2nd-order Butterworth IIR. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- DeepMag band-passes its accumulated change with a 6th-order Butterworth 0.7–2.5 Hz. It notes the filter is there "to remove noise rather than select the target", so the band can be conservative. — [Chen & McDuff, DeepMag (arXiv 1808.03338)](https://ar5iv.labs.arxiv.org/html/1808.03338)
- HHI local rPPG: 5th-order Butterworth 0.6–4.0 Hz. — [Kossack et al. 2022](https://ar5iv.labs.arxiv.org/html/2208.13840)
- Computed |H| at f = 0.3 / 0.5 / 0.83 / 1.0 / 1.3 / 2.0 / 3.0 / 5.0 / 10 Hz:
  - Contract DoLP: 0.195 / 0.291 / 0.380 / 0.401 / 0.412 / 0.383 / 0.316 / 0.223 / 0.137
  - DoLP + laptop 1st-order LP at 2.5 Hz: 0.194 / 0.286 / 0.361 / 0.373 / 0.367 / 0.301 / 0.206 / 0.104 / 0.040
  - DoLP cascaded twice: 0.038 / 0.085 / 0.145 / 0.161 / 0.170 / 0.146 / 0.100 / 0.050 / 0.019
  - Butterworth N=1 band (biquad) 0.83–2 Hz: 0.219 / 0.385 / 0.707 / 0.870 / 1.0 / 0.707 / 0.424 / 0.224 / 0.073
  - Butterworth N=2 band 0.83–2 Hz: 0.050 / 0.171 / 0.707 / 0.952 / 1.0 / 0.707 / 0.214 / 0.053 / 0.005
  - Fraction of white-noise output power inside 0.83–2 Hz: DoLP 0.26; DoLP+LP2.5 0.43; Butter N=1 0.56; Butter N=2 0.79.

### Inferences
- **Laptop post-filter (recommended, no contract change):** run a per-pixel IIR on the pooled output (40x30 or 160x120 float32, which is negligible cost). Option A: one first-order LP at fc 2.5 Hz (`y += r*(d - y)`, r = 1 − exp(−2π·2.5/30) = 0.408). Option B (better): a SciPy biquad, `sos = butter(2, [0.75, 2.2], 'band', fs=30, output='sos')`, applied per pixel with `sosfilt` state (`zi` arrays). Cascaded with the DoLP, the response is dominated by the Butterworth skirts. Then rescale the gain (the cascade's peak ≈ 0.4) with a laptop-side gain so the "alpha" slider still means the visible strength.
- Extra cascade group delay (Butterworth N=2 at ~1.2 Hz): roughly 100–200 ms (not computed precisely). That is acceptable visually.
- **Adaptive narrowband (big win when HR is known):** once `bpm.py` has a stable HR f0, centre a narrow laptop-side biquad on f0 ±0.15 Hz (`butter(1,[f0-0.15,f0+0.15],'band')`). This approximates MIT's 0.83–1.0 Hz ideal filter, which was tuned to the subject. Re-tune it slowly (e.g. every 5 s) to avoid transients. Inference, but it is exactly what MIT did offline.
- Clipping caveat: post-filtering cannot undo the ±127 clip at motion edges. Keep the FPGA alpha moderate (≈20–30; band clips at 6.35 levels for alpha 20, computed) and apply the remaining gain on the laptop *after* pooling and post-filtering. This is the single most useful use of the alpha register.

### Gaps
- I found no published quantitative visual comparison of DoLP vs Butterworth vs ideal on faces (same gap as the previous notes).

## 4. Masking (skin only)

### Takeaway
MIT manually masked faces. Do it automatically: a face landmark/detector mask intersected with a YCrCb skin threshold, eroded, then feathered. Apply it to the output delta (after pooling, or inside masked normalised pooling). This removes background speckle and most hair and edge halos, and is the cheapest large visual gain.

### Cited Findings
- MIT restricted magnification to user-marked regions for face/wrist "since some regions... when amplified, are perceptually unappealing". — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Widely reused YCrCb skin thresholds (Chai & Ngan): 77 ≤ Cb ≤ 127 and 133 ≤ Cr ≤ 173. Variants use 136 ≤ Cr. — [arXiv 2002.01779 (robotics paper restating the ranges)](https://arxiv.org/pdf/2002.01779); [arXiv 1407.6318](https://arxiv.org/pdf/1407.6318)
- Use as many skin pixels as possible and few non-skin pixels. Some regions carry more pulse. — [SciTePress 2019](https://www.scitepress.org/Papers/2019/73898/73898.pdf); [Po et al. 2017](https://www.ee.cityu.edu.hk/~lmpo/publications/2017_MTA_ROI_VPPG.pdf)

### Inferences
- The repo already has `models/face_detection_yunet_2023mar.onnx` (cv2.FaceDetectorYN) and `lbfmodel.yaml` (cv2.face.createFacemarkLBF, 68 landmarks, in contrib). Mask recipe at 160x120: convex hull of landmarks 0–26 (jaw + brows), minus eye and mouth polygons. Shift the brow line up ~0.25 face-height to include the forehead. Intersect with YCrCb skin. **At your dim luma, Cr/Cb thresholds get unreliable**, so compute them on a pyrDown-ed (denoised) frame and use a loose range such as Cr 130–180, Cb 70–135. Erode 2 px at 160x120, then `GaussianBlur` sigma 3 px (feather). Recompute every 3–5 frames and EMA-smooth the mask (0.7 old / 0.3 new) to avoid mask flicker, because mask flicker itself creates temporal edges.
- Apply the mask to the *output* (laptop), never to the input sent to the FPGA. Masking the input would create a moving hard edge that the filter amplifies.
- Optional per-region weighting: weight forehead/cheeks 1.0 and nose/chin/jaw edge 0.5 (pulse-rich regions, per the ROI literature).

### Gaps
- No source on landmark-mask feather widths for EVM. Tune by eye.

## 5. Motion

### Takeaway
Halos arise because any edge displacement passes the band-pass (step response peak 0.31, ~0.9 s decay, from the previous notes). Feasible real-time fixes in order of value: (a) mask (Q4); (b) **send a stabilised face crop instead of the whole frame**, i.e. track the face box, smooth it, crop to 4:3, resize to 160x120 (contract-compatible; this is a Lagrangian-style alignment); (c) motion gating with alpha ramp and FPGA `init` reset; (d) residual sub-pixel alignment by phase correlation. Dense optical-flow warping is feasible at 160x120 but adds its own artefacts. Learned approaches (DeepMag) solve motion best but are not real-time-ready here.

### Cited Findings
- Eulerian methods amplify any in-band change, so head motion at similar frequencies produces "numerous artifacts". On DeepMag's head-rotation task the pulse is "completely buried" in EVM output. EVM PSNR falls from 36.5 dB (still) to 20.3 dB (strong rotation), while DeepMag stays at 38–43 dB. — [Chen & McDuff, DeepMag](https://ar5iv.labs.arxiv.org/html/1808.03338)
- MC-EVM (Benhamida & Kozlovszky, Applied Sciences, Feb 2025) tracks the face with SSD/EfficientDet-Lite0 detectors, crops the face from each frame and runs EVM on the crop to reduce movement effects. No numbers were available in the abstract. — [MC-EVM record](https://library.kab.ac.ug/Record/doaj-art-d3cbabe0dd34406eaec9f462e57dcd9f/Description)
- Forehead-crop EVM with Viola-Jones attenuates jitter and breathing motion. — [Search summary of Ricanek/Honec-style work, VUT](https://dspace.vut.cz/bitstreams/175d02d1-7d46-40ab-af3f-ec3008e6ad9d/download) (not fetched in full)

### Inferences
- **Stabilised face crop (contract-compatible).** Each frame: YuNet box → EMA smooth centre and size (α ≈ 0.1–0.2, plus a dead-band of ~2 px so a still face gives an *exactly* constant box) → expand 1.4x to include forehead and chin → crop at 4:3 → `cv2.resize(..., (160,120), INTER_AREA)` (INTER_AREA = pooling). Send that to the FPGA. Warp the int8 output back into frame coordinates with the inverse affine (`cv2.warpAffine`) and add. Benefits: the face fills all 19,200 px instead of maybe 20%, so you get better skin sampling and pool after. The face also stays registered. Risk: any box jitter becomes global motion, so the dead-band matters. When the box moves more than a threshold, pulse `init`.
- **Residual alignment:** `cv2.phaseCorrelate(prev_gray_crop, cur_gray_crop)` at 160x120 gives a sub-pixel translation in under 1 ms. Fold it into the crop affine. Use ECC (`cv2.findTransformECC`, MOTION_EUCLIDEAN) only if rotation matters. It costs a few ms at 160x120.
- **Gating:** `m = mean(|g_t − g_{t−1}|)` on the masked 40x30 face. If m > T_hi (start at ~1.5 levels), set gain → 0 instantly and send `init=1` once m < T_lo for 5 frames, then ramp the laptop gain 0→1 over 20 frames (~0.7 s). The FPGA filter settles in ~26 frames anyway. Because `init` is a register in the contract, this is free.
- **Per-pixel halo suppression (laptop):** the pulse is small and spatially smooth, while halos are large and edge-aligned. So soft-limit the pooled output with `d_out = T * tanh(d/T)`, T ≈ 2–3x the expected pulse amplitude in output LSB, and zero pixels where the input gradient magnitude is high (edge map from Sobel on the 160x120 frame, dilated, used as an inverse weight).
- Dense Farneback flow at 160x120 costs ~2–5 ms in OpenCV (estimate, not measured). It is feasible, but flow errors on low-texture skin create new artefacts, as the Lagrangian literature warns ([Wadhwa et al. CACM 2017](https://people.csail.mit.edu/ju21743/docs/wadhwa_CACM_2017.pdf) discusses Lagrangian vs Eulerian trade-offs). A rigid crop plus phase correlation is enough for a seated subject.

### Gaps
- No published thresholds for gating. MC-EVM numeric results were not accessible.

## 6. Newer methods and the rPPG-guided hybrid

### Takeaway
Learning-based magnification (Oh et al. 2018, and EulerMormer 2023) targets **motion**, not colour. Phase-based methods are also motion-only. DeepMag (Chen & McDuff 2018) is the closest thing to "rPPG-guided colour magnification": it backprops from a pulse-estimation CNN, so only pulse-related pixel changes are amplified, and it beats EVM strongly under head motion. It is not real-time on a CPU. A **hybrid** is defensible and looks much better: estimate a clean pulse waveform from skin (POS over the face ROI), then use it to *gate or project* the per-pixel FPGA output rather than to *replace* it. Pure "render a synthetic waveform onto the mask" is the best looking but is a visualisation, not magnification. It would also make the FPGA output cosmetic, which judges may call out.

### Cited Findings
- Oh et al., "Learning-based Video Motion Magnification" (ECCV 2018 oral): a CNN learned from synthetic data on two-frame inputs gives fewer ringing artefacts and better noise behaviour. Temporal filters work on its representation up to moderate magnification. It is a motion method. — [arXiv 1804.02684](https://arxiv.org/abs/1804.02684v3)
- EulerMormer (2023) is a transformer-based Eulerian *motion* magnification method that discusses spatial-frequency collapse and blur artefacts. — [arXiv 2312.04152](https://arxiv.org/pdf/2312.04152)
- DeepMag: a CNN trained on video and contact PPG. The input is gradient-ascended to increase the predicted pulse derivative's L2 norm (N = 20 iterations, step 6e-5, L1-normalised gradient with sign correction), then the change is Butterworth band-passed 0.7–2.5 Hz. Result: fewer artefacts and blur with a similar magnification degree. No runtime is reported. — [Chen & McDuff 2018](https://ar5iv.labs.arxiv.org/html/1808.03338)
- A 2024 pulse-signal magnification network targets rPPG on compressed video. — [arXiv 2405.02652](https://arxiv.org/pdf/2405.02652v1.pdf)
- The local rPPG literature visualises pulse as per-pixel amplitude, SNR and **reference-correlation** maps, where the reference is a face-region rPPG signal. It treats correlation with a global reference as a per-pixel reliability measure. — [Kossack et al. 2022](https://ar5iv.labs.arxiv.org/html/2208.13840); [HHI blood-flow project](https://www.hhi.fraunhofer.de/en/departments/vit/research-groups/computer-vision-graphics/research-topics/video-based-blood-flow-analysis.html)

### Inferences
- **Recommended hybrid ("reference-gated EVM", laptop-only, contract unchanged):**
  1. Global reference p(t): POS on the masked face mean (the existing `bpm.py` ROI). Band-pass it with Butterworth N=2 0.75–2.5 Hz, normalised to unit std over 3 s.
  2. Per-pixel pooled FPGA output d(x,t) at 40x30 (after colour projection to scalar s, Q2). Run EMA correlation `C(x) = EMA(s·p) / sqrt(EMA(s²)·EMA(p²))` with a time constant of ~3 s.
  3. Display `delta(x,t) = gain · max(C(x),0)^γ · s(x,t) · mask(x)` with γ ≈ 1–2, or the projection `delta = gain · EMA(s·p)(x) · p(t)`. The second form gives each pixel a *measured* amplitude (from the FPGA output) times a *clean* shared waveform. It is very smooth and noise-free, and phase-locked to the real pulse.
  - Form 1 keeps genuine per-pixel temporal behaviour and only suppresses incoherent noise and halos. Form 2 is the "pulse-synchronous colour modulation" look. It is honest if described as "amplitude map from the FPGA, waveform from rPPG".
- **Pure synthetic render** (`delta = gain·p(t)·mask·colour`) ignores the FPGA output. It looks great, but defend it only as a separate "rPPG overlay" mode, not as EVM.
- Costs: all at 40x30, a few elementwise ops per frame, well under 1 ms (estimate).
- Failure mode: if `bpm.py` locks onto the wrong frequency (motion, or flicker at ~10 Hz aliasing), form 2 renders a confident wrong pulse. Gate it on the SNR of p (Q7 formula, e.g. require > 0 dB) and fall back to plain EVM.

### Gaps
- No 2020–2026 paper found that does real-time CPU rPPG-guided colour magnification specifically. The closest are DeepMag (offline CNN) and the HHI reference-correlation maps (analysis, not magnification video). The hybrid above is my synthesis.

## 7. Objective quality metrics

### Takeaway
There is no standard colour-magnification metric. Papers use (a) PSNR/SSIM of the magnified video vs the original as an *artefact* proxy (DeepMag), (b) pulse SNR in a spectral band around HR (the rPPG standard, e.g. Kossack Eq. 4), and (c) HR accuracy vs a contact reference. For tuning, measure on the output delta: skin pulse SNR, background noise RMS, their ratio, and peak artefact during a scripted head turn.

### Cited Findings
- DeepMag evaluated colour magnification by per-frame PSNR and SSIM against the unmagnified video ("artefacts affect these scores more than magnified physiology"). EVM was 36.5/35.1/24.8/20.3 dB and DeepMag 38.2/42.8/42.8/38.5 dB across tasks with increasing motion. — [Chen & McDuff 2018](https://ar5iv.labs.arxiv.org/html/1808.03338)
- Pulse SNR (Kossack 2022 Eq. 4): SNR = 10·log10( Σ(U·M)² / Σ((1−U)·M)² ) over 0.6–4.0 Hz, with U = 1 within ±0.05 Hz of f_HR and 2·f_HR. Also reference correlation ρ_ref against the region rPPG or a sine at HR. — [Kossack et al. 2022](https://ar5iv.labs.arxiv.org/html/2208.13840)
- MIT gives only qualitative results plus an analysis showing error grows with alpha and with spatial frequency. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)

### Inferences (concrete test harness, all on a recorded 30 s clip so variants are comparable)
- **M1 skin pulse SNR:** mean of output delta (scalar s) over the face mask, then Welch PSD, then the Kossack SNR with f_HR from a phone or watch. Higher is better.
- **M2 background noise:** RMS of delta outside an enlarged face box (and inside the skin mask with the pulse band notched out). Lower is better.
- **M3 contrast:** pulse amplitude in skin / M2 in dB, which is the single "how clean does it look" number.
- **M4 spatial coherence:** mean per-pixel correlation of s(x,t) with the face-mean waveform over the skin mask. High means a uniform "blush"; low means speckle.
- **M5 motion artefact:** during a scripted 10° head turn, max |delta| outside the skin mask and the time for |delta| to return below 2x baseline.
- **M6 fidelity:** PSNR/SSIM of the output vs the original (DeepMag-style) at equal M1, as an artefact proxy.
- **M7 fixed-point:** golden-vector bit-exactness is already in PLAN (unchanged).

### Gaps
- No accepted single metric for visual colour-magnification quality. A quick 5-person A/B preference vote would be a legitimate supplement for the pitch.

## 8. Camera trade-offs (fps, exposure, resolution, denoising)

### Takeaway
Your dim scene (skin luma 23–61) is photon-starved. **More light (a diffuse desk lamp or monitor-white facing the subject, ideally not PWM-flickering) is the biggest single gain and costs nothing.** Dropping to 15 fps with ~66 ms exposure gives about √2 better per-frame SNR. The pulse band is still well below Nyquist (7.5 Hz), and r registers can be recomputed (r_hi = 0.567, r_lo = 0.294 at 15 fps, computed). But PLAN section 3 says "30 fps, fixed", so this is a **contract deviation to flag**, though no FPGA rebuild is needed. 1280x720 helps only if the camera really delivers it at 30 fps without heavy MJPG compression. Compression hurts rPPG more than frame rate does.

### Cited Findings
- Reducing frame rate from 120 to 30 Hz and halving resolution (658x492 to 329x246) "had little impact on the accuracy of pulse rate measurements" (Blackford & Estepp). — [cited via search summary; McDuff 2017 compression study](https://microsoft.com/en-us/research/uploads/prod/2020/11/McDuff_2017_Impact2.pdf)
- Video compression reduces rPPG accuracy as bitrate drops. rPPG trained on lossless data "effectively fails" at H.264/H.265 CRF ≥ 22. — [McDuff et al. 2017](https://microsoft.com/en-us/research/uploads/prod/2020/11/McDuff_2017_Impact2.pdf); [Springer 2023 chapter](https://link.springer.com/chapter/10.1007/978-3-031-31438-4_38)
- Vendor guidance (Noldus FaceReader vital signs): at least 15 fps, 30 fps preferred. — [Noldus FaceReader 10 guidelines](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/facereader/page/facereader-10-guidelines-for-an-optimal-vital-signs-measurement)

### Inferences
- **Order of actions:** (1) add light so skin luma reaches ~100–180. Shot-noise SNR scales with √photons, so 3x the light gives ~1.7x SNR at the same fps. (2) Keep YUY2 640x480 at 30 fps (uncompressed, which matters given the compression findings). (3) Only if still noisy, try 15 fps with DSHOW exposure -4 (≈62 ms), and set r_hi/r_lo for fs = 15 through the registers. Also check whether the driver actually honours it (measure the real fps). Laptop cams often ramp *gain* instead, which adds noise rather than photons.
- **720p:** many integrated cams give 1280x720 at 30 fps only as MJPG, and YUY2 at 720p often only 10 fps (driver-dependent, not verified for this laptop). MJPG blocking adds structured temporal noise exactly at block edges. If YUY2 720p at 30 is available, it gives ~3x the pixels per face area, so ~1.7x lower noise after pooling *if* the 640x480 mode isn't already a binned/downscaled readout. Test it with M2 rather than assume.
- **Mains flicker:** UK 100 Hz lighting aliases at 30 fps. The driver "power line frequency = 50 Hz" setting (via `CAP_PROP_SETTINGS` dialog) helps. At 15 fps, exposure ≈ 60–70 ms spans ~6–7 flicker cycles, which also averages it out.
- **Denoising before the filter:** spatial denoising *is* pooling (Q1), so do it post-FPGA. Temporal denoising (e.g. `cv2.fastNlMeansDenoisingColoredMulti`) is too slow (tens of ms or more at 640x480, estimate) and it is itself a temporal filter that could smear the pulse. Avoid it. Turn *off* the camera's own noise reduction or sharpening if the driver dialog exposes it, because in-camera temporal NR can suppress the pulse.

### Gaps
- No source isolates 15 vs 30 fps for webcam rPPG. The capabilities of this specific webcam (YUY2 720p fps, whether the gain is fixed) are unknown. The user must test.

## 9. Ranked improvements (impact / effort / contract impact)

### Takeaway
The top five are all laptop-side, need no contract change, and should together turn "speckly with halos" into a smooth, face-only blush: light, post-FPGA pooling, skin mask, post temporal filter with gain split, and motion gating with `init`. The rPPG-gated hybrid is the "wow" upgrade after those.

### Cited Findings
- Supporting evidence for each item is cited in Q1–Q8 above (MIT pooling and masking, Kossack POS and SNR, DeepMag motion and metrics, McDuff compression).

### Inferences
| # | Change | Expected impact | Effort | Contract impact | Key parameters |
|---|---|---|---|---|---|
| 1 | More light on face | High (speckle ↓ ~1.5–2x) | 5 min | None | Skin luma 100–180, diffuse, non-PWM, power-line 50 Hz |
| 2 | Post-FPGA spatial pooling (pyrDown/Up the int8 output; masked normalised) | High (noise std ↓ ~4x at +2 levels) | 30 min | None (linearity) | +2 levels (effective level 4); slider 0–3 |
| 3 | Feathered face+skin mask on output | High (removes background speckle, most halos) | 1 h (YuNet/LBF exist in repo) | None | Landmark hull + forehead, YCrCb Cr 130–180 / Cb 70–135, erode 2 px, blur σ 3 px @160x120, EMA mask |
| 4 | Gain split: FPGA alpha ≈ 20–30, rest on laptop after pooling and filtering | Medium–High (prevents ±127 clipping, cleaner) | 10 min | None (uses alpha register) | alpha_fpga 20–30; laptop gain 2–5x; soft-limit tanh |
| 5 | Motion gating + `init` reset + alpha ramp | High for halos | 30–45 min | None (init register exists) | T ≈ 1.5 levels mean abs diff on face @40x30; ramp 20 frames |
| 6 | Laptop temporal post-filter (Butterworth biquad, or HR-locked narrow band) | Medium (in-band noise fraction 26% → 40–80%) | 45 min | None | `butter(2,[0.75,2.2],'band',fs=30,output='sos')`; narrow f0 ± 0.15 Hz once HR is stable |
| 7 | Colour rendering: YIQ chroma/luma weights or POS-direction projection | Medium (cuts luma flicker; better "blush") | 30 min | None (post-FPGA 3x3) | k_Y 0.3–1, k_C 1–2; or s = G − (R+B)/2 → render on red |
| 8 | Reference-gated hybrid (POS waveform × per-pixel FPGA amplitude) | Very high visual quality | 2–3 h | None; be transparent in pitch | EMA corr τ ≈ 3 s; SNR gate > 0 dB; fallback to plain EVM |
| 9 | Stabilised face crop sent as the 160x120 input (+ phaseCorrelate) | Medium–High (more skin pixels, registered) | 2–3 h | None in format; changes *what* the 160x120 shows (tell team; golden vectors should use the same crop mode) | 1.4x box, EMA 0.1–0.2, 2 px dead-band, INTER_AREA, init on big box jumps |
| 10 | 15 fps / long exposure | Medium in dim light | 15 min | **Deviation: section 3 says 30 fps fixed**; r regs recomputed (r_hi 0.567, r_lo 0.294) | DSHOW exposure -4; verify real fps |
| 11 | 720p capture then pool | Low–Medium, uncertain | 20 min test | None (laptop pre-processing) | Only if YUY2 720p at 30 fps; crop 4:3 |
| 12 | Learned (DeepMag / Oh) | High offline, not real-time | Days | N/A | Skip for hackathon |

- Pre-FPGA YCrCb conversion, a cascaded filter in hardware, or a Butterworth in hardware would each change section 3. **None are needed**, because equivalent laptop-side post-processing exists (linearity). Do not propose them to Jack.
- Suggested measurement loop: record one 30 s YUY2 clip (still, then a head turn). Run variants offline through `evm_fixed` + post-processing, and log M1–M5 per variant. Pick defaults, then go live.

### Gaps
- Impact estimates are engineering judgement from the noise area law and the computed filter responses, not measured on this camera. Validate them with M1–M5.
