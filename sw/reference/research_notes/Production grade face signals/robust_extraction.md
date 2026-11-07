# Production-grade facial pulse extraction (rPPG) and face-only colour magnification on a 30 fps laptop webcam (Python 3.11 / OpenCV 4.14 / NumPy-SciPy, CPU)

Scope: new ground beyond `Webcam pulse and bluff tells/rppg_heart_rate.md` and `EVM quality upgrades/evm_quality_options.md`, which already cover POS/CHROM formulas, basic ROIs, detrending, FFT, SNR and rPPG-Toolbox tables. Our measured facts: 640x480 @ 30 fps, face ~180 px wide, very still subject, skin luma ~53/255 (gain and exposure fixed), skin-mean green SNR 3.6 dB at 80x60 pooling, dropping to -3 dB at 20x15 because the Gaussian pyramid mixes background into the face.

Ranked recommendations (impact/effort tags; the evidence for each is in the sections below):
1. **Mask-normalised pooling** [HIGH impact / LOW effort]. Replace the plain pyramid pooling with `blur(I*M)/blur(M)`, where M is a soft skin mask built from the landmarks (eroded, with eyes, brows, nostrils and lips cut out). This removes background leakage, which is the likely cause of the drop from 3.6 dB to -3 dB. (Inference, see Q1.)
2. **Landmark-anchored ROI set: glabella, medial forehead and both malar (upper-cheek) areas** [HIGH / LOW]. Drop the nasal sidewalls, upper lip and chin. (Kim 2021, Kwon 2015.)
3. **Per-block SNR map with SNR-weighted fusion** [MED-HIGH / MED]. Use a landmark-warped grid of about 8x8 px blocks with POS per block, a weight of max(0, SNR_dB - T), and a rolling update every 1 s over a 10 s window. (Po 2018, Ryu 2023.)
4. **Keep POS as the default extractor and also run OMIT/LGI** [MED / LOW]. All three tie at rest. Choose per window by SNR, or simply stay with POS. (Bioengineering 2022 comparison.)
5. **Try ME-rPPG (Apache-2.0, ONNX shipped) as a learned option** [potentially HIGH / MED]. It is the only SOTA model found with ready ONNX files, a CPU-latency claim and a strong MMPD cross-dataset MAE. Run it side by side with tuned POS on our own recordings before trusting it.
6. **HRV: only report it when SQI passes, with sub-frame peak interpolation and IBI filtering** [MED / MED]. At 30 fps, expect RMSSD errors of tens of ms.
7. **Product framing** [process]. Commercial systems require at least 400 lux, stillness, no talking and about 60 s windows. Our luma of 53/255 suggests we are below that light level, so add a UI gate or warning.

---

## Q1. Pixel selection and weighting: which skin regions, how to weight them, how to avoid background leakage

### Takeaway
The thin-skinned regions (glabella, lower and upper medial forehead, left and right malar cheek) carry the best pulse signal. Nasal sidewalls and upper lip are the worst. Combining the top-5 regions beat whole-face skin for CHROM, and roughly tied it for POS. Block-level SNR maps (Po 2018) and SNR-weighted block fusion (Ryu 2023) are the published routes to adaptive weighting. Our own leakage problem is best fixed with mask-normalised (normalised-convolution) pooling, which needs no new research.

### Cited Findings
- Kim, Lee and Sohn (Sensors 2021) tested 39 anatomical facial regions with CHROM, GREEN, ICA, PBV, POS, SSR and LGI on UBFC-rPPG and LGI-PPGI. Top-5 by their composite score: right malar (3.64), glabella (3.33), lower medial forehead (2.98), upper medial forehead (2.88), left malar (2.68). Bottom-5: the nasal sidewalls (left lower 1.07, left mid 1.14, right mid 1.16) and the upper lip (left 1.16, right 1.17). — [Kim et al. 2021, PMC8659899](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659899/)
- On UBFC, POS MAE was 1.85 for the TOP-5 combination, 1.87 for whole Face+Skin and 7.26 for BOT-5. CHROM MAE was 1.5, 2.67 and 5.9. On LGI-PPGI, CHROM MAE was 2.93 for TOP-5 and 4.04 for Face+Skin. The one exception: Face+Skin had higher PCC than TOP-5 for POS on UBFC (0.85 vs 0.80). — [Kim et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659899/)
- Signal quality correlated 0.50 with thinner skin and -0.53 with larger region rank. The authors sum it up as "the thinner the skin and the larger the region, the better". A single top region averaged about 696 px, against about 25,000 px for the whole face, and the authors warn that small regions are more exposed to noise from light and motion. The study covered only Fitzpatrick I-II skin, two datasets, and skin thickness taken from 10 cadavers. — [Kim et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659899/)
- Kwon et al. (EMBC 2015) split the face into 7 regions and scored each by SNR and correlation. The forehead and both cheeks were the best candidates for an efficient ROI. Mouth and chin were weaker, and nose and nasion were limited. — [Kwon 2015 record](https://search.isc.ac/inventory/49/3685055.htm); [AMiner](https://www.aminer.org/pub/568e703c0cf27119656712a0/roi-analysis-for-remote-photoplethysmography-on-facial-video)
- One 20-subject study found the cheeks had the highest mean signal quality of the ROIs tested. Another work chose the forehead over the cheeks because beards distort cheek signals. — [Augsburg study](https://opus.bibliothek.uni-augsburg.de/opus4/files/100529/100529.pdf); [VisioPhysioENet](https://arxiv.org/pdf/2409.16126)
- Po et al. (MTA 2018), block-based adaptive ROI: the face video is divided into spatio-temporal blocks, an SNR map of rPPG quality is estimated, and adaptive ROIs are taken by mean-shift clustering plus adaptive thresholding on the SNR maps. It beat conventional fixed ROIs on both still and moving subjects. The precursor (ICASSP 2015) used non-overlapping blocks, two no-reference quality features and K-means. — [Po 2017/18 PDF](https://www.ee.cityu.edu.hk/~lmpo/publications/2017_MTA_ROI_VPPG.pdf); [Po ICASSP 2015](https://www.ee.cityu.edu.hk/~lmpo/publications/2015_ICASSP_rPPG.pdf)
- Ryu et al. (J. Biomed. Opt. 2023), adaptive-weight spatial averaging: each small block gets a weight from a power-spectrum ratio. If the ratio is below a threshold the weight is 0, otherwise the weight equals the ratio, and the weights are applied to the raw block signals before averaging. — [Ryu 2023 JBO](https://spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-28/issue-8/085003/Spatial-averaging-method-based-on-adaptive-weight-for-imaging-photoplethysmography/10.1117/1.JBO.28.8.085003.pdf) (full text could not be fetched; details come from the search snippet)
- VISAPP 2019 proposed a spatially weighted average of skin pixels, with the weights from a trained model, to favour regions richer in rPPG information. — [doi 10.5220/0007389803830388](https://doi.org/10.5220/0007389803830388)
- The Bioengineering 2022 comparison recommends larger ROIs (forehead plus cheeks) under hard conditions, and the forehead alone under easy ones. — [PMC9598377](https://pmc.ncbi.nlm.nih.gov/articles/PMC9598377/)

### Inferences (concrete algorithm for our pipeline; these are engineering derivations, not cited results)
- **Why our SNR collapses with coarser pooling:** at 20x15 a pyramid cell is about 32 px, so a face 180 px wide is only about 5-6 cells across, and most cells straddle background, hair or eyes. That non-pulsatile content dilutes the pulse and adds motion and edge flicker. Mask-normalised pooling removes the dilution:
  `S = GaussianBlur(I * M, sigma) / (GaussianBlur(M, sigma) + eps)`, sampled only where `blur(M) > 0.5`. Use `cv2.GaussianBlur` or `cv2.resize(INTER_AREA)` on both numerator and denominator, and compute in float32. This is normalised convolution (Knutsson-Westin). The same trick applies to every pyramid level: build pyramids of `I*M` and `M`, then divide level by level.
- **Building the mask M from the 68-pt LBF landmarks** (iBUG indices, 0-based):
  - Face hull: points 0-16 plus a forehead extension. Shift brow points 17-26 upward by about 0.25 x the face height (the distance from the brow line to the chin 8).
  - Subtract polygons for the eyes (36-41, 42-47, each dilated by about 4 px), the brows (17-21, 22-26, dilated by about 3 px), the mouth (48-59, dilated by about 4 px) and the nostril area (31-35).
  - Erode the result by about 3% of face width (about 5 px) to drop the boundary pixels where motion and edge artefacts concentrate.
  - Feather with a Gaussian blur of sigma about 3 px.
- **Regions to keep** (following Kim 2021 and Kwon 2015):
  - Glabella and medial forehead: the polygon between brow inner points 19-24, extended upward.
  - Left malar: roughly the polygon through 1, 2, 3, 31, 39 and 40, kept below the eye.
  - Right malar: the mirror, through 15, 14, 13, 35, 42 and 47.
  - Exclude the nose bridge and sidewalls (27-35) and the upper lip.
- **Specular highlights:** skin luma is about 53/255, so saturation is unlikely, but forehead and nose-tip glints can still appear. Per frame, mark pixels with luma above the ROI's 98th percentile, or any channel at 250 or above, and drop them from M. Also drop pixels darker than the 2nd percentile (shadows, hair).
- **Adaptive skin colour:** inside the landmark hull, fit a 2D Gaussian in normalised rg or YCrCb (Cr, Cb) over about 2 s of frames. Reject pixels beyond a Mahalanobis distance of about 3. This catches hair, glasses and beard without a learned segmenter.
- **Online SNR-weighted fusion (a Po + Ryu hybrid):**
  1. Warp the face to a canonical frame using a similarity or affine transform from the landmarks (`cv2.estimateAffinePartial2D` on stable points 27-30, 36, 39, 42, 45). Put a grid of blocks over the warped skin; about 8x8 px blocks in a 180 px face gives about 60-120 skin blocks.
  2. Per block, keep the masked mean RGB time series (mask-normalised as above).
  3. Every 1 s, over the last 10 s (300 samples): run POS per block, then compute SNR_b = 10·log10(power within ±0.1 Hz of the fundamental and first harmonic, divided by remaining power in 0.7-3 Hz). Find the fundamental from the global POS signal, not per block, to avoid each block locking onto its own noise peak.
  4. Weight w_b = max(0, SNR_b - T), with T about -3 dB (or the median block SNR), and smooth w_b over time with an EMA (alpha about 0.2 per update). Fused pulse = Σ w_b · pulse_b / Σ w_b.
  5. Fallback: if Σ w_b is about 0, revert to the fixed TOP-5 region mean.
  Expected gain is a few dB over a flat mean; this is untested, so measure it.
- When blocks are small, POS per block is noisy. An alternative is to fuse the RGB means with weights first and run POS once (cheaper and more stable), then use per-block SNR only to set the weights.

### Gaps
- Po 2018's block size, mean-shift bandwidth and thresholds, and Ryu 2023's exact ratio definition and threshold value, could not be retrieved (paywalled or failed to load).
- No paper found quantifying normalised-convolution pooling for rPPG specifically. The recommendation is an engineering inference.
- No region-ranking evidence was found for darker skin types; Kim 2021 covered only Fitzpatrick I-II.

---

## Q2. Motion and illumination robustness; POS vs OMIT vs LGI vs CHROM; rendering a clean pulse onto the face

### Takeaway
Among unsupervised methods, POS, LGI and OMIT are effectively tied at the top (overall scores 0.85-0.86). LGI and OMIT edge ahead at rest and POS leads under motion. For a very still subject, extractor choice matters less than ROI and pooling. For face-only magnification, render a synthetic, pulse-locked colour modulation inside the skin mask instead of amplifying raw band-passed pixels.

### Cited Findings
- The Bioengineering 2022 comparison tested 8 unsupervised methods against fingertip PPG over 4 activities. Overall scores: LGI 0.86, OMIT 0.85, POS 0.85, CHROM 0.81, PCA 0.58, GREEN 0.38, PBV 0.30, ICA 0.17. At rest, LGI and OMIT were best (OS 1.00, |ΔBPM| about 1.87). Under motion, POS was best for Gym and Rotation and CHROM for Talk. Correlations were low overall (best about 0.41 at rest). There were only 6 participants. — [PMC9598377](https://pmc.ncbi.nlm.nih.gov/articles/PMC9598377/)
- OMIT (Casado and López, Face2PPG) is a QR-decomposition-based orthogonal transform designed to tolerate video compression artefacts. — [arXiv 2202.04101](https://arxiv.org/abs/2202.04101v1)
- A 2025 paper (PRISM) reports that POS beat CHROM, GREEN, PCA and ICA across skin tones and activity levels in controlled settings. — [arXiv 2511.21903](https://arxiv.org/html/2511.21903v1)
- With the ROI held fixed, CHROM outperformed POS on the TOP-5 ROI in Kim 2021 (UBFC MAE 1.5 vs 1.85), so the ranking depends on the ROI. — [PMC8659899](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659899/)

### Inferences
- **Tracking:** our landmark motion is 0.001 face-widths per frame, so landmark-anchored polygons are enough. Smooth the landmarks temporally with a One-Euro or EMA filter (alpha about 0.3), because LBF jitter of ±1 px on ROI edges injects noise at frame rate. Optical flow (`cv2.calcOpticalFlowPyrLK` on about 50 good features inside the skin mask) is only worth adding when motion grows. Warping blocks to a canonical face (Q1) handles both cases.
- **Illumination compensation:**
  1. Take a background reference patch from a static non-skin region, for example the wall beside the head, outside the face box by at least 0.3 face widths.
  2. Normalise its RGB the same way as the skin trace, then regress it out with LMS or plain least squares per window: `s_clean = s - β·b` with β = ⟨s,b⟩/⟨b,b⟩ over the 10 s window.
  3. Apply this only when the background band-power in 0.7-3 Hz exceeds a threshold. Otherwise it injects the background's own camera noise.
  Mains flicker at 50 Hz aliases to near 0 or 10 Hz at 30 fps, so it is not in the pulse band unless the actual fps drifts. Monitor this.
- **Low light (luma about 53):** the pulse amplitude is fixed in relative terms (about 0.1-0.5% of DC), while quantisation and shot noise grow relatively at low DC. This is the main reason pooling area (Q1) matters so much for us. Use float32 accumulation and avoid any 8-bit intermediate.
- **Face-only magnification render path (pulse-guided colour modulation):**
  1. Get a clean global pulse p(t), normalised to unit variance, from POS on the fused ROI. Band-pass it to HR ± 0.3 Hz around the tracked HR to make it near-sinusoidal.
  2. Get a spatial amplitude map A(x,y) from the per-block SNR weights (Q1), upsampled with the same mask-normalised blur so it falls to 0 at the skin edge.
  3. Render `out = frame + α · A(x,y) · p(t) · c`, where c is a fixed colour direction. Use the POS projection direction mapped back to RGB, or simply (−0.3, 1, −0.3) scaled to the mean skin colour so it reads as flushing. Gate α by SQI: fade to 0 when SNR is below the threshold so noise is never amplified.
  This produces a clean, believable face-only magnification even when raw per-pixel EVM is noise-dominated (our case at luma 53). It is honest as long as it is labelled a visualisation of the measured pulse. If the FPGA path must stay raw EVM (PLAN.md section 3), use this as the host-side reference or as a comparison only, and flag it to the team instead of changing the contract.

### Gaps
- No head-to-head of background-reference compensation versus none under webcam room lighting was found within the time budget.
- No published evaluation of "pulse-guided synthetic rendering" for EVM was found. This is a design proposal.

---

## Q3. SOTA supervised and self-supervised models (2024-2026): can one beat tuned POS on a laptop webcam?

### Takeaway
On the hardest cross-domain benchmark (MMPD, testing on unseen data), even recent deep models score MAE of about 5-18 BPM. ME-rPPG (2025) is the standout: 5.38 BPM trained on RLAP, about 3.6 MB of memory, 9.46 ms per frame on a laptop CPU, Apache-2.0, and ONNX files shipped. Older models (PhysNet, TS-CAN, EfficientPhys, PhysFormer) are 9-18 BPM on MMPD cross-dataset. For a still subject on a webcam, tuned POS is a strong baseline, and only ME-rPPG is cheap enough to A/B test right away.

### Cited Findings
- ME-rPPG (Wang et al., arXiv 2504.01774, Apr 2025): a temporal-spatial state-space-duality model that runs streaming inference frame by frame. The paper reports 3.6 MB memory and 9.46 ms latency, measured on CPU; the real-world deployment used an i7-13620H laptop CPU only. Cross-dataset MAE: 5.38 on MMPD, 0.70 on VitalVideo, 0.25 on PURE. — [arXiv 2504.01774 html](https://arxiv.org/html/2504.01774v1)
- MMPD cross-dataset MAE (BPM) from the ME-rPPG paper, trained on RLAP / trained on PURE:
  - EfficientPhys: 9.18 / 17.60
  - TS-CAN: 10.19 / 18.10
  - PhysNet: 11.20 / 15.10
  - PhysFormer: 12.80 / 13.00
  - RhythmMamba: 9.62 / 10.90
  - ME-rPPG: 5.38 / 10.50

  POS and CHROM are not in that table. — [arXiv 2504.01774](https://arxiv.org/html/2504.01774v1)
- The ME-rPPG demo repo is Apache-2.0 and ships `model.onnx`, `get_hr.onnx`, `welch_psd.onnx` and a BlazeFace short-range `.tflite` detector. It is a browser (onnxruntime-web) demo, and the input size is not documented on the page. — [GitHub Health-HCI-Group/ME-rPPG-demo](https://github.com/Health-HCI-Group/ME-rPPG-demo)
- RhythmFormer (2024): MMPD MAE 8.98 trained on PURE, 9.08 trained on UBFC. — [arXiv 2402.12788](https://arxiv.org/pdf/2402.12788)
- PhysMamba (SlowFast TD-Mamba, 2409.12031): MMPD MAE 10.31 trained on PURE, and the paper notes MMPD is much harder. A different "PhysMamba" (2408.01077, SSD) reports 5.32 trained on MMPD and tested on PURE, which is the opposite direction and not comparable. — [arXiv 2409.12031](https://arxiv.org/pdf/2409.12031); [arXiv 2408.01077](https://arxiv.org/pdf/2408.01077)
- FactorizePhys (NeurIPS 2024, FSAM attention): cross-dataset HR MAE of 1.66 (PURE to iBVP), 0.48 (UBFC to PURE) and 1.04 (PURE or iBVP to UBFC). No MMPD numbers. Latency was measured on an RTX 3070 laptop GPU. The repo has a `final_model_release` folder and inference configs. ONNX is not mentioned, and the licence file exists but its type was not read. — [GitHub PhysiologicAILab/FactorizePhys](https://github.com/PhysiologicAILab/FactorizePhys); [NeurIPS slides](https://neurips.cc/media/neurips-2024/Slides/93470_NmYSJaO.pdf)
- BeatFormer (ICCVW 2025) is an efficient, motion-robust method trained without supervision using "spectral zoomed attention filters". It is a further candidate, but its weights and licence were not checked. — [arXiv 2507.14885](https://arxiv.org/pdf/2507.14885)

### Inferences
- **Which to try first:** ME-rPPG. It has ONNX files, streams frame by frame (no 128- or 160-frame clip buffering, unlike PhysNet and PhysFormer), runs on a laptop CPU, and the Apache-2.0 licence is commercial-friendly. Load `model.onnx` in `onnxruntime` (CPU EP) in Python and inspect its input shape with `sess.get_inputs()`. The face crop comes from BlazeFace in the demo; substituting our YuNet crop may cause a domain shift, so match the crop scale.
- **Can it beat tuned POS?** On a still, frontal, well-lit subject, POS on good ROIs typically gets about 1-2 BPM MAE on UBFC-like data (see the previous notes and Kim 2021's 1.85). Deep models win mainly under motion, compression and lighting variation (MMPD). Our case is still but very dark, which is out of distribution for most training sets. Expect parity at best unless the low light favours the learned model. Decide empirically by recording 5 × 2 min clips with a contact-PPG or chest-strap reference and comparing MAE and SNR.
- The older 3D-CNN and transformer models (PhysNet, PhysFormer) need 128-160 frame clips at 128x128. That is CPU-feasible but laggy, and their MMPD cross-dataset results are worse than POS-level expectations, so they are not worth the integration effort.

### Gaps
- Licences of rPPG-Toolbox weights, FactorizePhys, Contrast-Phys and SiNC were not verified in this session. Check each LICENSE before use.
- No POS/CHROM numbers were found under the same MMPD protocol in the ME-rPPG paper, so a direct "deep vs POS on MMPD" comparison from one source is missing.
- No 2026 rPPG model with public weights was surfaced within the time budget.
- ME-rPPG's input resolution and the details of its preprocessing are undocumented on the repo page.

---

## Q4. Waveform quality for HRV: peaks, IBI filtering, SQI

### Takeaway
At 30 fps, webcam rPPG HRV has RMSSD errors of roughly 15-110 ms across studies. That is usable for coarse trends only, and only when signal quality is gated. Template-matching SQI (beat correlation with an average template, r > 0.8) and skewness (Elgendi's best single PPG SQI) are the standard quality gates.

### Cited Findings
- **IBI filtering in a webcam HRV pipeline (Gudi et al.):**
  - IBIs outside 250-2000 ms are dropped.
  - For RMSSD, only IBIs within 1 SD of the mean are used, because RMSSD is very noise-sensitive.

  — [arXiv 2012.15846](https://arxiv.org/pdf/2012.15846)
- A 30 fps facial-video study against 1000 Hz ECG (29 participants, driving simulator) reported a best RMSSD MAE of 0.081 s (RMSE 0.111 s). Beats were scored with 250 and 350 ms tolerance windows. This comes from a secondary summary. — [Moonlight review](https://www.themoonlight.io/en/review/a-signal-extraction-approach-for-remote-heart-rate-variability-assessment-using-proxy-measure-in-a-driving-simulator)
- An IST MSc thesis (webcam vs finger PPG) found about 39 ms RMSSD error and 20 ms SDNN error, judged not accurate enough for medical use. — [IST thesis](https://fenix.tecnico.ulisboa.pt/downloadFile/1407770020547324/IST_UL_MSc_Thesis___Real_time_pulse_rate_variability_via_remote_photoplethysmography%20(1).pdf)
- A vendor white paper (Noldus FaceReader 10) claims 14.6 ms RMSSD MAE and argues that about 30 ms or less is acceptable given typical RMSSD of 19-75 ms. This is vendor-run. — [Noldus white paper](https://noldus.com/download/document/facereader-10-vital-signs-white-paper)
- Peak-based HRV performs poorly on low-SNR iPPG. Pulse-frequency demodulation is proposed instead, cutting camera-HRV error by more than 50% for dark skin and face motion. — [NSF PAR 10319127](https://par.nsf.gov/servlets/purl/10319127)
- **Skewness as a PPG quality index:** Elgendi (2016) found skewness the best of 8 PPG SQIs at separating excellent, acceptable and unfit recordings. — [Elgendi 2016, PMC5597264](https://pmc.ncbi.nlm.nih.gov/articles/PMC5597264/)
- **Template-matching SQI:** build a per-beat template by averaging beats that correlate with a template at r > 0.8, then score each beat by its correlation with the template (after Warren et al.). — [arXiv 2412.17538](https://arxiv.org/pdf/2412.17538)
- Samsung applied six SQIs to rPPG (perfusion, kurtosis, skewness, zero-crossing, entropy, SNR) to improve HR estimation; its thresholds were not retrieved. — [Samsung Research](https://sra.samsung.com/publications/enhancement-of-remote-ppg-and-heart-rate-estimation-with-optimal-signal-quality-index)

### Inferences (recipe)
1. Band-pass the fused POS pulse with a zero-phase 4th-order Butterworth (`filtfilt`) over HR_est ± 0.5 Hz, which is tighter than the 0.7-3 Hz used for HR.
2. Upsample to 250 Hz with cubic spline (`scipy.interpolate.CubicSpline`), or refine each peak by parabolic interpolation over the 3 samples around it. Without this, timing quantises at 33 ms, which is the same size as the errors reported above.
3. Detect peaks with `scipy.signal.find_peaks` using `distance = 0.6 · 60/HR_est` seconds and prominence of at least 0.3 × the rolling MAD.
4. Reject IBIs outside 300-1500 ms, and any IBI that differs by more than 20% from the median of the surrounding 5 (a Malik-style rule). Interpolate gaps but do not count them. If more than 20% are rejected, report no HRV.
5. **SQI gate (rolling 10 s):**
   - Spectral SNR at or above the previous notes' HR threshold.
   - Template correlation: the median beat r against the template is at least 0.8, and at least 70% of beats have r > 0.8.
   - Skewness: inverted-pulse orientation is a common pitfall, so flip the sign so systolic peaks are positive. The skewness threshold has to be learned on our own reference recordings.
6. Report RMSSD and SDNN only from windows of 60 s or more that pass the gate. Present them as trends, not clinical values.

### Gaps
- No rPPG-specific numeric thresholds for skewness or template SQI were found; they must be calibrated locally.
- The full numeric IBI and RMSSD results of Gudi et al. 2020 were not retrieved.

---

## Q5. Commercial and clinical webcam products: reported accuracy and conditions

### Takeaway
Vendors report heart-rate MAE of about 1-3 BPM, but only under controlled conditions: at least 400 lux on the face, a still seated subject, no talking, a face 20-40 cm away filling the frame, and about 60 s windows. Independent head-to-head data is scarce and partly vendor-hosted.

### Cited Findings
- A head-to-head comparison (hosted by Shen.AI, so effectively vendor-published) reports HR MAE of 2.1 BPM for Binah.ai (SD 2.8), 2.7 for Nuralogix (SD 3.1) and 1.9 for Shen.AI (SD 2.7). It notes that Nuralogix issues occurred under 2500 lux natural light with facial light reflexes. — [Shen.AI comparison PDF](https://health.shen.ai/hubfs/Comparison%20of%20competitive%20Face-Scanning%20Technologies%20in%20terms%20of%20Blood%20Pressure%20and%20Heart%20Rate.pdf); [ACHA summary](https://medium.com/@johni.fisher/acha-americas-continental-health-alliance-08182d9829b7)
- Binah.ai's own validation reports show HR MAE well under 1 BPM on iOS and Android against Nellcor and Polar H10 references. These are vendor-run. — [Binah HR report](https://www.binah.ai/wp-content/uploads/2022/06/Heart-Rate-validation-report.pdf); [Binah PR report SDK 5.11](https://www.binah.ai/wp-content/uploads/2025/09/MED-000051-Pulse-Rate-Accuracy-Report-of-SDK-5.11_Rev1.pdf); [Binah HRV report SDK 5.11](https://www.binah.ai/wp-content/uploads/2025/09/MED-000052-HRV-Accuracy-Reprort-of-SDK-5.11-Rev2.pdf)
- **Binah best-practice conditions:**
  - At least 400 lux on the face, cool light above 4500 K, even illumination, no nearby reflective surfaces.
  - Sit still with feet flat, no talking, face unobstructed.
  - The SDK raises a "face too far" flag.
  - Validation sessions were about 60 s at 20-40 cm.
  - HR appears in about 10 s, other vitals in 35-60 s.

  — [Binah best practices](https://support.binah.ai/en-US/binah/article/5uQeVfY3-best-practices); [Binah image validity docs](https://developer.binah.ai/ios/latest/images/image-validity.html)
- A Frontiers 2025 pilot evaluated Shen.AI Vitals heart rate against a reference; its MAE was not retrieved. — [Frontiers Physiol 2025](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1638809/pdf)

### Inferences
- Our skin luma of about 53/255 with gain at 0 suggests room light well below 400 lux on the face. The cheapest accuracy gain in production is therefore operational: add a desk lamp or monitor-brightness prompt, and a "light too low" gate that uses skin luma as a proxy (for example, warn below 70/255 at our fixed exposure; calibrate this).
- Copy the vendor UX pattern: show HR after 10-15 s, show HRV only after 60 s of gated signal, and show image-validity flags (too dark, too far, moving, talking) instead of bad numbers.

### Gaps
- No independent peer-reviewed accuracy numbers were found for Nuralogix, Presage or Philips within the time budget. Philips' Vital Signs Camera is discontinued as a consumer product (from memory; not verified this session).
- The head-to-head comparison's methodology (subjects, lux, skin tones) was not fully retrieved.
