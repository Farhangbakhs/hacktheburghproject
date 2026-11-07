# rPPG heart-rate estimation from a 30 fps laptop webcam (Python/OpenCV/NumPy, CPU real time)

Recommended pipeline in short. The sections below give the evidence for each step.
1. Capture at 30 fps and record a `time.perf_counter()` timestamp for every frame.
2. Use MediaPipe Face Landmarker (Tasks API, `mediapipe` 1.2.2, VIDEO or LIVE_STREAM mode) to get forehead and both cheek polygons. Average the R, G, B of skin pixels in each polygon. If MediaPipe fails to install, fall back to YuNet or Haar plus a fixed-fraction forehead/cheek box.
3. Linearly resample the RGB trace to a uniform 30 Hz grid using the timestamps.
4. Run POS with a 1.6 s window (48 frames) and overlap-add.
5. Detrend (smoothness priors, lambda=100) or simply high-pass, then apply a Butterworth bandpass of 0.7–3.0 Hz (0.75–2.5 Hz in rPPG-Toolbox), order 1–3, run causally or with filtfilt on the buffer.
6. Use a 10 s sliding window (300 samples) updated every 0.5–1 s. Take a Hann-windowed FFT zero-padded to 2048 or more points (or Welch), find the peak in band, and refine it with parabolic interpolation.
7. Compute SNR (de Haan style, in dB, ±6 bpm around f0 and 2·f0, against the rest of the band) as the confidence score. Show "no reading" below about 0 dB, or when the face is lost or motion is high.
8. Smooth the output BPM with a median or EMA filter, and reject jumps above about 10–15 bpm per update unless they persist.

Expected accuracy: about 4 bpm MAE on the clean UBFC-rPPG and PURE datasets. Expect about 8–12 bpm in realistic mobile or webcam conditions (MMPD), and worse with walking, natural light or darker skin.

## Face detection/tracking: OpenCV Haar/YuNet vs MediaPipe Face Landmarker; Python/Windows status; ROI landmark indices

### Takeaway
MediaPipe ships Windows x64 wheels for Python 3.9–3.14 as of Nov 2026, and the current version is 1.2.2. Use the Tasks API `FaceLandmarker` because the legacy `mp.solutions.face_mesh` API has been removed or deprecated in newer releases. For ROIs, pyVHR's landmark sets are a ready-made reference: forehead `[10, 67, 69, 104, 108, 109, 151, 299, 337, 338]` and the lower cheek sets.

### Cited Findings
- PyPI `mediapipe` latest is 1.2.2, released 3 Nov 2026. Classifiers list Python 3.9–3.14. Wheels: `mediapipe-1.2.2-py3-none-win_amd64.whl` (25.3 MB) and win_arm64, linux and macOS arm64. There is no sdist. — [PyPI mediapipe](https://pypi.org/project/mediapipe/)
- Windows support for Python 3.11 was added in MediaPipe 0.10.1. — [GitHub issue #4292](https://github.com/google-ai-edge/mediapipe/issues/4292)
- Users hit `AttributeError: module 'mediapipe' has no attribute 'solutions'` on newer versions, and the fix is migrating to `mediapipe.tasks.python.vision`. Sources disagree on which version removed `solutions` (0.10.0 vs 0.11+), and neither is official. — [GitHub issue #6204](https://github.com/google-ai-edge/mediapipe/issues/6204); [theneuralbase (secondary)](https://theneuralbase.com/mediapipe/learn/beginner/import-errors-version-mismatch/)
- The face mesh has 468 landmarks, or 478 with iris refinement. Indices below 468 are shared. — [hackernoon cheat sheet](https://hackernoon.com/mediapipe-face-mesh-landmark-indices-cheat-sheet)
- pyVHR `MagicLandmarks`:
  - `high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]`
  - `forehead_center = [10, 151, 9, 8, 107, 336, 285, 55, 8]`
  - `cheek_left_bottom = [215, 138, 135, 210, 212, 57, 216, 207, 192]`
  - Further lists: `cheek_right_bottom`, `high_prio_left_cheek`/`right_cheek`, `cheek_left_top`/`right_top`
  - "High priority" ROIs are forehead_center plus the left and right lower cheeks
  - Motion-tracking landmarks are `[10, 151, 6, 197, 5, 4]`

  — [pyVHR extraction/utils.py](https://raw.githubusercontent.com/phuselab/pyVHR/master/pyVHR/extraction/utils.py)
- A Durham study found MediaPipe Face Mesh with cheek ROIs gave the best HR measurements. — [Durham repository](https://durham-repository.worktribe.com/OutputFile/1748767)
- Other work uses 60 landmarks with 20×20 px patches (19 forehead, 18 per cheek, 5 chin), or a 40×40 px forehead-first ROI that switches to a cheek when the forehead is occluded. — [Wong CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/CVPM/papers/Wong_Optimising_rPPG_Signal_Extraction_by_Exploiting_Facial_Surface_Orientation_CVPRW_2022_paper.pdf); [MDPI Sensors 21/23/7923](https://mdpi.com/1424-8220/21/23/7923)
- rPPG-Toolbox reproduces its paper results with a Haar Cascade face box, which is the default. It offers an optional YOLO5Face backend, re-detection every N frames, a box enlargement coefficient and a median box. — [rPPG-Toolbox README](https://github.com/ubicomplab/rPPG-Toolbox)

### Inferences
- A Haar face box with a median or EMA-smoothed box is good enough to reach the published CHROM/POS numbers on UBFC and PURE. MediaPipe mainly helps by excluding hair, eyes, mouth and background, and by tracking under head rotation.
- Practical fallback: crop the Haar/YuNet box to a forehead band (x 25–75%, y 5–25%) and cheeks (x 15–40% and 60–85%, y 45–70%). These fractions are a heuristic I have not verified.
- On a hackathon CPU, run the Face Landmarker every frame in VIDEO mode, which is typically real time on a laptop. Alternatively, detect every 2–3 frames and reuse the polygon. Fill the polygon mask with `cv2.fillConvexPoly` and average the masked pixels.

### Gaps
- I did not confirm Face Landmarker CPU fps on Windows or whether the 1.2.x API differs from 0.10.x Tasks code samples. Verify by `pip install mediapipe` on the target machine. The `face_landmarker.task` model file must be downloaded separately.
- I did not benchmark YuNet (`cv2.FaceDetectorYN`, OpenCV ≥4.5.4).

## Signal methods: GREEN, CHROM, POS formulas, windows, best real-time unsupervised choice

### Takeaway
Use POS: it is the best or joint-best unsupervised method in rPPG-Toolbox across datasets. Use a 1.6 s window, temporal normalisation, and the projection P=[[0,1,-1],[-2,1,1]] with alpha-tuning and overlap-add. CHROM is nearly as good. GREEN is much worse (MAE 10–22 bpm).

### Cited Findings
- **POS (rPPG-Toolbox implementation, after Wang et al. 2017 TBME 64(7):1479–1491).** `WinSec = 1.6` and `l = ceil(1.6·fs)`. For each n ≥ l, take the window m=n−l..n:
  - `Cn = RGB[m:n]/mean(RGB[m:n])`
  - `S = [[0,1,-1],[-2,1,1]] · Cn`
  - `h = S0 + (std(S0)/std(S1))·S1`
  - `h -= mean(h)`
  - `H[m:n] += h` (overlap-add)
  - Then detrend with λ=100 and apply a 1st-order Butterworth bandpass of 0.75–3 Hz with filtfilt.

  — [rPPG-Toolbox POS_WANG.py](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/unsupervised_methods/methods/POS_WANG.py)
- **CHROM (implementation of de Haan & Jeanne 2013, TBME 60(10):2878–2886).** Use a 1.6 s window, rounded to an even number of frames, with 50% overlap. In each window:
  - Normalise RGB by the window mean.
  - `Xs = 3R − 2G` and `Ys = 1.5R + G − 1.5B`
  - Bandpass both with a 3rd-order Butterworth, 0.7–2.5 Hz.
  - `α = std(Xf)/std(Yf)` and `S = Xf − α·Yf`
  - Multiply by a Hann window and overlap-add.

  — [rPPG-Toolbox CHROME_DEHAAN.py](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/unsupervised_methods/methods/CHROME_DEHAAN.py)
- MAE in bpm from rPPG-Toolbox Table 2 (POS / CHROM / GREEN / ICA):

  | Dataset | POS | CHROM | GREEN | ICA |
  |---|---|---|---|---|
  | UBFC-rPPG | 4.00 | 3.98 | 19.81 | 14.70 |
  | PURE | 3.67 | 5.77 | 10.09 | 4.77 |
  | UBFC-Phys | 4.51 | 4.49 | 13.55 | 10.03 |
  | MMPD | 12.36 | 13.66 | 21.68 | 18.60 |

  — [rPPG-Toolbox paper (NeurIPS 2023), arXiv 2210.00716](https://arxiv.org/html/2210.00716v3)
- An independent study found POS and CHROM the most robust methods for pulse-rate estimation. — [PMC9604655](https://pmc.ncbi.nlm.nih.gov/articles/PMC9604655)

### Inferences
- **Real-time POS.** Keep a ring buffer of RGB means. Each new frame lets you compute one 48-sample window at 30 fps and overlap-add it into the output buffer. This costs O(48) per frame and is trivial on CPU.
- **Multi-ROI.** Compute POS per ROI (forehead and both cheeks), then either average the RGB first or pick or weight the ROIs by per-ROI SNR.

### Gaps
- I did not fetch the original POS paper text. The formula above comes from the reference implementation, which matches its published description.

## Preprocessing: detrending lambda, normalisation, Butterworth order and band

### Takeaway
rPPG-Toolbox uses Tarvainen smoothness-priors detrending with λ=100, then a Butterworth bandpass. Its evaluation filter is 2nd order, 0.75–2.5 Hz (45–150 bpm). Its code default is 1st order, 0.6–3.3 Hz. The authors note that filter choice "can substantially affect heart rate errors".

### Cited Findings
- **Detrend.** Build the second-difference matrix D and compute `(I − inv(I + λ²·DᵀD))·x` with λ=100. — [rPPG-Toolbox evaluation/post_process.py](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/evaluation/post_process.py)
- **Filter.** The post-process default is `butter(1, [0.6, 3.3] Hz)`. The source comments recommend low=0.75, high=2.5 to match the NeurIPS paper. — [post_process.py](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/evaluation/post_process.py)
- The paper states it uses a 2nd-order Butterworth at 0.75–2.5 Hz for evaluation, and that filter choice substantially affects errors. — [arXiv 2210.00716](https://arxiv.org/html/2210.00716v3)
- POS normalises by the temporal mean within each 1.6 s window, and CHROM does the same. — code links above

### Inferences
- The 0.7–3.0 Hz band (42–180 bpm) is fine for seated users. For a calm bluffing game, 0.75–2.5 Hz (45–150 bpm) reduces harmonic and noise picks; consider 0.7–3.0 only if exercise is expected.
- Note that λ is squared in that code, so λ=100 is applied as 10⁴. Inverting a 300×300 matrix per update is fine. Alternatively, precompute the detrend operator once for a fixed window length.
- For causal live use, apply `sosfilt` with a persistent state (`sosfilt_zi`), or run `filtfilt` on the whole 10 s buffer each update. The latter is cheap at 300 samples and avoids phase lag. Order 2–3 as SOS is stable.

### Gaps
- I did not verify the exact cutoff frequency that corresponds to λ=100 at 30 fps.

## HR estimation: FFT vs Welch, window, zero-padding, tracking, harmonics

### Takeaway
rPPG-Toolbox estimates HR by zero-padding to the next power of 2, taking a periodogram, and choosing the in-band argmax × 60. Peak detection on inter-beat intervals is the alternative. For live use, take a 10 s window, zero-pad for resolution, refine with parabolic interpolation, and smooth or track over time.

### Cited Findings
- `_calculate_fft_hr` zero-pads to the next power of 2, takes a periodogram, picks the maximum-power frequency in band and returns ×60. `_calculate_peak_hr` uses `scipy.signal.find_peaks` and returns 60/mean(IBI). — [post_process.py](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/evaluation/post_process.py)
- The paper computes HR by FFT or by peak detection, depending on configuration. — [arXiv 2210.00716](https://arxiv.org/html/2210.00716v3)

### Inferences
- **Resolution.** A 10 s window has a native bin of 0.1 Hz (6 bpm). Zero-padding to 2048 points at 30 Hz gives about 0.88 bpm bins, and parabolic interpolation on the log power gets below 1 bpm. Windows of 6–8 s respond faster but are noisier. Use 10–15 s for stability.
- **Welch.** Use `nperseg` of about 256 (8.5 s) with 50% overlap on a 15–20 s buffer to reduce variance, at the cost of latency. A plain Hann-windowed FFT on 10 s is the simpler, standard choice.
- **Harmonic check.** If the power at f/2 is in band and at least about 50% of the peak power, prefer f/2, since the 2nd harmonic often dominates. These heuristic thresholds are not from a source.
- **Tracking.** Restrict the search to ±15 bpm of the previous estimate when SNR is good. Use a median of the last 5 estimates or an EMA (α≈0.3). Reset tracking when SNR stays low for more than 3 s.

### Gaps
- I found no source that compares FFT and Welch head-to-head for webcam rPPG, so that recommendation is engineering judgment.

## Signal quality: SNR definition and "no reading" thresholds

### Takeaway
Use de Haan's SNR: power within a template around the fundamental and 1st harmonic, divided by the remaining power in the band, in dB. rPPG-Toolbox uses ±6 bpm around f0 and 2·f0, with the band 0.6–3.3 Hz or 0.75–2.5 Hz. Live, use the estimated HR as f0. Report "no reading" below about 0 dB, a heuristic threshold.

### Cited Findings
- **rPPG-Toolbox `_calculate_SNR`.** Locate the 1st and 2nd harmonics of the HR. Sum the periodogram power within ±6 bpm (`deviation = 6/60` Hz) of each, divide by the remaining in-band power, and convert to dB. Return 0 if the remainder is zero. — [post_process.py](https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/main/evaluation/post_process.py)
- The SNR "by [de Haan]" is computed over 30–240 bpm as in-band energy in a binary template window divided by the energy outside it, in dB. — [Wong et al. CVPRW 2022 (secondary)](https://openaccess.thecvf.com/content/CVPR2022W/CVPM/papers/Wong_Optimising_rPPG_Signal_Extraction_by_Exploiting_Facial_Surface_Orientation_CVPRW_2022_paper.pdf)
- SNR gains of about 1.5–2.2 dB are reported as meaningful improvements in this literature (2SR vs CHROM, ICA and PBV). — [Wang et al. (secondary summary)](https://typeset.io/pdf/a-novel-algorithm-for-remote-photoplethysmography-spatial-420d2ru6ge.pdf)

### Inferences
- **Live confidence score.** Compute `snr_db` with the template centred on the detected peak. Map it with `conf = clip((snr_db + 3)/9, 0, 1)`, so −3 dB maps to 0 and +6 dB maps to 1.
- **Show "no reading" when any of these hold:**
  - SNR is below 0 dB for more than 2 s
  - The face is missing
  - Landmark motion exceeds a threshold
  - The ROI has under ~500 skin pixels
  - The buffer holds less than 6–8 s of data

  These thresholds are heuristics to tune on the real camera.
- **Second quality cue.** Use the ratio of peak power to total band power (spectral peakiness), or the agreement between forehead and cheek HR within 5 bpm.

### Gaps
- I did not retrieve the original de Haan 2013 template widths. I recall a template around f0 and the 1st harmonic, but did not verify it. I found no published absolute SNR threshold for "no reading".

## Expected accuracy; effect of motion, lighting, skin tone

### Takeaway
Unsupervised POS reaches about 4 bpm MAE on lab datasets (UBFC-rPPG, PURE) but about 12 bpm on MMPD. MMPD is mobile video compressed to 320×240 and covers Fitzpatrick 3–6. Errors are lowest for skin type 3 (POS 5.8 bpm), rise to 11–13 bpm for types 5–6, and reach about 17 bpm when walking.

### Cited Findings
- Benchmark numbers are in the signal-methods table above (POS 4.00 UBFC, 3.67 PURE, 12.36 MMPD). — [arXiv 2210.00716](https://arxiv.org/html/2210.00716v3)
- MMPD POS MAE in bpm, from Table II:

  | Condition | POS MAE |
  |---|---|
  | Skin tone 3 | 5.76 |
  | Skin tone 4 | 9.06 |
  | Skin tone 5 | 12.78 |
  | Skin tone 6 | 11.17 |
  | Stationary | 9.70 |
  | Rotation | 7.50 |
  | Talking | 8.05 |
  | Walking | 17.05 |
  | LED-low | 9.76 |
  | LED-high | 7.26 |
  | Incandescent | 8.24 |
  | Natural light | 10.71 |

  CHROM is 1–2 bpm worse in most conditions and GREEN is 12–30 bpm. MMPD videos are 30 fps, 1280×720, stored compressed at 320×240. — [MMPD paper arXiv 2302.03840](https://ar5iv.labs.arxiv.org/html/2302.03840)
- UBFC-rPPG mostly contains stationary subjects of Fitzpatrick 2–3. MMPD was built to span types 3–6. — [MMPD paper](https://ubicomplab.cs.washington.edu/pdfs/mmpd.pdf)
- Traditional POS, CHROM and ICA are reported to be far worse than deep models on darker skin (F5–6). — [arXiv 2404.05003](https://arxiv.org/pdf/2404.05003v1.pdf)
- RhythmNet reports CHROM MAE of 6.23, 11.37 and 16.9 bpm across three webcam-style datasets. — [RhythmNet arXiv 1910.11515](https://arxiv.org/pdf/1910.11515)

### Inferences
- For a seated player under decent indoor LED light with an uncompressed laptop webcam feed, expect an MAE of about 3–8 bpm once the window is full. Expect 10–15 bpm or more with talking, head turns, dim or natural light, or darker skin, so the confidence gate matters.
- The demo should present BPM as a trend or relative change, such as "elevated vs your baseline", not as a clinical number.

### Gaps
- I found no laptop-webcam-specific MAE study with per-skin-tone figures. The Freyja/IBV 73-participant study and the ASUS laptop clinical study were seen only as snippets.

## Pitfalls: fps jitter, compression, 50 Hz flicker

### Takeaway
Timestamp every frame and resample to a uniform grid. Avoid compressed or auto-exposure-heavy paths where possible. Mains flicker (100 Hz intensity at 50 Hz mains in the UK) beats with the frame rate, and the resulting alias can fall in the HR band.

### Cited Findings
- Flicker comes from the beat between the camera's frame rate and the light-source frequency. That beat can lie near HR frequencies, and camera anti-banding can suppress the rPPG signal. — [Mironenko et al. CVPRW 2020](https://ar5iv.labs.arxiv.org/html/2004.12695)
- A patent notes that video sampled at about 24 Hz cannot avoid aliasing 50 Hz lighting, which would need at least 100 Hz. Aliases often appear near 2–4 Hz and are hard to predict, so fixed notch filters don't work well. — [US patent 12039764](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12039764)
- **Frame jitter, conflicting evidence.** Mironenko et al. found no significant spectral distortion from irregular frame rates on the smartphones they tested. Other work argues irregular frame rates (as in VIPL-HR and PURE) matter for accurate HR and proposes timing correction. — [Mironenko CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Mironenko_Remote_Photoplethysmography_Rarely_Considered_Factors_CVPRW_2020_paper.pdf); [arXiv 2405.01230](https://arxiv.org/html/2405.01230v1)
- MMPD's heavy compression (720p down to 320×240) coincides with its much higher MAE than UBFC or PURE. — [MMPD paper](https://ar5iv.labs.arxiv.org/html/2302.03840)

### Inferences
- **Windows capture.** Use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` or MSMF and set the resolution to 640×480. Measure the real fps from timestamps rather than trusting `CAP_PROP_FPS`, because laptop webcams drop to about 15 fps in low light due to auto-exposure. Lock exposure and white balance if the driver allows it (`CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_AUTO_WB`). Check this on the actual hardware.
- **Resampling.** Use `np.interp(t_uniform, t_frames, rgb[:, c])` at 30 Hz before POS and the FFT.
- **Flicker.** With 50 Hz mains, 100 Hz flicker at exactly 30 fps aliases to 100 − 3·30 = 10 Hz, which folds to 10 Hz and is outside the band. Small fps drift moves the alias, and at 25 fps or other rates it can land at DC or in band. Prefer daylight or LED lighting, and treat a sharp, very stable spectral line as suspicious.
- Run capture and processing in separate threads so face detection latency does not create jitter.

### Gaps
- I found no quantitative study of the 50 Hz alias effect on laptop webcams specifically.

## Open-source references: rPPG-Toolbox, pyVHR, yarppg — lessons

### Takeaway
Copy rPPG-Toolbox's POS, CHROM and post-processing code (MIT-style research code) as the reference. Take ROI landmark sets from pyVHR. yarppg is a good model for the structure of a live demo app.

### Cited Findings
- rPPG-Toolbox implements GREEN, ICA, CHROM, LGI, PBV, POS and OMIT, plus supervised models. Its configs cover face crop backend, dynamic detection frequency, a larger box and a median box. — [rPPG-Toolbox GitHub](https://github.com/ubicomplab/rPPG-Toolbox)
- pyVHR (2022) provides MediaPipe landmark ROI sets in a priority order (forehead centre and lower cheeks first). — [pyVHR utils.py](https://raw.githubusercontent.com/phuselab/pyVHR/master/pyVHR/extraction/utils.py)
- yarppg is a hobby project "for demo purposes only". Its October 2024 overhaul separated the rPPG core from the UI. A stateful `Rppg.process_frame(frame)` returns `.value` and `.hr`. It ships an OpenCV UI (`run-yarppg`) and an optional Qt6 UI. — [yarppg GitHub](https://github.com/SamProell/yarppg)

### Inferences
- Lessons:
  - Keep a stateful per-frame `process_frame` API, as in yarppg, that is decoupled from the UI.
  - Validate offline on a UBFC-rPPG video first. The expected POS MAE is about 4 bpm.
  - Keep the band and filter settings identical between offline tests and live use, because the toolbox shows filter choice materially changes MAE.

### Gaps
- I did not inspect yarppg's internals (its ROI method, processors and HR estimator), the pyVHR 2022 paper's benchmark tables, or the pyVHR install status on Python 3.11.
