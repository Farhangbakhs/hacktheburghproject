# Capture-side requirements and validation protocol for webcam facial pulse (rPPG / colour EVM)

Context: Windows 11 laptop, Python + OpenCV 4.14, built-in webcam over DirectShow, 640x480 YUY2, 30 fps. Exposure is settable (log2 s scale; -5 is the longest setting that holds 30 fps). Gain is stuck at 0. Brightness is a digital offset. Measured skin luma was 23-61/255 and pulse SNR 0-3.6 dB, with the pulse detectable but faint (74-81 bpm). This builds on `../Webcam pulse and bluff tells/evm_realtime.md` and does not repeat it: DSHOW property handling, the 100 Hz-to-10 Hz alias, spatial pooling, and motion gating are covered there.

"Computed" means I worked it out from first principles for these notes, with no external source. Research date: 2026-11-07. Some full texts were paywalled or blocked (the Odinaev CVPRW 2023 exposure paper returned 403, and the Gudi 2020 PDF was binary-only). These are listed as gaps.

## 1. Light: level, luma target, SNR scaling, spectrum, flicker, direction, screen as a light source

### Takeaway
Aim for at least 300 lux on the face, ideally about 500 lux, front-lit and diffuse. The one controlled study found RRI error was lowest at 500 lux or more with front lighting, worst at 100 lux, and images overexposed above about 700 lux on that camera. In 8-bit terms, aim for a mean face-ROI luma of about 100-170 (green channel ideally above 90) with less than 1% clipped pixels. Our measured 23-61 is roughly 2-5x too dark. Under shot noise, every doubling of light is worth about 3 dB of pulse SNR, so getting from luma 50 to 150 is worth about 4-5 dB. Freeze screen content and brightness during capture.

### Cited Findings
- Controlled study (3 subjects, chin rest, 0.5 m, uncompressed 720x540, ECG reference): tested 100/300/500/700 lux from front, top, or front+top. **Front lighting gave the lowest RRI MAE. Error was minimised at ≥500 lux, accuracy was worst at 100 lux front-lit, and the image was "overexposed when it exceeds 700 lux"** (the exposure was not re-optimised). — [Evaluation of rPPG Measurement Conditions toward Telemedicine, Sensors 2021, PMC8704576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704576/)
- Güler et al. 2022: in 3 of 7 participants, putting the camera at the same level as the light source lost up to 3 dB of signal quality in the 30-60 lux range. They used cross-polarised light to separate out specular reflection. — [EPFL Infoscience record 296647](https://infoscience.epfl.ch/record/296647?ln=fr); [Bournemouth eprint](https://eprints.bournemouth.ac.uk/37452/)
- Xi et al. built a 1-100 lux dataset and found that in low light, face detection fails and the falling pulsatile amplitude lets noise dominate. Low-light enhancement improved SNR. — [arXiv 2303.09336 summary](https://bohrium.dp.tech/paper/arxiv/2303.09336)
- Changing illuminance is its own artefact. Lee et al. (EMBC 2015) report average SNR of -11.7 dB (low dynamic brightness) and -15.3 dB (high dynamic brightness) before correction, versus -4.2 / +8.0 dB after. — [search.isc.ac record](https://search.isc.ac/Inventory/49/3684524.htm)
- "Constantly flickering illumination from a computer monitor... degrades the performance severely since such fluctuations can interfere with the pulsating component in the skin pixels". Bright frontal light can also hurt through pixel saturation (VIPL-HR). — [arXiv 2107.13193](https://arxiv.org/pdf/2107.13193)
- The BH-rPPG benchmark (35 subjects, low/medium/high light) found CHROM reached 1.04 bpm MAE under medium illumination. Accuracy depends on light level. — [arXiv 2107.13193](https://arxiv.org/pdf/2107.13193)
- Odinaev et al. (CVPRW 2023) varied exposure time and gain in low light and found "the optimal brightness of facial videos for remote vital sign measurement depends on the vital sign being measured". Manual gain/exposure "can significantly improve" rPPG quality. (Abstract only; the full text returned 403.) — [CVPRW 2023 page](https://openaccess.thecvf.com/content/CVPR2023W/CVPM/html/Odinaev_Optimizing_Camera_Exposure_Control_Settings_for_Remote_Vital_Sign_Measurements_CVPRW_2023_paper.html); [QUT: Dedicated exposure control for rPPG](https://eprints.qut.edu.au/202877)
- Vendor guidance (Noldus FaceReader vitals): diffuse light, indirect daylight preferred, diffused artificial light acceptable. Light from the front with no strong shadows, and no specular hot spots on the forehead. It gives no lux number. — [Noldus FaceReader 10 guidelines](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/facereader/page/facereader-10-guidelines-for-an-optimal-vital-signs-measurement)

### Inferences
- **Shot-noise scaling (computed):** the pulse is a fixed fractional modulation of the reflected light, so its amplitude scales with photon count N while shot noise scales with sqrt(N). Amplitude SNR therefore goes as sqrt(N), and power SNR rises about 3 dB per doubling of light at fixed exposure. Raising face luma from about 50 to about 150 (3x) by adding light gives about +4.8 dB. Doing it with digital "brightness" offset or gain gives about 0 dB: an offset adds no photons, and gain amplifies signal and noise together while adding quantisation headroom problems.
- **Quantisation (computed, approximate):** if the green-channel pulse is on the order of a few tenths of a percent of DC (a commonly quoted rPPG magnitude; I did not verify an exact figure this session, see Gaps), then at luma 50 the per-pixel pulse is about 0.1-0.2 LSB, and at luma 150 about 0.3-0.6 LSB. Spatial averaging over the ROI dithers this. At low luma, though, sensor read noise plus the camera's own denoiser dominate. This is another reason to target luma 100-170.
- **Target histogram for the face ROI:** mean luma (Y) 100-170, green-channel mean ≥90, p99 ≤245, and <1% of ROI pixels at ≥250 in any channel (allowing for specular forehead spots). Highlights clipping in the forehead will kill the pulse in those pixels, so exclude saturated pixels from the ROI mask rather than lowering exposure globally.
- **Metering problem:** whole-scene auto exposure with ceiling lights behind the subject underexposes the face. That is exactly our 53/255. The fix is manual exposure at -5 (31 ms) plus more light on the face. Do not use auto exposure during capture: its hunting falls in the 0.1-1 Hz band (see evm_realtime.md section 3).
- **Spectrum:** green (~530-570 nm) carries the strongest skin PPG, so the light needs good green content. Daylight and good LED (high CRI) are fine. Old fluorescent tubes have spiky spectra and flicker at 100 Hz. Whatever the source, keep it constant: do not mix daylight from a window, which drifts with clouds, with artificial light if avoidable.
- **UK 100 Hz flicker (computed):** the alias of 100 Hz at fs = 30 is 10 Hz. At 22 fps it is 10 Hz, at 18 fps 8 Hz, at 15 fps 5 Hz. All are outside the HR band (0.7-3.5 Hz) but inside a broad IIR's skirt. The residual ripple from a square exposure window T on a 100 Hz sinusoidal flicker of depth m is m·|sin(π·100·T)|/(π·100·T). That gives **0.039m at T = 31.25 ms (-5)**, 0.036m at 62.5 ms, 0.20m at 15.6 ms (-6), and 0 at exactly 10/20/30/40 ms. **Exposure -5 is therefore already a good anti-flicker setting (about 96% suppression). Avoid -6 and -7.** Also set the driver's "Powerline frequency = 50 Hz" if the DSHOW dialog offers it.
- **Flicker detector:** take the mean of a static non-skin patch (wall, or the whole background minus the face box) per frame. Compute an FFT over 10 s, and flag if the peak at the expected alias (10 Hz at 30 fps, 5 Hz at 15 fps) is more than 6 dB above the median 4-14 Hz floor, or if the wall's band-limited (4-14 Hz) RMS exceeds about 0.3 LSB. Rolling-shutter flicker also shows as horizontal bands, so a second check is the variance of row means within the wall patch.
- **Laptop screen:** the screen is a large, close, front light source. Any change in screen content or auto brightness modulates the face colour, and in rapid UI animation it lands right in band. **Freeze screen content during capture.** Use a static UI with no video and no animated waveform, or put the live preview in a small window. Disable Windows adaptive brightness and content-adaptive brightness control (CABC). A full-screen static white or warm-white page at maximum brightness is a plausible, free front fill light: it is diffuse, frontal, and DC on most laptop LCD/OLED panels at full brightness. PWM dimming is used at lower brightness on many panels, so use 100% brightness. I found no study quantifying a screen as an rPPG light source (see Gaps), so measure it: compare face luma and SNR with screen white versus normal.

### Gaps
- I found no source giving a published lux-to-SNR curve in dB. The Sensors 2021 per-condition MAE values exist only in a figure. The Odinaev full text (optimal pixel intensity numbers) was not accessible.
- I did not get a verified figure for the AC/DC pulse ratio of face skin in green, so the quantisation numbers above are order-of-magnitude.
- There is no study of a laptop screen as a deliberate rPPG fill light, and no measurements of screen-induced face illuminance (lux at 50 cm).

## 2. Camera: bit depth, gain vs exposure, YUY2 vs MJPEG, resolution, rolling shutter, external webcams, 15 fps trade-off

### Takeaway
Keep the built-in camera's uncompressed 640x480 YUY2 at 30 fps with exposure -5 locked. That is the right format: uncompressed beats any codec for rPPG. Fix the problem with light, not camera settings. Dropping to 15 fps for a longer exposure is acceptable for mean HR (Nyquist 7.5 Hz is well above HR harmonics) but measurably worse for beat-to-beat/HRV timing. On this driver, -4 gives 22 fps, not 16, so the driver's exposure values are not true log2 seconds. If you buy one camera, a Logitech C920/C922 with manual exposure and white balance is the standard dataset camera (UBFC-rPPG). Avoid the Logitech Brio and Microsoft LifeCam per one vendor's guidance.

### Cited Findings
- Compression noise "can be a major source of errors for rPPG", and the cleanest signal is uncompressed. FFV1 (lossless) showed zero accuracy drop at about 10x lower bitrate. H.265 was the most efficient lossy codec. rPPG trained on lossless data "effectively fails" on H.264/H.265 at CRF ≥22. — [Gudi et al., arXiv 2012.15846](https://arxiv.org/pdf/2012.15846) (via search snippets; full PDF not text-readable); [arXiv 2405.02652](https://arxiv.org/pdf/2405.02652)
- Another five-codec study found MJPG "provided a reasonable balance between data compression and preservation of rPPG signals". — [EAH Jena VitaWave: resolution-aware compression for rPPG](https://www.eah-jena.de/vitawave/publications/videocompressionrppg)
- UBFC-rPPG was recorded with a **Logitech C920 HD Pro at 640x480, 30 fps, uncompressed 8-bit RGB** through a custom C++ acquisition app. Specific APIs, not default settings, are needed to get lossless frames from USB webcams. — [UBFC-rPPG site](https://sites.google.com/view/ybenezeth/ubfcrppg); [arXiv 2404.05003](https://arxiv.org/html/2404.05003v1)
- Noldus: ≥15 fps, preferably 30, analyse every frame. Face ≥100x100 px. Least compression possible (HuffYUV, FFV1, MJPEG, or H.264/5 at high bitrate), with ≥0.15 bits/pixel and preferably 1 bit/pixel (640x480@30: ≥1.38 Mbit/s, preferably 9.2 Mbit/s). **Logitech Brio and Microsoft LifeCam are "unsuitable" due to internal video processing.** — [Noldus FaceReader 10 guidelines](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/facereader/page/facereader-10-guidelines-for-an-optimal-vital-signs-measurement)
- Frame rate (500 lux, 6 min, ECG): RRI error falls as fps rises (15/30/60/100 tested). LF/HF correlations are lower below 15 fps and change little above 30 fps, with HRV correlation >90% at ≥30 fps. The theoretical peak-timing error is about ±16.6 ms at 30 fps. **Recommended minimum is 30 fps for HRV.** — [Sensors 2021, PMC8704576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704576/)
- Frame-rate irregularity (jitter) did not significantly distort PPG spectra. One HRV dataset assumed 30 fps while the actual rate varied between 15 and 30, which is a known pitfall. — [Mironenko et al., CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Mironenko_Remote_Photoplethysmography_Rarely_Considered_Factors_CVPRW_2020_paper.pdf); [arXiv 2305.04161](https://arxiv.org/html/2305.04161v3)
- Image resolution and compression both affect rPPG accuracy. An ROI slightly smaller than the face boundary gave the best accuracy. — [Zhao et al., CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w19/Zhao_Remote_Estimation_of_Heart_Rate_Based_on_Multi-Scale_Facial_ROIs_CVPRW_2020_paper.pdf)
- There is no direct bit-depth comparison. HiDURPA stores 10-bit YUV 4:2:2, but no 8- vs 10-bit result was found. — (search result, [arXiv 2405.02652](https://arxiv.org/pdf/2405.02652) context)

### Inferences
- **YUY2 vs MJPEG:** YUY2 is uncompressed luma at full resolution with half-horizontal chroma. Since rPPG pools the ROI spatially, 4:2:2 chroma subsampling is irrelevant. MJPEG only helps when you need higher resolution or fps than USB 2 bandwidth allows. Its 8x8 block quantisation can erase sub-LSB temporal changes at low quality settings. **Stay on YUY2 at 640x480.**
- **Resolution vs noise:** at 640x480 a face at 0.5-0.7 m is about 150-250 px wide (see section 3), so the ROI already holds about 10-30k pixels. Higher resolution adds pixels but each is noisier (same sensor, more bandwidth, possibly MJPEG). For rPPG the total photons in the ROI matter, not the pixel count. 640x480 is fine.
- **Gain vs exposure:** for a given scene, longer exposure collects more photons (better SNR). Gain only scales the signal and adds noise, and it costs clipping headroom. Order of preference: more light > longer exposure (up to the frame-time limit) > gain > digital brightness (useless). Our driver's stuck gain is no loss.
- **8-bit:** 8-bit is sufficient if luma is mid-range and the ROI is large (the pulse is recovered by averaging thousands of pixels). It becomes the bottleneck only at low luma (our 23-61 case), where the pulse is a small fraction of an LSB and the camera's denoiser/temporal NR may smooth it away. Built-in webcams cannot output more than 8-bit over UVC YUY2.
- **Rolling shutter:** rows are exposed at different times, which matters for flicker banding (the detector in section 1) and gives a small (<1 frame) row-dependent phase offset for the pulse. That is negligible for HR and irrelevant for a single pooled ROI.
- **15 fps / 66 ms trade:** Nyquist at 15 fps is 7.5 Hz. HR up to 200 bpm (3.3 Hz) plus its 2nd harmonic (6.7 Hz) fits, so mean HR is fine. Peak timing quantisation grows to ±33 ms before interpolation, and the Sensors 2021 study shows HRV degrades below 30 fps. On this driver, though, -4 gives 22 fps, so the real exposure at -4 is ≤45 ms, not 62.5 ms. That is roughly +1.6 dB of light at best for a 27% frame-rate loss. **Recommendation: keep 30 fps at -5. Use 22 fps (-4) only if the face cannot be lit and you report only mean HR. Never let fps vary mid-recording, and always use measured timestamps.**
- **External webcams:** the C920/C922 (UBFC heritage) exposes manual exposure, gain, white balance, and focus via DSHOW. Kurokesu measured the C920 exposure range as [-11, -2] (see evm_realtime.md). Lock autofocus too, because AF hunting changes the face scale and blur. Avoid the Brio per Noldus. A C920 also lets you raise gain, which our built-in camera cannot, but light is still preferred.

### Gaps
- I found no study isolating 8- vs 10-bit, or YUY2 vs MJPEG at a fixed quality, on a webcam.
- I could not confirm which C920/C922 controls UBFC locked.
- There is no study comparing affordable webcams' sensor noise for rPPG. The Brio/LifeCam warning comes from one vendor only.

## 3. Distance, pose, behaviour, face pixel size, glasses, beard, skin tone

### Takeaway
Seat the subject 0.5-0.7 m from the camera with the face at least 100x100 px (target 150-250 px at 640x480), frontal, eye-level, still, and not talking. Talking roughly doubles RRI error and destroys HRV correlations. Darker skin (Fitzpatrick V-VI) is a large, documented error source for classic chrominance methods (MAE 5.2 to 14.1 bpm in one secondary report), and the mitigation is more light, larger ROI, and longer windows.

### Cited Findings
- Face size: PhysNet's MAE rose to >5 bpm once faces dropped below about **50x50 px**. With resolution-robust blocks it stayed <2 bpm at most sizes. — [Li, Yu & Shi, AAAI 2023, arXiv 2211.16922](https://arxiv.org/pdf/2211.16922)
- At 0.5-0.7 m, faces measured about 150-250 px on 640x480 webcams and about 350-550 px on 1920x1080. — [Architectural Tricks for Deep Learning in rPPG, arXiv 1911.02202](https://arxiv.org/pdf/1911.02202)
- Noldus: face ≥100x100 px, distance 0.5-1 m, forehead and upper face uncovered (no hair or hat), camera at eye level, frontal view throughout. Abrupt head-pose changes degrade results. — [Noldus guidelines](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/facereader/page/facereader-10-guidelines-for-an-optimal-vital-signs-measurement)
- Motion (POS, 500 lux, 30 fps, ECG reference): head fixed, RRI MAE 23.9 ms, LF/HF r = 0.976. Still but unrestrained, 26.1 ms, r = 0.643. **Speaking, 53.9 ms, r = 0.047.** — [Sensors 2021, PMC8704576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704576/)
- Side camera views raise deep rPPG MAE up to 3.15x versus frontal. — [CinC 2026 preprint 347](https://cinc.org/2026/Program/accepted/347_Preprint.pdf)
- On MMPD (mobile, includes talking), baseline MAE 4.81 versus 0.99 on the static subset. — [arXiv 2407.06653](https://arxiv.org/pdf/2407.06653)
- Skin tone: chrominance-method MAE 5.2 bpm (Fitzpatrick I-III) versus **14.1 bpm (V-VI)**, p<0.01. Deep models 6.0 versus 9.5 bpm (Dasari et al., cited secondarily). Nowara et al. found significantly poorer recovery for type VI. — [arXiv 2407.21519](https://arxiv.org/html/2407.21519v1); [Frontiers Digital Health 2025](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1553565/text); [PMC10568358](https://pmc.ncbi.nlm.nih.gov/articles/PMC10568358)
- Occlusion (toothbrushing, hair-combing) gave MAE 4.99 bpm. Clothing, hair, and cosmetics affect the signal. — [arXiv 2506.09718](https://arxiv.org/pdf/2506.09718v2)

### Inferences
- **Distance for our camera:** at 640x480, 0.5-0.7 m gives about 150-250 px faces, comfortably above the 100 px floor. Do not go beyond about 1 m (it falls toward 100 px and less light reaches the face). Closer than 0.4 m risks the face filling the frame, so AE and edges dominate.
- **ROI choice:** forehead plus cheeks, excluding eyes, mouth, beard, and glasses. A beard occludes the cheeks and chin, so use forehead and upper cheeks. Glasses frames give specular glints and edge motion, so mask the eye band (landmark-based), and use anti-glare positioning by tilting the light so reflections leave the lenses. Fringes covering the forehead: ask the subject to push hair back.
- **Skin tone mitigation:** darker skin absorbs more of the incident light (melanin), lowering both DC and pulsatile return. Compensate with light (≥500 lux), a larger ROI, longer windows (≥20-30 s for HR), and report results stratified by Fitzpatrick (or Monk) group. Do not claim accuracy across skin tones unless V-VI subjects are in the validation set.
- **Behaviour:** seated, back supported, forearms resting, no talking, normal breathing, gaze at a fixed point on the screen. Allow a 1-2 min settle before recording. Ask the subject not to touch their face.

### Gaps
- There is no single-protocol paper splitting MAE by glasses or beard.
- The Dasari skin-tone figures come from a secondary citation.

## 4. Validation: ground truth, protocol, metrics, pass criteria

### Takeaway
Use a **Polar H10 chest strap over BLE (standard Heart Rate Service 0x180D, characteristic 0x2A37, with RR intervals)** as the reference. It is ECG-grade for RR intervals at rest and readable on Windows with Python `bleak` (`bleakheart` or `polar-python`). A fingertip oximeter is a fallback for mean HR only. Pass criterion to claim consumer-tolerance HR: **error ≤ ±10% of reference or ±5 bpm, whichever is greater (ANSI/AAMI EC13 wording)**, applied per window, with MAE ≤ 5 bpm (and MAPE <10%, per ANSI/CTA-2065) as the summary bar. The stricter medical-oximeter bar is ARMS ≤ 3 bpm (ISO 80601-2-61 as used in FDA submissions).

### Cited Findings
- ANSI cardiac monitor / heart rate meter standard (EC13): accuracy is a "readout error of no greater than ±10% of the input rate or ±5 bpm, whichever is greater", with ECG as the gold standard. (Secondary quote; the EC13 text itself was not read.) — [PMC12568089](https://pmc.ncbi.nlm.nih.gov/articles/PMC12568089); [EC13:2002 document listing](https://mdcpp.com/doc/standard/ANSIAAMIEC13-2002.pdf)
- Consumer wearable HR: IEC 63203-402-3:2024 specifies comparative tests against a reference ECG. Wrist devices have been reported meeting the ANSI/CTA (2018) threshold of **<10% MAPE**. — [IEC 63203-402-3 preview](https://webstore.ansi.org/preview-pages/IEC/preview_iec63203-402-3%7Bed1.0%7Db.pdf); [PMC7010823](https://pmc.ncbi.nlm.nih.gov/articles/PMC7010823)
- FDA's 2013 pulse-oximeter 510(k) guidance points to ISO 80601-2-61 clause 201.12.1 testing. A clinical protocol states PR accuracy success as **ARMS within ±3 bpm over 30-300 bpm** (same in the 2011 and 2017 editions). Cleared devices claim ±2-3 bpm. — [FDA guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pulse-oximeters-premarket-notification-submissions-510ks-guidance-industry-and-food-and-drug); [NCT05297500 protocol](https://cdn.clinicaltrials.gov/large-docs/00/NCT05297500/Prot_SAP_000.pdf); [K231979](https://510k.innolitics.com/device/K231979)
- Polar H10 versus 12-lead ECG (n = 25, rest and incremental cycling): small RR bias and limits of agreement, with linear HRV agreeing well (DFA-a1 weaker at high intensity). A deep-breathing test showed near-identical RSA values versus a 1000 Hz lead II. — [Schaffarczyk et al., Sensors 2022, PMC9459793](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9459793/); [Georgia Southern record](https://scholars.georgiasouthern.edu/en/publications/validity-of-the-polar-h10-during-the-deep-breathing-autonomic-fun/)
- Polar white paper (manufacturer material): its predecessor, the H7, measured 95% of RR intervals within 2 ms in Holter comparisons. — [Polar H10 white paper](https://polar.com/img/static/whitepapers/pdf/polar-h10-heart-rate-sensor-white-paper.pdf)
- Python/BLE: `bleakheart` supports the standard BLE HR service including RR intervals, tested mainly on the Polar H10 (under Linux). `polar-python` lists the H10's "standard BLE heart rate stream with RR intervals". HR service 0x180D, characteristic 0x2A37 via bleak notifications. — [bleakheart (PyPI)](https://pypi.org/project/bleakheart/); [GitHub fsmeraldi/bleakheart](https://github.com/fsmeraldi/bleakheart); [polar-python](https://pypi.org/project/polar-python/1.0.0/); [SO answer](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/a/72541361)
- Wearables lose accuracy in transient states (motion onset, step HR changes). — [Sensors 2025, UHasselt](https://documentserver.uhasselt.be/bitstream/1942/47708/1/Accuracy%20of%20Heart%20Rate%20Measurement%20Under%20Transient%20States_%20A%20Validation%20Study%20of%20Wearables%20for%20Real-Life%20Monitoring.pdf)
- Exemplar rPPG validation metrics: RRI MAE (ms) against ECG, Pearson r of LF/HF. HRV used 2-min windows with a 15 s step. — [Sensors 2021, PMC8704576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704576/)

### Inferences
- **0x2A37 parsing** (from the Bluetooth Heart Rate Service spec; I did not re-fetch it this session, so verify): byte 0 is flags. Bit 0 set means HR is uint16, otherwise uint8. Bits 1-2 are sensor contact. Bit 3 set means a 2-byte Energy Expended field is present. Bit 4 set means RR intervals follow as uint16 little-endian in **units of 1/1024 s**, possibly several per notification. Timestamp each notification with `time.perf_counter()` on the same clock as video frames.
- **Windows bleak note:** Windows uses the WinRT backend. The strap must be worn (moistened electrodes) to advertise. Do not pair it in Windows Settings first, or scanning may miss it; connect by address. The bleakheart author tested mainly on Linux, so do a 5-minute smoke test on Windows.
- **Sync:** log video frame timestamps and BLE notification timestamps on one perf_counter clock. Add a sync event (e.g. a deliberate 3 s breath-hold, or a hand clap visible on camera) at the start. BLE notification latency is roughly tens of ms, which matters for IBI alignment but not for 10 s window HR.
- **Oximeter fallback:** cheap CMS50-type oximeters (PURE used a CMS50E) give HR and a pleth stream over USB-serial, but with seconds of averaging. Use them for mean HR only, not HRV. Smartwatches are not acceptable as references (they are themselves ±5-10%).
- **Metrics (per 10 s window, 1 s step, plus per-recording mean):** MAE, RMSE, MAPE, Pearson r, Bland-Altman bias and 95% LoA (±1.96 SD), %windows within the EC13 tolerance, and **coverage** (% of windows where the tool output a value rather than "no signal / gated"). Report accuracy only over covered windows, alongside coverage. For IBI/HRV: RRI MAE (ms), RMSSD and SDNN error, r of LF/HF on 2-min windows.
- **Minimal protocol:** see the paste-ready block below. For a hackathon claim, N = 6-10 subjects is defensible for "pilot", with ≥2 subjects of Fitzpatrick IV-VI. Formal consumer standards use larger N; IEC 63203-402-3 details were not read.

### Gaps
- I did not read the full EC13 or IEC 63203-402-3 texts (paywalled), including their N-subjects and conditions requirements.
- The FDA guidance PR clause text was not fetched directly. The ±3 bpm ARMS figure is from a sponsor protocol.

## 5. Automatable self-checks (go/no-go)

### Takeaway
Add six checks to the recorder: face ROI luma, clipping, measured fps and jitter, flicker on a wall patch, motion, and face size. Compute them live, refuse to record when hard limits fail, and stamp them into the recording metadata. Thresholds below are engineering values derived from the cited findings, and need tuning on our hardware.

### Cited Findings
- Bright frontal light leading to saturation degrades rPPG. — [arXiv 2107.13193](https://arxiv.org/pdf/2107.13193)
- Front light at ≥500 lux is best, 100 lux worst, overexposure above about 700 lux. — [PMC8704576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704576/)
- Face ≥100 px (Noldus). Below 50 px, error rises sharply. — [Noldus](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/facereader/page/facereader-10-guidelines-for-an-optimal-vital-signs-measurement); [arXiv 2211.16922](https://arxiv.org/pdf/2211.16922)
- Speaking doubles RRI error. Even unrestrained stillness drops the HRV correlation. — [PMC8704576](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704576/)
- Assumed versus actual fps mismatch is a known dataset error. — [arXiv 2305.04161](https://arxiv.org/html/2305.04161v3)

### Inferences (thresholds; tune on hardware)

| Check | How | GO | WARN | NO-GO |
|---|---|---|---|---|
| Face luma | mean Y in skin ROI (forehead+cheeks) | 100-170 | 70-100 or 170-200 | <70 or >200 |
| Green level | mean G in ROI | ≥90 | 60-90 | <60 |
| Clipping | % ROI px with any channel ≥250 | <1% | 1-5% | >5% |
| Dark clipping | % ROI px with Y ≤16 | <1% | 1-5% | >5% |
| fps | median of 1/dt over 5 s | 29-31 (or set target ±3%) | ±3-10% | >10% off, or target changes mid-run |
| Frame jitter | % of dt > 1.5x median (drops) | <1% | 1-3% | >3% |
| Flicker | wall-patch PSD at alias (10 Hz @30 fps) vs median 4-14 Hz floor | <6 dB | 6-12 dB | >12 dB, or row-mean banding visible |
| Light stability | wall-patch mean, 0.5-4 Hz RMS (LSB) | <0.2 | 0.2-0.5 | >0.5 (screen/AE/room light modulating) |
| AE/AWB lock | exposure/WB read back unchanged; global mean drift over 10 s | locked, <2 LSB drift | - | auto on, or >5 LSB drift |
| Motion | face-box centre displacement RMS (px/frame) and scale change | <1 px, <1% | 1-3 px | >3 px, or >3% scale change in 1 s |
| Face size | face box width, px | ≥150 | 100-150 | <100 |
| Pulse SNR (post-hoc) | power within ±0.1 Hz of HR peak & harmonic vs rest of 0.7-3.5 Hz (de Haan-style) | ≥3 dB | 0-3 dB | <0 dB (report "no reading") |

- Gate the output: any NO-GO in a window marks it "not covered" (this counts toward coverage, not error).
- Pre-flight sequence (10 s): open DSHOW, set YUY2 640x480@30, exposure -5, auto exposure off, WB locked if possible, then verify read-backs. Measure fps. Show the luma/clipping meter with "add light" or "move closer" prompts. Run the flicker check on the background. Only then enable Record.

### Gaps
- No published source gives these specific go/no-go numbers for webcam rPPG. They are derived and need calibration against Polar H10 runs.

---

### Paste-ready: CAPTURE CHECKLIST

```
CAPTURE CHECKLIST (webcam rPPG / colour EVM)
Camera
[ ] DSHOW, 640x480 YUY2 (uncompressed), 30 fps; read back FOURCC/FPS
[ ] Auto-exposure OFF, exposure = -5 (31 ms; also ~96% 100 Hz flicker suppression). Never -6/-7.
[ ] Auto-WB OFF / locked (driver dialog if API fails); autofocus OFF on external cams
[ ] Powerline frequency = 50 Hz (UK) in driver dialog if offered
[ ] Brightness/gain at default (digital brightness adds no signal)
Light
[ ] Face lit from the FRONT, diffuse, >=300 lux (aim ~500); no bright lights/windows behind subject
[ ] One constant source (LED/daylight-balanced lamp or bounced lamp); no mixed window light
[ ] Laptop screen: 100% brightness, adaptive brightness OFF, STATIC content (optionally full-screen white as fill)
[ ] Face ROI luma 100-170, green >=90, clipped <1% (tool meter green)
[ ] Flicker check on wall patch passes (no 10 Hz peak, no banding)
Subject
[ ] 0.5-0.7 m from camera, camera at eye level, frontal; face box >=150 px wide
[ ] Forehead uncovered; glasses off if possible (else tilt light to remove lens glints)
[ ] Seated, back supported, arms resting, 2 min settle; no talking, no face touching, normal breathing
Recording
[ ] fps stable (29-31), drops <1%; timestamps logged with perf_counter
[ ] Reference (Polar H10) streaming, same clock; sync event at start (3 s breath-hold or clap)
[ ] Log metadata: lux (phone meter app), exposure value, luma, Fitzpatrick type, glasses/beard, notes
```

### Paste-ready: VALIDATION PROTOCOL

```
VALIDATION PROTOCOL (pilot, "within consumer-device HR tolerance")
Reference: Polar H10 chest strap, BLE HR service 0x180D / char 0x2A37 (HR + RR in 1/1024 s),
           Python bleak (bleakheart or polar-python); same perf_counter clock as video.
Subjects:  N >= 6 (target 10), >= 2 of Fitzpatrick IV-VI, mix of glasses/beard; record lux & type.
Per subject (seated, 0.5-0.7 m, checklist passed), each block 2 min (HRV needs 2 min windows):
  B1 Baseline     : good light (~500 lux front), still, silent
  B2 Low light    : room light only (~100-200 lux), still
  B3 Natural      : unrestrained but still-ish, small head turns allowed
  B4 Talking      : reading aloud (expected degradation; report separately)
  B5 HR change    : after 30 s of stair-step/squats, seated recovery (HR range coverage)
  (optional B6: screen-white fill light vs B2)
Total ~12-15 min/subject incl. setup.
Analysis:
  Windows: 10 s HR windows, 1 s step; 2 min HRV windows, 15 s step.
  Per-window: HR error vs H10 (mean HR over same window).
  Metrics: MAE, RMSE, MAPE, Pearson r, Bland-Altman bias & 95% LoA,
           % windows within max(±5 bpm, ±10%), coverage (% windows with a valid output);
           IBI: RRI MAE (ms), RMSSD/SDNN error, r(LF/HF) [only B1/B3].
  Report per block and per Fitzpatrick group; never pool talking into the headline number.
Pass (claim "within consumer-device tolerance" for seated, still, lit conditions = B1):
  - >= 90% of covered windows within max(±5 bpm, ±10% of reference)  [ANSI/AAMI EC13 wording]
  - MAE <= 5 bpm and MAPE < 10%  [ANSI/CTA-2065-style]
  - Bland-Altman |bias| <= 2 bpm, LoA within ±10 bpm
  - coverage >= 80%
  Stretch (oximeter-grade): ARMS <= 3 bpm  [ISO 80601-2-61 PR accuracy as used by FDA]
  HRV claim only if RRI MAE <= ~30 ms and r(LF/HF) >= 0.9 (cf. 23.9 ms / 0.97 fixed-head benchmark).
Caveats to state: pilot N, single camera, not a medical device, skin-tone coverage limits.
```
