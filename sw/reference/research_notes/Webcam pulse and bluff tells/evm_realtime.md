# Real-time Eulerian Video Magnification (colour/pulse) on a laptop webcam (Python/OpenCV/NumPy)

Target pipeline under evaluation: webcam frame -> Gaussian pyramid down to ~160x120 -> per-pixel temporal IIR bandpass = LP(fc_hi=2.0 Hz) - LP(fc_lo=0.83 Hz), r = 1 - exp(-2*pi*fc/fs), fs = 30 -> multiply by alpha (~50) -> pyrUp -> add to full-res frame.

Numbers marked "computed" were calculated for these notes with NumPy from the filter equations (no external source). They are reproducible from the formulas.

## 1. What did Wu et al. 2012 use for colour/pulse magnification?

### Takeaway
For the face/pulse results Wu et al. used heavy spatial pooling: a Laplacian pyramid with the finest two levels zeroed in the paper, or a Gaussian downsample to level 4 (face) / level 6 (face2) in the released code. They used an **ideal (DCT/FFT) bandpass of 0.83–1.0 Hz**, alpha of 50 in the code (100–120 in the paper's table), and YIQ processing with chrominance attenuation available. The difference-of-two-first-order-lowpass IIR you propose is the paper's own suggested real-time filter, but they did not use it for the face pulse demos.

### Cited Findings
- Face and face2 colour results: "we applied a Laplacian pyramid and set alpha for the finest two levels to 0. Essentially, we downsampled and applied a spatial lowpass filter to each frame to reduce both quantization and noise and to boost the subtle pulse signal... passed each sequence of frames through an ideal bandpass filter with a passband of 0.83 Hz to 1 Hz (50 bpm to 60 bpm). Finally, a large value of alpha ≈ 100 and lambda_c ≈ 1000 was applied." — [Wu et al. 2012, SIGGRAPH, vidmag.pdf](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Table 1 (alpha, lambda_c, fl, fh, fs): face 100, 1000, 0.83, 1, 30; face2 pulse 120, 960, 0.83, 1, 30; face2 motion 20, 80, 0.83, 1, 30; baby2 150, 600, 2.33, 2.67, 30; wrist 10, 80, 0.4, 3, 30; baby 10, 16, 0.4, 3, 30. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Released code result filenames for face: ideal filter, 0.83333–1 Hz, **alpha 50, level 4, chromAtn 1**. face2 colour: ideal, 0.83333–1, alpha 50, **level 6**, chromAtn 1. baby2: ideal 2.3333–2.6667, alpha 150, level 6, chromAtn 1. baby/wrist (IIR, motion): r1 0.4, r2 0.05, alpha 10, lambda_c 16, chromAtn 0.1. The page says adult heart rate is around 1 Hz and **0.5–3 Hz is a "safe" range**. It also says to account for the frame rate. — [MIT EVM project page](https://people.csail.mit.edu/mrub/evm/)
- On filter choice: "for color amplification of blood flow, a narrow passband produces a more noise-free result... We use ideal bandpass filters for color amplification... Low-order IIR filters can be useful for both color amplification and motion magnification and are convenient for a real-time implementation. In general, we used two first-order lowpass IIR filters with cutoff frequencies ωl and ωh to construct an IIR bandpass filter." — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Colour space and chroma: "one can account for color clipping artifacts by attenuating the chrominance components of each frame. Our approach achieves this by doing all the processing in the YIQ space. Users can attenuate the chrominance components, I and Q, before conversion to the original color space." — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Pyramid kernel: "a separable binomial filter of size five" (i.e. [1 4 6 4 1]/16, the same as cv2.pyrDown). — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- The real-time prototype was C++, CPU-only, and processed **640x480 at 45 fps on a standard laptop**. Its Butterworth filter "is used to convert a user-specified frequency band to a second-order IIR structure and is used in our real-time application." — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Spatial pooling is essential. With added noise, a binomial filter of size 20 hid the pulse, while size 80 revealed it. "If the spatial filter applied is not large enough, the signal of interest will not be revealed." — [Wu et al. 2012, Fig. 11 / Sec. 5](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Magnification was manually restricted to marked regions for the face and wrist videos. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- The `eulerian-magnification` Python package example uses freq_min=50/60, freq_max=1.0, amplification=50, pyramid_levels=3. The author marks the package "not in a working state". — [brycedrennan/eulerian-magnification](https://www.github.com/brycedrennan/eulerian-magnification)

### Inferences
- The MIT code's "level 4" Gaussian downsample on a ~960x544 face video gives roughly a 60x34 image. On a 640x480 webcam, **~160x120 is only level 2**, which is far less spatial pooling than MIT used. If the pulse is not visible, go to level 3 (80x60) or level 4 (40x30) before raising alpha. pyrUp of a coarse level gives the smooth "blush" look seen in the MIT videos.
- MIT's 0.83–1.0 Hz band was tuned to the known heart rate of that subject. For an unknown live subject, 0.83–2.0 Hz (50–120 bpm) is a sensible widening, consistent with MIT's "0.5–3 Hz safe" guidance.
- Rather than full YIQ, a cheap equivalent is to amplify in RGB/YCrCb and scale the chroma part. Or amplify mainly the luma/green channel and keep chroma at about 1 (MIT used chromAtn = 1 for pulse, so no attenuation). For a "visible pulse" demo, keeping chroma (redness) is what makes the effect read as blood flow.

### Gaps
- The exact MATLAB function signatures (e.g. `amplify_spatial_Gdown_temporal_ideal(...)` in reproduceResults.m) could not be fetched (GitHub mirror 404). The parameters above come from the official result filenames and the paper.

## 2. Difference-of-two-first-order-lowpass IIR vs ideal FFT and Butterworth (phase, ringing, noise, settling)

### Takeaway
Your proposed filter (fc 0.83/2.0 Hz, fs 30) has r_hi = 0.3422 and r_lo = 0.1596. Its peak gain is only **~0.41 at ~1.3 Hz**, and its -3 dB band is very broad (~0.50–3.42 Hz). **Effective amplification is therefore about 0.41 x alpha (alpha=50 gives ~20x at the peak).** It also still passes ~14% at 10 Hz, so it amplifies flicker and noise. It is causal, O(1) memory (2 float frames), and settles in ~1 s. An ideal FFT bandpass is sharper and cleaner but needs a buffer of seconds and adds latency. A 2nd-order Butterworth bandpass (what MIT's real-time app used) is a modest upgrade with steeper skirts.

### Cited Findings
- MIT used ideal (DCT) filters for colour amplification because "they have passbands with sharp cutoff frequencies". The second-order IIR filters "have a broader passband than an ideal filter". — [Wu et al. 2012, Sec. 4 & Fig. 9](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- MIT's real-time application used a Butterworth band converted to a second-order IIR. — [Wu et al. 2012, Fig. 9 caption](https://people.csail.mit.edu/mrub/papers/vidmag.pdf); also restated in the [Linear-based Eulerian motion modulation patent US9811901](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9811901)
- Per-pixel FFT-based temporal filtering in a Python webcam project "proved to be too computationally expensive and is now used as a calibration step". — [kevroy314/respmon](https://github.com/kevroy314/respmon)
- MIT notes that Eulerian error "grows quadratically with alpha" and is more sensitive to high spatial frequencies. This supports pooling first and not over-driving alpha. — [Wu et al. 2012, Sec. 5](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Computed (NumPy, discrete-time, fs=30, causal update `lp += r*(x - lp)`):
  - r_hi = 1-exp(-2π·2.0/30) = **0.3422**; r_lo = 1-exp(-2π·0.83/30) = **0.1596**.
  - |H| / phase: 0.5 Hz 0.29 / +48°; 0.83 Hz 0.38 / +27°; 1.0 Hz 0.40 / +19°; 1.3 Hz **0.412 peak** / ~+7°; 2.0 Hz 0.38 / -12°; 3 Hz 0.32 / -25°; 5 Hz 0.22 / -32°; 10 Hz 0.14 / -20°.
  - -3 dB passband: **0.50–3.42 Hz** (much wider than the nominal 0.83–2.0).
  - Continuous-time check: H(s) = s(ωh-ωl)/((s+ωh)(s+ωl)). Peak at sqrt(fh·fl) = 1.29 Hz with gain (fh-fl)/(fh+fl) = 0.41.
  - Step response (e.g. a sudden brightness change or head move): peak 0.31 of the step, decays below 1% after **~26 frames (~0.9 s)**. There is no oscillatory ringing (two real poles), just an overshoot-and-decay "ghost".

### Inferences
- **Phase:** Near the heart-rate band the phase is small (+27° at 50 bpm, -12° at 120 bpm), so the amplified colour is roughly in sync with the real pulse. That is fine visually. Latency is negligible compared with an FFT window (several seconds).
- **Startup:** If the two low-pass states start at zero, the first frames output about 0.18 x pixel value x alpha. That is a white or black flash, then a ~1 s settle. **Initialise both states to the first downsampled frame** (lp_hi = lp_lo = frame0) so the output starts at 0. Optionally ramp alpha from 0 to target over the first 30 frames.
- **Noise:** Because the skirts roll off at only 6 dB/octave, sensor noise and lighting flicker at 5–15 Hz still get ~0.15–0.2 x alpha gain (about 7–10x at alpha=50). Fixes: (a) more spatial pooling (higher pyramid level), (b) cascade two stages (apply the same DoLP twice, or use a 2nd-order Butterworth bandpass via `scipy.signal.butter(1, [0.83, 2.0], btype='band', fs=30)`, which gives a 2nd-order biquad, implemented per pixel with a direct-form-II state of 2 frames), (c) a small extra temporal low-pass at ~3 Hz.
- **Gain calibration:** If you want "alpha" to mean the peak gain, divide by 0.41 (i.e. use alpha_eff = alpha / 0.41), or normalise the filter. Practically, start at alpha ≈ 50 with this DoLP filter (≈ 20x actual) and expose a slider of 20–150.
- **Butterworth vs DoLP:** both are causal and cheap (2 state frames). Butterworth gives a flatter top and steeper skirts, so less flicker for the same peak gain. Ideal FFT gives the cleanest result but needs N≈150–300 frames of history at 160x120x3 float32 (~35–70 MB at 300 frames) and a window delay. Fine for offline, awkward for live.

### Gaps
- I found no published side-by-side quantitative comparison (SNR, artefacts) of DoLP vs Butterworth vs ideal on face video. MIT only shows frequency responses.

## 3. Windows webcam control with OpenCV (exposure / WB / fps / backend)

### Takeaway
Property support on Windows is **driver- and backend-specific and often silently ignored**. Use `CAP_DSHOW` for manual camera control (it supports `CAP_PROP_SETTINGS`, the driver dialog, and the log2-seconds exposure scale). Expect `CAP_PROP_AUTO_WB` to be unsupported on older builds. Always read values back and verify visually. The most reliable hackathon fallback: open the DSHOW settings dialog once, untick auto exposure/WB by hand, and set exposure there.

### Cited Findings
- Windows 11, OpenCV 4.6.0: setting `CAP_PROP_AUTO_EXPOSURE` to 0.25 had no effect and `CAP_PROP_EXPOSURE` read back -1.0. The advice was to pass CAP_DSHOW or CAP_MSMF explicitly, and the forum noted those backends get little maintenance. — [OpenCV forum: Dino-Lite exposure](https://forum.opencv.org/t/exposure-setting-issue-with-opencv-in-python-microscope-dino-lite/10174)
- Windows 10: DSHOW returned -1 for the property, while MSMF accepted the set but always read back 0. Other backends treat AUTO_EXPOSURE as boolean, and V4L2 uses 0–3. — [OpenCV forum: set camera parameter](https://forum.opencv.org/t/set-camera-paramter-without-reset-of-it/15021)
- 0.25 / 0.75 are the values some drivers map to manual / auto, but this is camera-dependent. — [Stack Overflow mirror](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/53545945/how-to-set-camera-to-auto-exposure-with-opencv-3-4-2)
- `CAP_PROP_AUTO_WB` was not handled in `cap_msmf.cpp` or `cap_dshow.cpp` as of OpenCV 4.5.1 (get = -1, set fails). The issue was closed with "nearly all videoio properties are optional and backend-specific". PR #23300 (merged March 2023) appears to add AUTO_WB support. — [opencv/opencv#19621](https://github.com/opencv/opencv/issues/19621)
- DSHOW exposure is log2 seconds (value n means 2^n s): -6 ≈ 15.6 ms, and each step halves or doubles the time. A Logitech C920 range was [-11, -2]. Some drivers treat the values as steps rather than true times. — [Kurokesu: UVC exposure timing in OpenCV](https://www.kurokesu.com/main/2020/05/22/uvc-camera-exposure-timing-in-opencv/); [SO mirror](https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/45449780/finding-exposure-duration-of-a-webcam-in-time-units); [OpenCV forum](https://forum.opencv.org/t/change-exposure-with-opencv-linux/10647/9)
- `CAP_PROP_SETTINGS` pops up the camera driver dialog and is "only supported by DSHOW backend currently". Usage is `cap.set(cv2.CAP_PROP_SETTINGS, 1)`. — [OpenCV forum](https://forum.opencv.org/t/change-exposure-with-opencv-linux/10647/9) (quoting the OpenCV enum docs)
- MSMF opening can take seconds to minutes. Setting env `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` *before* creating the capture "may improve initialization time". CAP_DSHOW avoids the delay. Each extra `cap.set()` reportedly lengthens MSMF init. DSHOW may use more USB bandwidth. — [OpenCV forum 13954](https://forum.opencv.org/t/using-opencv4-7-0-cap-open-0-to-open-the-camera-is-slow/13954); [OpenCV forum: MSMF slow initial frame](https://forum.opencv.org/t/the-msmf-backend-is-slow-to-acquire-the-initial-frame/24121); [OpenCV forum 12248](https://forum.opencv.org/t/opencv-webcam-error/12248)
- Live EVM apps note that the *capture* fps (not the processing fps) is what the temporal filter's Hz band depends on. If frames are not dropped when the buffer fills, capture rate falls to the processing rate. — [MattW2/Live-Video-Magnification](https://github.com/MattW2/Live-Video-Magnification); [tschnz/Live-Video-Magnification](https://www.GitHub.com/tschnz/Live-Video-Magnification)

### Inferences (code-level recommendation; verify on the actual laptop)
```python
import os, time, cv2
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"   # only matters if MSMF used
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # set FOURCC before size/fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # often ignored on DSHOW/MSMF; harmless
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)    # DSHOW: 0.25 = manual on many drivers
cap.set(cv2.CAP_PROP_EXPOSURE, -6)           # ~1/64 s; try -5..-7
cap.set(cv2.CAP_PROP_AUTO_WB, 0)             # may be unsupported -> returns False
for p in ("FOURCC","FPS","AUTO_EXPOSURE","EXPOSURE","AUTO_WB","WB_TEMPERATURE"):
    print(p, cap.get(getattr(cv2, "CAP_PROP_"+p)))
# fallback: cap.set(cv2.CAP_PROP_SETTINGS, 1) -> untick auto exposure/WB in the driver dialog
```
- **Measure the real fs.** Time 60 `cap.read()` calls and compute r from the *measured* fps, not 30. In low light, auto-exposure laptop cams often drop to 15 fps. If fps changes and the r values are not recomputed, the band shifts (at 15 fps, r_hi=0.3422 corresponds to ~4 Hz).
- Use the measured per-frame dt to recompute r = 1 - exp(-2π fc dt) every frame. This makes the filter robust to dropped frames at negligible cost.
- If auto-exposure/WB cannot be disabled, the AE/AWB "hunting" (slow brightness/colour drift around 0.1–1 Hz) falls right in or near the passband and gets amplified into whole-frame pulsing. This is the most likely failure mode. Lock exposure, or at minimum subtract the frame-global mean of the bandpassed signal (outside the face) from the per-pixel bandpass.
- **UK mains flicker (100 Hz lighting)** aliases into the video at 30 fps (to ~10 Hz). An exposure that is a multiple of 10 ms suppresses it. The DSHOW log2 scale cannot hit 10 ms exactly (-7 ≈ 7.8 ms, -6 ≈ 15.6 ms), so prefer daylight or DC LED light, or the driver's "powerline frequency 50 Hz" option in the settings dialog.
- Run capture in a separate thread that always keeps only the newest frame, so processing lag does not build a backlog. Still filter using the timestamps of the frames actually processed.

### Gaps
- I could not fetch the official OpenCV videoio flags doc page (403) to quote the CAP_PROP_BUFFERSIZE backend support. I also found no source confirming whether MJPG fourcc setting is honoured on MSMF vs DSHOW for typical integrated laptop cams (many integrated cams deliver only YUY2/NV12 at 640x480@30, which is fine without MJPG).

## 4. Motion artefacts and mitigations

### Takeaway
Any head motion or edge shift produces a large temporal derivative that the bandpass passes and alpha amplifies. The result is coloured halos/ghosts around the face outline and hair that decay over ~1 s with this filter. Restrict magnification to a skin/face mask, clip the amplified delta, and reduce or reset when motion is detected.

### Cited Findings
- MIT manually restricted magnification to user-marked regions for face and wrist "since some regions... when amplified, are perceptually unappealing". — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Using a broader second-order IIR on face2 magnified head motion instead of skin colour change. Filter choice decides whether you see motion or colour. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Chrominance attenuation in YIQ is MIT's fix for colour clipping artefacts. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- Python face-heart-rate projects use OpenCV Haar cascade face detection to restrict processing to the face. — [rohintangirala/eulerian-remote-heartrate-detection](https://gh.nn.ci/rohintangirala/eulerian-remote-heartrate-detection)
- Computed: the DoLP filter's step response peaks at 0.31 of a step and takes ~26 frames to decay below 1%. A head movement that shifts a 100-level edge leaves ~31 x alpha of "ghost" before clipping.

### Inferences
- **Face-ROI mask:** detect the face every N frames (Haar `haarcascade_frontalface_default.xml` or MediaPipe Face Detection). Build a soft elliptical mask at the downsampled resolution (Gaussian-feathered, shrunk ~10–15% to exclude hair, eyes and background edges). Multiply the amplified delta by the mask before pyrUp, or after (pyrUp of the masked delta naturally feathers it).
- **Clip:** `delta = np.clip(alpha*bp, -C, C)` with C ≈ 20–40 levels (8-bit). Then `out = np.clip(frame + up(delta), 0, 255).astype(np.uint8)`. Do the arithmetic in float32 or int16 to avoid uint8 wrap-around, which is the classic "neon speckle" bug.
- **Motion gating:** compute mean |frame_t - frame_{t-1}| in the face ROI at low res. If it exceeds a threshold, (a) ramp alpha down to 0, and/or (b) re-initialise lp_hi = lp_lo = current frame to kill the ghost, then ramp alpha back up over ~0.5–1 s.
- Ask the subject to keep still with their head rested. In live EVM demos, motion is the number one artefact.
- Lower alpha (20–30) and a higher pyramid level make the result more robust than a high alpha at fine resolution.

### Gaps
- I found no quantitative study of motion-gating thresholds for live EVM. The thresholds above need empirical tuning.

## 5. Lighting, skin tone, 8-bit quantisation and noise

### Takeaway
The pulse colour signal is sub-quantisation-level per pixel in 8-bit video. It only becomes visible after spatial averaging (pyramid pooling), which beats down sensor and quantisation noise in proportion to the pooled area. Darker skin and dim light lower the SNR, so they need more pooling and steady, bright, flicker-free light.

### Cited Findings
- "Temporal filtering needs to be applied to lower spatial frequencies (spatial pooling) to allow such a subtle input signal to rise above the camera sensor and quantization noise." — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- "Spatial low pass filtering reduces the variance of the noise according to the area of the low pass filter". Filtered noise power is ∝ σ²/r², and the filter radius r should be chosen so that signal power exceeds noise. — [Wu et al. 2012, Sec. 5](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- MIT demonstrated colour amplification on both a lighter-complexion (face) and darker-complexion (face2) subject. face2 used a coarser pyramid level (6 vs 4) and higher alpha in Table 1 (120 vs 100). — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf); [MIT EVM page](https://people.csail.mit.edu/mrub/evm/)

### Inferences
- Each pyramid level is a 2x downsample, so it averages ~4x more pixels and cuts noise std by ~2x. Going from 160x120 (level 2) to 40x30 (level 4) is roughly a 4x lower noise std in the bandpassed signal. Use this before raising alpha.
- Laptop webcams apply heavy compression, denoising and sharpening. Denoising can smear the pulse, and compression adds blocky temporal noise. Coarse pooling also hides these.
- Lighting: bright, diffuse, frontal, constant (daylight or non-PWM LED), with no flickering monitors behind the subject. Keep the face reasonably bright (not saturated) so 8-bit headroom exists on both sides after adding the amplified delta.
- Process in float32 throughout. Convert the input to float once, keep the filter state float32, and quantise only at the final display.
- Green carries the strongest PPG signal (well known in rPPG literature), but for visible "redness pulse" EVM, amplifying all channels (or YIQ with chroma kept) gives the red-green alternation seen in MIT's demo.

### Gaps
- I did not find quantitative per-skin-tone SNR numbers for webcam EVM within the time budget.

## 6. Open-source real-time Python EVM implementations and performance

### Takeaway
There is no maintained, real-time, Python, webcam EVM reference with published fps. The real-time open-source apps are C++/Qt. Python repos are mostly offline (whole-video FFT). Writing the ~40-line IIR loop yourself is the practical route. At 160x120 (or smaller) per-pixel NumPy ops are trivially real-time; the full-res pyrUp/add/clip and capture dominate.

### Cited Findings
- tschnz/Live-Video-Magnification and its fork mfkiwl/Live-Video-Magnification are OpenCV/Qt (C++) real-time EVM apps that support cameras. The Riesz phase-based mode was "not yet tested for Live Webcams". — [tschnz](https://www.GitHub.com/tschnz/Live-Video-Magnification); [mfkiwl](https://github.com/mfkiwl/Live-Video-Magnification)
- MattW2/Live-Video-Magnification: a Qt 6 GUI with a live FPS readout implementing the MIT techniques (language not confirmed from the search snippet). — [MattW2](https://github.com/MattW2/Live-Video-Magnification)
- brycedrennan/eulerian-magnification: a pip-installable Python package operating on loaded video files. The author states it is "not in a working state". — [brycedrennan](https://www.github.com/brycedrennan/eulerian-magnification)
- PyVHR `evm2.py` demo: uses cv2.pyrDown and an ideal FFT bandpass (offline). — [PyVHR evm2.py](https://huggingface.co/datasets/navagg/PyVHR/blob/main/evm2.py)
- MIT's C++ CPU prototype ran at 640x480 @ 45 fps on a 2012 laptop. — [Wu et al. 2012](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)
- A per-pixel FFT webcam pipeline in Python was "too computationally expensive" for real time. — [kevroy314/respmon](https://github.com/kevroy314/respmon)

### Inferences (reference implementation sketch)
```python
import cv2, numpy as np, time
LEVELS, ALPHA, FL, FH, CLIP = 2, 50.0, 0.83, 2.0, 30.0   # try LEVELS=3/4 if noisy
def down(f):
    for _ in range(LEVELS): f = cv2.pyrDown(f)
    return f
def up(d, shape):
    for _ in range(LEVELS): d = cv2.pyrUp(d)
    return cv2.resize(d, (shape[1], shape[0])) if d.shape[:2] != shape[:2] else d
lp_hi = lp_lo = None; t_prev = None
while True:
    ok, frame = cap.read();  t = time.perf_counter()
    if not ok: break
    f = frame.astype(np.float32)
    s = down(f)
    if lp_hi is None:
        lp_hi = s.copy(); lp_lo = s.copy(); t_prev = t; continue
    dt = t - t_prev; t_prev = t
    rh = 1 - np.exp(-2*np.pi*FH*dt); rl = 1 - np.exp(-2*np.pi*FL*dt)
    lp_hi += rh * (s - lp_hi); lp_lo += rl * (s - lp_lo)      # in-place, float32
    delta = np.clip(ALPHA * (lp_hi - lp_lo), -CLIP, CLIP)      # * face_mask[...,None]
    out = np.clip(f + up(delta, f.shape), 0, 255).astype(np.uint8)
    cv2.imshow("evm", out)
    if cv2.waitKey(1) == 27: break
```
- Cost estimate: pyrDown x2 + 6 elementwise ops on 160x120x3 float32 (~58k values) take well under 1 ms. pyrUp x2 to 640x480, add and clip are ~1–3 ms in OpenCV/NumPy on a modern laptop. 30 fps should be easy, so capture is the bottleneck. These are estimates, not measured. Have the user print a per-stage timing.
- Pitfall: pyrDown/pyrUp sizes are ceil/floor-sensitive for non-power-of-2 dims. Handle the mismatch with a final resize (as above) or by cropping the frame to multiples of 2^LEVELS.

### Gaps
- I found no published fps numbers for a Python webcam EVM. The timing figures above are engineering estimates to be validated on the laptop.

## 7. Published FPGA / fixed-point EVM implementations

### Takeaway
FPGA EVM has been demonstrated: a streaming core of up to 90 Mpixel/s (Shi & Luo, TCAS-II 2018), a Kintex-7 1080p accelerator (Zhang, Jin & Wu), and an MIT 6.111 student project (2012). The key hardware simplification is to apply the temporal bandpass once, at a single coarse scale, rather than per pyramid level. That matches the "single Gaussian level" colour pipeline here. I could not access fixed-point word lengths within the time budget.

### Cited Findings
- Shi & Luo, "A Streaming Motion Magnification Core for Smart Image Sensors", IEEE TCAS-II 2018: a modified EVM that applies the pixel-wise temporal bandpass once instead of on every scale layer, cutting memory and multipliers. The FPGA prototype handles up to **90 Mpixel/s** with results comparable to the PC algorithm. — [BIDMC research page](https://research.bidmc.org/eye/publications/streaming-motion-magnification-core-smart-image-sensors)
- Zhang, Jin & Wu, "Accelerating Eulerian Video Magnification Using FPGA" (ICACT 2017): hardware Laplacian pyramid, temporal filter and reconstruction, pipelined and parallel, on a Xilinx Kintex-7, with a speedup over an Intel i5 at 1920x1080. — [ICACT 2017 listing](https://icact.org/upload/2017/0021/20170021_biography.pdf)
- MIT 6.111 (2012) team 01: an FPGA real-time video magnification project. Its main challenges were timing and the memory to store multiple frames. — [MIT 6.111 report](https://fpga.mit.edu/videos/2012/team01/report.pdf) (403 when fetched; summary from search snippet only), [proposal](https://fpga.mit.edu/videos/2012/team01/proposal.pdf)
- The MIT patent family repeats that low-order IIR filters "are convenient for a real-time implementation". — [US9811901](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9811901)

### Inferences
- The DoLP filter is very FPGA-friendly. Each low-pass is `lp += (x - lp) * r`. Quantising r to a power of two (r_hi ≈ 0.3422 → 3/8 or 11/32; r_lo ≈ 0.1596 → 5/32) makes it shift-add only. State needs fractional bits (e.g. 8.8 or 8.10 fixed point) because the pulse delta is sub-LSB in 8-bit. With integer-only state, the slow filter's update (0.16 x small differences) rounds to zero and the filter stalls (a dead-band/limit-cycle issue).
- Memory: two state frames at 160x120x3 x 16 bit ≈ 230 KB, which fits in BRAM on a Zynq-7020 (~560 KB BRAM). At 40x30 it is trivial.
- Such a port touches the shared interface (PLAN.md Section 3) and the hw/ folder, so it must be coordinated with the hw owner. These notes do not change any contract.

### Gaps
- No fixed-point bit widths or accuracy-vs-float data were retrievable (the Shi & Luo and Zhang papers are paywalled, and the MIT 6.111 report returned 403).
