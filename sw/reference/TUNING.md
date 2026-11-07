# Webcam tuning notes (plan step 2)

Fill one row per configuration tried. Paste the `[camera]` block printed at startup.

| Date/time | Laptop / camera | Backend | Resolution | Measured fps | Exposure locked? (how) | WB locked? | Lighting | alpha / band | Pulse visible? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-07 eve | team laptop, built-in cam | dshow | 640x480 (YUY2; MJPG rejected) | 30.0 | yes, auto-settle then lock -> -6 (15.6 ms) | yes, 4000 K | ceiling fluorescents, no face lamp | 50 / 0.83-2.0 | no - skin brightness 23/255, centre box missed face | buffersize + autofocus rejected; proc 11 ms |
| 2026-11-07 eve | same | dshow | 640x480 | 29.7 | yes, `--exposure -5` (31 ms) | yes, 4000 K | same | 50 / 0.83-2.0 | not yet - brightness 61, subject moving/laughing: motion halos dominate | face ROI (Haar) tracks well |

## What worked

- DirectShow honours fps=30, exposure and auto_wb=0 on this camera (read back OK).
- `--exposure -5` doubles brightness vs the auto-chosen -6 without dropping fps.
- Still needed: a lamp/screen light on the face (target skin brightness >= 80) and a still subject.

## Known-good starting point (to fill)

```
python evm_float.py --backend ... --exposure ... --wb ...
```

## Tips

- A steady lamp in front of the face (not behind) is the biggest win. Avoid
  windows and flickering LEDs.
- Sit still; head motion shows up as coloured halos at edges.
- If the fps readout is red, the room is too dark (auto-exposure lengthens the
  shutter) or exposure is too long: add light or lower `--exposure`.
- `d` opens the DirectShow driver dialog: untick auto exposure and auto WB there
  if the CLI lock doesn't take.
