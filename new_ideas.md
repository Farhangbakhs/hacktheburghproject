# HackTheBurgh 2026: other ideas (parked)

Chosen project: Eulerian video magnification (see evm-build-plan.md). Everything else we considered is below, grouped by theme, with tracks, feasibility and why we parked it.

Tracks: HRT (data for predictions and decision making), G-Research (make the invisible visible), Bank of Scotland (smart cost of living), JetBrains (help the developer), Wordsmith (agentic browser workflow), Instalily (coding agent that adapts to the user), Viridien (lithology).

## Signal processing and hidden signals

### Hidden images in audio

Encode an image into a sound's spectrogram; an FPGA FFT shows a live spectrogram that reveals it. Interactive version: a judge types a word, we play it from a phone, it scrolls across the screen.

- Tracks: G-Research
- Feasibility: good. Over the air works for bold images at 1-10 kHz; the hidden-in-a-song version needs a cable.
- Status: strong backup. Full plan in hidden-images-in-audio-plan.md.

### Inaudible data channel

A phone sends a message at 18-20 kHz; the FPGA decodes it in real time.

- Tracks: G-Research
- Feasibility: medium. Phone speakers and laptop mics roll off at high frequencies; test first.
- Status: possible add-on to the audio idea.

### Hidden pictures in video bits

A hidden image sits in the lowest bits of normal video; the FPGA extracts and amplifies it live.

- Tracks: G-Research
- Feasibility: easy.
- Status: small add-on or fallback.

### Timestamping recordings from mains hum (ENF)

Grid frequency drifts around 50 Hz and the drift leaks into recordings as hum. Matching it against NESO's public 1-second GB frequency data tells you when a recording was made, and jumps reveal edits.

- Tracks: G-Research; HRT if extended to predicting grid frequency
- Feasibility: medium. NESO data is monthly, so this weekend's recordings can't be matched by Sunday; use older recordings or two simultaneous laptop recordings as a live reference. Recordings need several minutes with hum present.
- Status: favourite non-FPGA idea. 30-minute test: two laptops on chargers record 10 minutes; check for the same wandering line at 50/100/150 Hz.

### Data hidden in LED flicker

An Arduino blinks an LED too fast for eyes; a phone camera's rolling shutter turns it into stripes we decode.

- Tracks: G-Research
- Feasibility: good; very controllable hardware, work is in decoding phone video.

## Trading and markets

### FPGA order-book signal engine

Replay recorded market data from memory into the PYNQ-Z2, compute signals in a streaming pipeline, decide per tick, and report latency in clock cycles vs Python.

- Hidden signal options: iceberg orders (hidden liquidity inferred from refills and trades exceeding displayed size), order book imbalance (classic baseline), spoofing patterns.
- Tracks: HRT, G-Research
- Feasibility: medium. PYNQ-Z2 Ethernet goes to the ARM, not the fabric, so no true network-to-trade latency. Needs suitable order-book + trade data; iceberg detection needs detailed data.
- Status: parked; judges would care more about signal quality than speed.

### Alternative data for markets

Use a hidden public signal (night-time satellite lights, shipping traffic, sentiment) to predict an economic or market variable; evaluate against a baseline.

- Tracks: HRT, G-Research
- Feasibility: good, but a common hackathon idea; needs a novel signal.

## WiFi sensing (ESP32 CSI)

### WiFi presence and breathing sensing

ESP32s exchange packets; Channel State Information changes as people move or breathe. Espressif's esp-csi supports classic ESP32 (our 4 DevKit V1 + 2 camera boards). RuView repo used as reference only (targets S3/C6).

- Tracks: G-Research
- Feasibility: presence/motion likely, breathing fragile, heart rate unreliable.
- Why parked: too many uncontrolled variables (crowded 2.4 GHz atrium, room-specific calibration, unknown demo room, noisy classic ESP32 CSI).

### Red Light, Green Light

Players freeze on red; WiFi sensing detects who moved. Fun hook, doubles as detector test.

- Status: parked with WiFi sensing.

## Poker

### Poker tells / arousal sensing

Measure stillness, fidgeting, breathing (WiFi), blink rate, rPPG pulse, voice jitter, wrist IMU, or a DIY skin-conductance sensor, relative to each player's baseline; Bayesian update of bluff probability on top of poker odds.

- Tracks: HRT, G-Research
- Why parked: no reliable physiological bluff tell; stress is not bluffing; too little data to show anything real; ESP32-CAM unsuitable for rPPG.

### Poker with card vision + seat consoles

Our working card-detection model gives community cards and showdown hands; equity (eval7/treys) auto-labels bluff vs value. Arduino seat consoles (fold/call/raise, bet size) log action timing and sizing, which are more reliable tells than physiology. Planted-tell experiment (randomised instructions, detection rate by exaggeration level) proves the rig works without claiming real-tell detection.

- Tracks: HRT, G-Research
- Why parked: still too uncontrollable for the time and players available; no chips/table kit.

## Home energy and environment

### Hidden heat and damp map

ESP32 + DHT22 nodes per room; detect cold spots, draughts, mould-risk humidity, heat loss after heating goes off; convert to £ with the energy price cap.

- Tracks: Bank of Scotland, G-Research
- Why parked: no reliable room to instrument; atrium barely changes.

### Home that senses, predicts and acts

Thermal model per room, mould-risk forecasting, pre-heating savings, then a browser agent completes one workflow (tariff comparison, damp report) on a demo site.

- Tracks: Bank of Scotland, G-Research, Wordsmith, HRT
- Why parked: same room problem; agent workflow adds scope.

### Digital twin / scale-model house

Cardboard house with sensors per room, window/door event sensing (ultrasound as event detector, not mapping), live heat-flow model, predictions, closed-loop control vs naive thermostat.

- Why parked: no building materials, servos, motor drivers or safe heater.

### Energy waste in empty rooms

Presence detection + temperature shows heating running with nobody there, in £.

- Status: parked with WiFi sensing and room access.

## Developer tools and agents

### Embedded debugging agent

Agent/IDE plugin that flashes ESP32/Arduino, reads serial output and iterates on fixes.

- Tracks: JetBrains, Instalily
- Why parked: plugin learning curve; agent is the real work.

### Developer-state-aware coding assistant

Infer focus or frustration from typing rhythm, undo bursts, error loops and pauses (or physical presence); the agent adapts (breaks, simpler explanations, batches work while you're away). Team's own data only, with consent.

- Tracks: JetBrains, Instalily, G-Research
- Why parked: needs a solid coding agent underneath.

## Data visualisation

### Your own hidden activity

Log the team's presence and restlessness at the table through the hackathon and visualise it.

- Tracks: G-Research
- Status: quick but unambitious.

### Hidden music evolution

Map how a genre's tempo, loudness or harmony changed over decades from public audio features.

- Tracks: G-Research
- Status: low risk, but judges will have seen it before.
