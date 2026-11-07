# HackTheBurgh 2026: WiFi sensing + poker plan

Deadline: Sunday 12:00. Team of 3. One codebase, multiple tracks allowed.

## 1. Concept

Camera-free sensing of players' stillness and motion with WiFi CSI on ESP32s,
used as one input to a poker decision model. A phone camera reads cards so
bluff/value labels are generated automatically.

Honest framing: we measure arousal relative to each player's own baseline, not
lies. We report accuracy against labels, including "near chance" if that is
what we get.

Tracks: HRT (predictions and decision making), G-Research (make the invisible
visible), Bank of Scotland (energy-waste slide, presence detection, optional).

## 2. Hardware

- 4x ESP32 DevKit V1 (classic ESP32, built-in USB)
- 2x ESP32 camera boards (classic ESP32, no USB; one plug-in programmer module shared)
- Temp/humidity sensors, Pi 4B, laptop, phone (overhead camera)

Layout:

- 1-2 DevKits wired to the laptop as CSI receivers (serial).
- Remaining DevKits + camera boards are transmitters near each seat, powered by
  USB battery or charger.
- Keep links short (2-3 m). One link per player.
- Dedicated fixed WiFi channel, not the venue network.

## 3. Architecture

```
ESP32 mesh -> CSI pipeline -> arousal signal (per player, vs baseline) --\
                                                                          +-> poker decision model -> live dashboard
Phone camera -> card reader (or manual entry) -> hand equity + labels ---/
```

Decision model: standard poker odds, with a Bayesian update of bluff
probability from the arousal signal.

## 4. First steps on the ESP32s

Fastest path: Espressif's `esp-csi` repo supports all ESP32 series including
the classic ESP32. Its get-started examples are:

- `csi_recv`: ESP32 as receiver
- `csi_send`: ESP32 as sender
- `csi_recv_router`: router sends, ESP32 pings it to get CSI
- `tools/csi_data_read_parse.py`: CSI analysis script

The repo's `esp-radar` `wifi_sensing_demo` also does motion and presence
detection with on-site training and a browser Web Serial monitor. Check its
README for classic ESP32 support. If it works, it is the quickest viability
test.

Steps:

1. Install ESP-IDF on the laptop now (slow download; start in background).
   Follow the version the esp-csi README names.
2. Clone `https://github.com/espressif/esp-csi`.
3. Set target to `esp32`, build and flash `csi_send` on one board and
   `csi_recv` on a DevKit. Use `idf.py -p <port> flash monitor`. Check the
   example README for channel/MAC config.
4. Confirm CSI lines stream over serial from the receiver. Save a capture to file.
5. Run the three tests in section 5.
6. Only after tests pass: add more transmitters, then the full pipeline.

Camera boards: flash via the programmer module one at a time (GPIO0 to GND for
boot mode if the module does not auto-reset). Power from battery afterwards.
CSI does not need the camera running.

Parallel jobs while the above runs:

- Person 3: manual card-entry UI + equity calc (`eval7` or `treys`), poker
  model skeleton, dashboard.

## 5. Viability tests

Run in the actual hall, then again in the demo room.

| Test          | Procedure                                                                          | Pass                                                      |
| ------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 0. Link alive | CSI streams steadily, near-constant packet rate for 60 s, few gaps                 | Gaps under about 5% (adjust by eye)                       |
| 1. Motion     | 60 s empty, 60 s waving/walking between boards, 60 s sitting still; repeat 3 times | Motion variance clearly above empty baseline every repeat |
| 2. Presence   | Person seated still vs empty room                                                  | Reliably separable, not just once                         |
| 3. Breathing  | Seated 1-2 m from link, breathing to metronome at about 15 breaths/min             | Spectral peak near 0.25 Hz in most 30 s windows           |

Note: thresholds are rules of thumb. Judge on plots, and repeat to rule out luck.

If a test fails, try before giving up: different channel (1/6/11), shorter links, nodes closer to the person, raise packet rate (50-100 Hz), move away from walkways, test at quiet times.

## 6. Decision points

- Tests 0-2 pass: build the poker/presence project.
- Test 3 passes: add breathing as a feature.
- Test 3 fails: ship motion/stillness only, drop breathing.
- No clear response to motion within about 3 hours of starting, after tuning: pivot.

Pivot options (same hardware):

1. Card-vision poker analytics (equity, auto bluff labels, dashboard) as the
   main project.
2. Microclimate and heating-cost project with temp/humidity sensors (Bank of
   Scotland).

## 7. Venue notes

- Developing in a big atrium: crowds and 2.4 GHz congestion will add noise.
  Test breathing at quiet times.
- Demo likely in a separate room: baselines and models are room-specific. Build
  a one-minute "empty room baseline" step into the demo and use baseline-relative
  thresholds.
- Ask organisers where judging happens and whether the room can be tested.
- Record a fallback demo clip early.

## 8. Scope and build order

1. Must have: viability tests, CSI pipeline, live per-player trace.
2. Should have: manual card entry, equity, bluff/value labels, evaluation.
3. Nice to have: automatic card reading with a pretrained detector (verify a
   model exists early).
4. Last: GSR sensor (two finger contacts, resistor, ESP32 ADC), energy-waste
   slide, Bayesian polish.

## 9. Task split

- Person 1: firmware, network, CSI capture, placement.
- Person 2: signal processing, features, evaluation.
- Person 3: dashboard, cards/equity, poker model, pitch.

## 10. Pitch skeleton (about 3 minutes)

1. Hook: WiFi can see you move with no camera.
2. Live demo: Red Light, Green Light, then a poker hand with the live signal trace.
3. Method: baseline calibration, features, Bayesian update.
4. Honest results against camera/equity ground truth, with limitations.
5. Where next: energy waste, sleep, elder care.

## 11. Ethics

Get consent from anyone sensed or filmed. Use the camera only for card reading
and labelling.
