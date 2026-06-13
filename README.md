# OTO 音 — Interactive Audio Mascot

Reactive creature that lives on a Raspberry Pi 5 + Waveshare 7" DSI display.
Listens to audio frequencies, sees presence via camera, responds to touch.

## Architecture

```
[Audio FFT]  ──┐
[OpenCV]     ──┤──→ State Bus ──→ State Machine ──→ Renderer (pygame 60fps)
[Touch]      ──┘
```

All sensing runs in isolated threads. The renderer only reads shared state.
Core never depends on internet — WiFi used only for Phase 4 (LLM/voice).

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core engine + real-time audio FFT reactivity | 🔄 In progress |
| 2 | Camera presence detection (OpenCV) | ⏳ Pending |
| 3 | Touch zones + persistent personality | ⏳ Pending |
| 4 | Wake word + Whisper + Claude API voice | ⏳ Pending |

## Hardware

- Raspberry Pi 5 (4GB+)
- Waveshare 7" DSI Capacitive Touch Display (800×480)
- Innomaker U20CAM-1080P USB camera
- USB microphone

## Setup

```bash
pip install -r requirements.txt
python src/main.py
```

## Config

Edit `config/config.json` to adjust mascot behavior, audio sensitivity, display resolution.
