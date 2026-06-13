#!/usr/bin/env python3
"""
OTO 音 — Interactive Audio Mascot
Phase 1: Core engine + real-time audio FFT reactivity
"""

import os
import json
import time
import threading

# Pi5 display config — set before pygame import
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XAUTHORITY", "/home/pivic/.Xauthority")

from state import StateBus
from state_machine import StateMachine
from audio_thread import AudioThread
from renderer import Renderer


def load_config(path="config/config.json") -> dict:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, path)) as f:
        return json.load(f)


def state_machine_loop(sm: StateMachine, stop: threading.Event):
    while not stop.is_set():
        sm.tick()
        time.sleep(1 / 30)  # 30Hz is enough for state transitions


def main():
    cfg  = load_config()
    bus  = StateBus()
    sm   = StateMachine(bus, cfg)
    stop = threading.Event()

    audio = AudioThread(bus, cfg)
    audio.start()
    print("[oto] audio thread started")

    sm_thread = threading.Thread(target=state_machine_loop, args=(sm, stop), daemon=True, name="state")
    sm_thread.start()
    print("[oto] state machine started")

    renderer = Renderer(bus, cfg)
    print("[oto] renderer starting — press ESC to quit")
    try:
        renderer.run()
    finally:
        stop.set()
        audio.stop()
        print("[oto] shutdown complete")


if __name__ == "__main__":
    main()
