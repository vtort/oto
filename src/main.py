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
from vision_thread import VisionThread
from llm_thread import LLMThread
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
    import sys
    demo = "--demo" in sys.argv
    cfg  = load_config()
    bus  = StateBus()
    sm   = StateMachine(bus, cfg, demo=demo)
    stop = threading.Event()

    audio = AudioThread(bus, cfg)
    audio.start()
    print("[oto] audio thread started")

    vision = VisionThread(bus, cfg)
    vision.start()
    print("[oto] vision thread started")

    sm_thread = threading.Thread(target=state_machine_loop, args=(sm, stop), daemon=True, name="state")
    sm_thread.start()
    print("[oto] state machine started")

    llm = LLMThread(bus, cfg)
    if not demo:
        llm.start()
        print("[oto] LLM thread started — tap screen to talk")

    renderer = Renderer(bus, cfg, demo=demo)
    print("[oto] renderer starting — press ESC to quit")
    try:
        renderer.run()
    finally:
        stop.set()
        audio.stop()
        vision.stop()
        llm.stop()
        print("[oto] shutdown complete")


if __name__ == "__main__":
    main()
