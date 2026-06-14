#!/usr/bin/env python3
"""OTO 音 — Interactive Audio Mascot"""
import os
import sys
import json
import time
import queue
import threading
import platform

if platform.system() == "Linux":
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XAUTHORITY", "/home/pivic/.Xauthority")

from state import StateBus
from state_machine import StateMachine
from audio_thread import AudioThread
from vision_thread import VisionThread
from llm_thread import LLMThread
from renderer import Renderer


def load_config() -> dict:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = "config.mac.json" if platform.system() == "Darwin" else "config.json"
    with open(os.path.join(base, "config", name)) as f:
        return json.load(f)


def main():
    demo = "--demo" in sys.argv
    cfg  = load_config()
    bus  = StateBus()
    stop = threading.Event()

    speech_q = queue.Queue()

    audio = AudioThread(bus, cfg, speech_q)
    audio.start()
    print("[oto] audio thread started")

    vision = VisionThread(bus, cfg)
    vision.start()
    print("[oto] vision thread started")

    sm = StateMachine(bus, cfg, demo=demo)
    sm_thread = threading.Thread(
        target=lambda: [sm.tick() or time.sleep(1/30) for _ in iter(lambda: stop.is_set(), True)],
        daemon=True, name="state"
    )
    sm_thread.start()
    print("[oto] state machine started")

    llm = LLMThread(bus, cfg, speech_q)
    if not demo:
        llm.start()
        print("[oto] LLM thread started")

    renderer = Renderer(bus, cfg, demo=demo)
    print("[oto] renderer starting — SPACE to talk, ESC to quit")
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
