"""
LLM thread: VAD+Whisper wake word → STT → Claude → TTS → ANSWER state

Reads raw audio from StateBus (published by AudioThread) — never opens
the microphone directly, avoiding device contention.
"""
import threading
import queue
import time
import os
import tempfile
import wave
import subprocess
import re
import numpy as np
import anthropic

from state import MascotState, StateBus

# ── Audio constants (must match AudioThread) ──────────────────────────────────
SAMPLE_RATE   = 44100          # AudioThread default
CHANNELS      = 1
SAMPLE_WIDTH  = 2              # paInt16

VAD_THRESHOLD = 0.015          # raw_rms threshold (AudioThread normalizes by /32768)
SILENCE_LIMIT = 1.2            # seconds of silence to end recording
MAX_WAKE_S    = 4              # max seconds to capture for wake word check
MAX_QUESTION_S = 15            # max seconds for full question

# ── Wake word variants ────────────────────────────────────────────────────────
WAKE_PATTERNS = [
    r"oye\s+oto",
    r"ey\s+oto",
    r"hey\s+oto",
    r"oi\s+oto",
    r"oye\s+otto",
    r"hey\s+otto",
    r"oye\s+auto",    # Whisper mishears "OTO" as "auto"
    r"oh[,\s]+y\s+eso",  # "Oh, y eso" variant
    r"oye\s+ot[ao]",
    r"oye\s+o\.?t\.?o",
    r"\boto\b",       # bare "OTO" — fallback if clear enough
]

SYSTEM_PROMPT = (
    "Eres OTO, una mascota interactiva inteligente y cercana. "
    "Responde siempre en el mismo idioma que el usuario (castellano o catalán). "
    "Respuestas concisas, máximo 2-3 frases. Sin markdown, solo texto plano."
)


def _is_wake_word(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in WAKE_PATTERNS)


class LLMThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict):
        super().__init__(daemon=True, name="llm")
        self.bus     = bus
        self.cfg     = cfg
        self._stop   = threading.Event()
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._whisper = None
        # Detect actual sample rate from config
        self._rate = cfg.get("audio", {}).get("sample_rate", SAMPLE_RATE)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self):
        self._load_whisper()
        self._loop()

    def stop(self):
        self._stop.set()

    def _load_whisper(self):
        try:
            from faster_whisper import WhisperModel
            print("[llm] loading Whisper 'tiny'...")
            self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("[llm] Whisper ready — say 'oye OTO' to activate")
        except Exception as e:
            print(f"[llm] ERROR loading Whisper: {e}")

    # ── Main loop (polls StateBus for raw audio) ───────────────────────────────

    def _loop(self):
        chunk_s = 1024 / self._rate   # approximate seconds per chunk

        while not self._stop.is_set():
            # Don't activate while LLM is already working
            if self.bus.get("state") in (MascotState.THINKING, MascotState.ANSWER):
                time.sleep(0.1)
                continue

            rms = self.bus.get("raw_rms", 0.0)
            if rms < VAD_THRESHOLD:
                time.sleep(0.02)
                continue

            # Voice detected — collect frames until silence
            frames, detected = self._collect_frames(MAX_WAKE_S)
            if not frames:
                continue

            text = self._transcribe(frames)
            if not text:
                continue
            print(f"[llm] heard: {text!r}")

            if not _is_wake_word(text):
                continue

            # ── Wake word confirmed ────────────────────────────────────────────
            print("[llm] wake word! listening for question...")
            time.sleep(0.4)  # brief pause before question

            q_frames, _ = self._collect_frames(MAX_QUESTION_S, wait_for_voice=True)
            if not q_frames:
                continue

            question = self._transcribe(q_frames)
            if not question:
                continue
            question = re.sub(r"(?i)oye\s+oto[,.]?\s*", "", question).strip()
            if len(question) < 2:
                continue

            self._handle_question(question)

    def _collect_frames(self, max_s: float, wait_for_voice: bool = False) -> tuple:
        """Poll StateBus for raw audio until silence. Returns (frames, started)."""
        frames = []
        silent_count = 0
        started = not wait_for_voice
        chunk_s = 1024 / self._rate
        silence_chunks = int(SILENCE_LIMIT / chunk_s)
        max_chunks = int(max_s / chunk_s)
        wait_chunks = int(3.0 / chunk_s)
        waited = 0

        last_raw = None

        for _ in range(max_chunks + wait_chunks):
            if self._stop.is_set():
                break

            raw = self.bus.get("raw_audio", b"")
            rms = self.bus.get("raw_rms", 0.0)

            # Skip duplicate chunks (bus hasn't updated yet)
            if raw == last_raw:
                time.sleep(0.01)
                continue
            last_raw = raw

            if not started:
                if rms >= VAD_THRESHOLD:
                    started = True
                else:
                    waited += 1
                    if waited >= wait_chunks:
                        break
                    continue

            frames.append(raw)

            if rms < VAD_THRESHOLD:
                silent_count += 1
                if silent_count >= silence_chunks:
                    break
            else:
                silent_count = 0

        return frames, started

    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, frames: list) -> str:
        if not self._whisper or not frames:
            return ""
        try:
            # Decode int16 PCM, resample 44100→16000 for Whisper
            pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
            if self._rate != 16000:
                ratio = 16000 / self._rate
                n_out = int(len(pcm) * ratio)
                pcm = np.interp(
                    np.linspace(0, len(pcm) - 1, n_out),
                    np.arange(len(pcm)),
                    pcm,
                )
            pcm_int16 = pcm.astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            with wave.open(path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(16000)
                wf.writeframes(pcm_int16.tobytes())

            segs, _ = self._whisper.transcribe(path, language="es", beam_size=2)
            text = " ".join(s.text for s in segs).strip()
            os.unlink(path)
            return text
        except Exception as e:
            print(f"[llm] transcribe error: {e}")
            return ""

    # ── LLM + TTS ─────────────────────────────────────────────────────────────

    def _handle_question(self, question: str):
        print(f"[llm] question: {question!r}")
        self.bus.update(state=MascotState.THINKING)
        response = self._ask_claude(question)
        if not response:
            self.bus.update(state=MascotState.IDLE)
            return
        print(f"[llm] response: {response!r}")
        self.bus.update(state=MascotState.ANSWER)
        self._speak(response)
        self.bus.update(state=MascotState.IDLE)

    def _ask_claude(self, text: str) -> str:
        try:
            msg = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            print(f"[llm] claude error: {e}")
            return ""

    def _speak(self, text: str):
        try:
            for model_path in [
                os.path.expanduser("~/.local/share/piper/es_ES-sharvard-medium.onnx"),
                os.path.expanduser("~/.local/share/piper/es_ES-davefx-medium.onnx"),
                os.path.expanduser("~/.local/share/piper/en_US-lessac-medium.onnx"),
            ]:
                if os.path.exists(model_path):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav_path = f.name
                    subprocess.run(
                        ["piper", "--model", model_path, "--output_file", wav_path],
                        input=text.encode(), capture_output=True, timeout=30,
                    )
                    subprocess.run(["aplay", wav_path], capture_output=True, timeout=60)
                    os.unlink(wav_path)
                    return
            # fallback
            subprocess.run(["espeak-ng", "-v", "es", "-s", "140", text], timeout=30)
        except Exception as e:
            print(f"[llm] TTS error: {e}")
