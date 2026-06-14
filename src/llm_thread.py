"""
LLM thread: VAD+Whisper wake word → STT → Claude → TTS → ANSWER state

Flow:
  1. Always listen with VAD (chunk-level RMS)
  2. When speech detected, record until silence → transcribe short clip
  3. If transcription matches "oye oto" (fuzzy) → record full question
  4. THINKING → Claude haiku → ANSWER → piper TTS → IDLE
"""
import threading
import time
import os
import tempfile
import wave
import subprocess
import re
import numpy as np
import pyaudio
import anthropic

from state import MascotState, StateBus

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK         = 1024          # ~64ms chunks for VAD
FORMAT        = pyaudio.paInt16

VAD_THRESHOLD = 250           # RMS to consider "voice present" — tune if needed
SILENCE_LIMIT = 1.2           # seconds of silence to end recording
MAX_RECORD_S  = 4             # max for wake word clip
MAX_QUESTION_S = 15           # max for full question

# ── Wake word variants (whisper sometimes adds punctuation or varies) ──────────
WAKE_PATTERNS = [
    r"oye\s+oto",
    r"ey\s+oto",
    r"hey\s+oto",
    r"oi\s+oto",       # Catalan variant
    r"oye\s+otto",
    r"hey\s+otto",
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
        self._pa      = None
        self._whisper = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self):
        self._pa = pyaudio.PyAudio()
        self._load_whisper()
        try:
            self._loop()
        finally:
            self._pa.terminate()

    def stop(self):
        self._stop.set()

    def _load_whisper(self):
        try:
            from faster_whisper import WhisperModel
            print("[llm] loading Whisper 'small' model...")
            self._whisper = WhisperModel("small", device="cpu", compute_type="int8")
            print("[llm] Whisper ready — say 'oye OTO' to activate")
        except Exception as e:
            print(f"[llm] ERROR loading Whisper: {e}")

    # ── Main VAD loop ──────────────────────────────────────────────────────────

    def _loop(self):
        stream = self._pa.open(
            rate=SAMPLE_RATE, channels=CHANNELS,
            format=FORMAT, input=True,
            frames_per_buffer=CHUNK,
        )

        while not self._stop.is_set():
            # Don't activate while LLM is already working
            if self.bus.get("state") in (MascotState.THINKING, MascotState.ANSWER):
                time.sleep(0.1)
                stream.read(CHUNK, exception_on_overflow=False)  # drain
                continue

            data = stream.read(CHUNK, exception_on_overflow=False)
            rms  = _rms(data)

            if rms < VAD_THRESHOLD:
                continue

            # Voice detected — record until silence (short clip for wake word)
            frames = [data] + self._record_until_silence(stream, MAX_RECORD_S)
            if not frames:
                continue

            text = self._transcribe(frames)
            print(f"[llm] heard: {text!r}")

            if not _is_wake_word(text):
                continue

            # ── Wake word confirmed ────────────────────────────────────────────
            print("[llm] wake word! recording question...")
            # Small pause so user can start speaking the question
            time.sleep(0.3)

            # Wait for voice to start, then record the question
            q_frames = self._wait_and_record(stream, MAX_QUESTION_S)
            if not q_frames:
                continue

            question = self._transcribe(q_frames)
            # Strip the wake word if it bled into the question clip
            question = re.sub(r"(?i)oye\s+oto[,.]?\s*", "", question).strip()
            if len(question) < 2:
                continue

            stream.stop_stream()
            self._handle_question(question)
            stream.start_stream()
            print("[llm] listening for 'oye OTO'...")

        stream.stop_stream()
        stream.close()

    # ── Recording helpers ──────────────────────────────────────────────────────

    def _record_until_silence(self, stream, max_s: float) -> list:
        frames = []
        silent = 0
        silence_chunks = int(SILENCE_LIMIT * SAMPLE_RATE / CHUNK)
        max_chunks     = int(max_s * SAMPLE_RATE / CHUNK)

        for _ in range(max_chunks):
            if self._stop.is_set():
                break
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if _rms(data) < VAD_THRESHOLD:
                silent += 1
                if silent >= silence_chunks:
                    break
            else:
                silent = 0
        return frames

    def _wait_and_record(self, stream, max_s: float) -> list:
        """Wait up to 3s for voice to start, then record until silence."""
        wait_chunks = int(3.0 * SAMPLE_RATE / CHUNK)
        for _ in range(wait_chunks):
            if self._stop.is_set():
                return []
            data = stream.read(CHUNK, exception_on_overflow=False)
            if _rms(data) >= VAD_THRESHOLD:
                return [data] + self._record_until_silence(stream, max_s)
        return []

    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, frames: list) -> str:
        if not self._whisper or not frames:
            return ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            with wave.open(path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self._pa.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))
            segs, _ = self._whisper.transcribe(path, language="es", beam_size=2)
            text = " ".join(s.text for s in segs).strip()
            os.unlink(path)
            return text
        except Exception as e:
            print(f"[llm] transcribe error: {e}")
            return ""

    # ── LLM + TTS pipeline ────────────────────────────────────────────────────

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
            # piper TTS — Spanish voice preferred
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

            # fallback: espeak-ng
            subprocess.run(["espeak-ng", "-v", "es", "-s", "140", text], timeout=30)
        except Exception as e:
            print(f"[llm] TTS error: {e}")


def _rms(data: bytes) -> float:
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
