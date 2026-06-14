"""
LLM thread: wake word → STT → Claude → TTS → ANSWER state
"""
import threading
import time
import os
import tempfile
import wave
import subprocess
import numpy as np
import pyaudio
import anthropic

from state import MascotState, StateBus

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHANNELS       = 1
CHUNK          = 1280          # 80ms @ 16kHz — openWakeWord expects this
FORMAT         = pyaudio.paInt16

WAKE_WORD      = "hey jarvis"  # modelo más fiable disponible; configurable
SILENCE_LIMIT  = 1.5           # segundos de silencio para cortar grabación
MAX_RECORD_S   = 15            # máximo de grabación por seguridad
VAD_THRESHOLD  = 300           # RMS para considerar "hay voz"

SYSTEM_PROMPT  = (
    "Eres OTO, una mascota interactiva inteligente y cercana. "
    "Responde siempre en el mismo idioma que el usuario. "
    "Respuestas concisas, máximo 3 frases. Sin markdown, solo texto plano."
)


class LLMThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict):
        super().__init__(daemon=True, name="llm")
        self.bus   = bus
        self.cfg   = cfg
        self._stop = threading.Event()
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._pa = None
        self._oww = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            import openwakeword
            from openwakeword.model import Model
            openwakeword.utils.download_models(["hey_jarvis"])
            self._oww = Model(wakeword_models=["hey_jarvis"], inference_framework="tflite")
            print("[llm] openWakeWord loaded — say 'hey jarvis' to activate")
        except Exception as e:
            print(f"[llm] WARNING: openWakeWord failed ({e}), falling back to VAD-only mode")
            self._oww = None

        self._pa = pyaudio.PyAudio()

        try:
            self._loop()
        finally:
            self._pa.terminate()

    def stop(self):
        self._stop.set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        stream = self._pa.open(
            rate=SAMPLE_RATE, channels=CHANNELS,
            format=FORMAT, input=True,
            frames_per_buffer=CHUNK,
        )
        print("[llm] listening for wake word...")

        while not self._stop.is_set():
            # Don't activate if OTO is already busy
            state = self.bus.get("state")
            if state in (MascotState.THINKING, MascotState.ANSWER):
                time.sleep(0.1)
                continue

            chunk = stream.read(CHUNK, exception_on_overflow=False)

            if self._oww is not None:
                audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                self._oww.predict(audio_np)
                scores = self._oww.prediction_buffer.get("hey_jarvis", [0])
                triggered = len(scores) > 0 and scores[-1] > 0.5
            else:
                # fallback: loud sound triggers
                rms = np.sqrt(np.mean(np.frombuffer(chunk, dtype=np.int16).astype(np.float32)**2))
                triggered = rms > VAD_THRESHOLD * 3

            if triggered:
                print("[llm] wake word detected!")
                stream.stop_stream()
                self._handle_conversation(stream)
                stream.start_stream()
                print("[llm] listening for wake word...")

        stream.stop_stream()
        stream.close()

    # ── Conversation pipeline ─────────────────────────────────────────────────

    def _handle_conversation(self, stream):
        # 1. Record until silence
        print("[llm] recording...")
        frames = self._record_until_silence(stream)
        if not frames:
            return

        # 2. STT with faster-whisper
        text = self._transcribe(frames)
        if not text or len(text.strip()) < 3:
            print("[llm] transcription empty, ignoring")
            return
        print(f"[llm] heard: {text!r}")

        # 3. LLM → THINKING state while we wait
        self.bus.update(state=MascotState.THINKING)
        response = self._ask_claude(text)
        if not response:
            self.bus.update(state=MascotState.IDLE)
            return
        print(f"[llm] response: {response!r}")

        # 4. TTS → ANSWER state while speaking
        self.bus.update(state=MascotState.ANSWER)
        self._speak(response)

        # 5. Back to IDLE
        self.bus.update(state=MascotState.IDLE)

    def _record_until_silence(self, stream) -> list:
        frames = []
        silent_chunks = 0
        max_chunks = int(MAX_RECORD_S * SAMPLE_RATE / CHUNK)
        silence_chunks_needed = int(SILENCE_LIMIT * SAMPLE_RATE / CHUNK)
        started = False

        for _ in range(max_chunks):
            if self._stop.is_set():
                break
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms  = np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float32)**2))

            if rms > VAD_THRESHOLD:
                started = True
                silent_chunks = 0
                frames.append(data)
            elif started:
                frames.append(data)
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed:
                    break

        return frames if started else []

    def _transcribe(self, frames: list) -> str:
        try:
            from faster_whisper import WhisperModel
            if not hasattr(self, "_whisper"):
                print("[llm] loading Whisper model (first time)...")
                self._whisper = WhisperModel("small", device="cpu", compute_type="int8")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            with wave.open(path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self._pa.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            segments, _ = self._whisper.transcribe(path, language=None)
            text = " ".join(s.text for s in segments).strip()
            os.unlink(path)
            return text
        except Exception as e:
            print(f"[llm] transcribe error: {e}")
            return ""

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
            # Try piper TTS first
            model_path = os.path.expanduser("~/.local/share/piper/es_ES-sharvard-medium.onnx")
            if not os.path.exists(model_path):
                model_path = os.path.expanduser("~/.local/share/piper/en_US-lessac-medium.onnx")

            if os.path.exists(model_path):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                subprocess.run(
                    ["piper", "--model", model_path, "--output_file", wav_path],
                    input=text.encode(), capture_output=True, timeout=30
                )
                subprocess.run(["aplay", wav_path], capture_output=True, timeout=60)
                os.unlink(wav_path)
            else:
                # fallback: espeak
                subprocess.run(["espeak-ng", "-v", "es", text], timeout=30)
        except Exception as e:
            print(f"[llm] TTS error: {e}")
