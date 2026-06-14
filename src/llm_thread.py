"""
LLM thread: push-to-talk → STT → Claude → TTS → ANSWER state

Touch screen to record, release to transcribe and respond.
Reads raw audio from StateBus (published by AudioThread).
"""
import threading
import time
import math
import os
import tempfile
import wave
import subprocess
import numpy as np
import anthropic

from state import MascotState, StateBus

SAMPLE_RATE  = 44100
CHANNELS     = 1
SAMPLE_WIDTH = 2   # paInt16

SYSTEM_PROMPT = (
    "Eres OTO, una mascota interactiva inteligente y cercana. "
    "Responde siempre en el mismo idioma que el usuario (castellano o catalán). "
    "Respuestas concisas, máximo 2-3 frases. Sin markdown, solo texto plano."
)


class LLMThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict):
        super().__init__(daemon=True, name="llm")
        self.bus     = bus
        self.cfg     = cfg
        self._stop   = threading.Event()
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._whisper       = None
        self._whisper_ready = threading.Event()
        self._rate = cfg.get("audio", {}).get("sample_rate", SAMPLE_RATE)

    def preload_whisper(self):
        """No-op kept for compatibility — loading is always async now."""
        pass

    def run(self):
        loader = threading.Thread(target=self._load_whisper, daemon=True, name="whisper-load")
        loader.start()
        self._loop()

    def stop(self):
        self._stop.set()

    def _load_whisper(self):
        try:
            from faster_whisper import WhisperModel
            print("[llm] loading Whisper 'tiny'...")
            self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("[llm] Whisper ready — tap screen to talk")
        except Exception as e:
            print(f"[llm] ERROR loading Whisper: {e}")
        finally:
            self._whisper_ready.set()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _loop(self):
        was_recording = False

        while not self._stop.is_set():
            recording = self.bus.get("recording", False)

            if recording and not was_recording:
                # Touch just started — collect frames until released
                frames = self._collect_while_recording()
                was_recording = False
                self.bus.update(recording=False)  # safety

                if not frames:
                    continue

                if not self._whisper_ready.wait(timeout=0.1):
                    print("[llm] Whisper still loading, ignoring tap")
                    continue
                if not self._whisper:
                    print("[llm] Whisper failed to load")
                    continue

                question = self._transcribe(frames)
                if not question or len(question.strip()) < 2:
                    print(f"[llm] nothing heard: {question!r}")
                    continue

                print(f"[llm] question: {question!r}")
                self._handle_question(question)

            was_recording = recording
            time.sleep(0.02)

    def _collect_while_recording(self) -> list:
        """Collect raw audio frames from StateBus while recording=True."""
        frames   = []
        last_raw = None

        while not self._stop.is_set():
            if not self.bus.get("recording", False):
                break
            raw = self.bus.get("raw_audio", b"")
            if raw and raw != last_raw:
                frames.append(raw)
                last_raw = raw
            time.sleep(0.005)

        return frames

    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, frames: list) -> str:
        if not self._whisper or not frames:
            return ""
        try:
            pcm = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
            if self._rate != 16000:
                n_out = int(len(pcm) * 16000 / self._rate)
                pcm   = np.interp(
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
        self.bus.update(state=MascotState.THINKING, heard_text=question, response_text="")
        response = self._ask_claude(question)
        if not response:
            self.bus.update(state=MascotState.IDLE, speaking_level=0.0)
            return
        print(f"[llm] response: {response!r}")
        self.bus.update(state=MascotState.ANSWER, response_text=response, stop_speaking=False)
        # TTS y animación en paralelo
        self._tts_proc = None
        tts = threading.Thread(target=self._speak, args=(response,), daemon=True)
        tts.start()
        self._simulate_speaking(response)
        if self._tts_proc and self._tts_proc.poll() is None:
            self._tts_proc.kill()
        tts.join(timeout=1)
        self.bus.update(state=MascotState.IDLE, speaking_level=0.0, stop_speaking=False)

    def _simulate_speaking(self, text: str):
        duration = max(2.0, len(text) / 14.0)
        start    = time.time()
        while time.time() - start < duration and not self._stop.is_set():
            if self.bus.get("stop_speaking", False):
                break
            t   = time.time() - start
            lvl = (0.5 + 0.5 * math.sin(t * 4.0 * math.pi)) * \
                  (0.6 + 0.4 * math.sin(t * 1.1 * math.pi))
            self.bus.update(speaking_level=float(lvl))
            time.sleep(0.033)

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
        import platform
        try:
            if platform.system() == "Darwin":
                self._tts_proc = subprocess.Popen(["say", "-v", "Mónica", text])
                self._tts_proc.wait()
                return
            # Linux: piper > espeak-ng
            for model_path in [
                os.path.expanduser("~/.local/share/piper/es_ES-sharvard-medium.onnx"),
                os.path.expanduser("~/.local/share/piper/es_ES-davefx-medium.onnx"),
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
            subprocess.run(["espeak-ng", "-v", "es", "-s", "140", text], timeout=30)
        except Exception as e:
            print(f"[llm] TTS error: {e}")
