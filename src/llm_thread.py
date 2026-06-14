"""
LLMThread — STT → Claude (streaming) → TTS per sentence

Pipeline:
  speech_q (np.float32 16kHz) → mlx-whisper turbo (Metal)
  → Claude streaming → sentence split → say per sentence
  → speaking_level animation
"""
import threading
import queue
import time
import math
import re
import os
import subprocess
import numpy as np
import anthropic

from state import MascotState, StateBus

HISTORY_TURNS = 6      # keep last N user+assistant pairs
SENTENCE_RE   = re.compile(r'(?<=[.!?])\s+')

SYSTEM_PROMPT = (
    "Eres OTO, una mascota interactiva inteligente y cercana. "
    "Responde siempre en el mismo idioma que el usuario (castellano o catalán). "
    "Respuestas concisas, máximo 2-3 frases. Sin markdown, solo texto plano."
)


class LLMThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict, speech_q: queue.Queue):
        super().__init__(daemon=True, name="llm")
        self.bus      = bus
        self.cfg      = cfg
        self.speech_q = speech_q
        self._stop    = threading.Event()
        self._client  = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._history  = []    # [{"role": ..., "content": ...}, ...]
        self._tts_proc = None
        self._whisper  = None

    # kept for compat — no-op, loading is lazy on first use
    def preload_whisper(self):
        pass

    def run(self):
        self._load_whisper()
        self._loop()

    def stop(self):
        self._stop.set()
        if self._tts_proc and self._tts_proc.poll() is None:
            self._tts_proc.kill()

    # ── Whisper ───────────────────────────────────────────────────────

    def _load_whisper(self):
        try:
            import mlx_whisper
            self._whisper = mlx_whisper
            # Warm up — downloads model on first call (~400MB, cached after)
            print("[llm] loading mlx-whisper turbo (Metal)...")
            mlx_whisper.transcribe(
                np.zeros(RATE * 1, dtype=np.float32),
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language="es",
            )
            print("[llm] Whisper ready — press SPACE to talk")
        except Exception as e:
            print(f"[llm] mlx-whisper failed ({e}), falling back to faster-whisper")
            self._load_faster_whisper()

    def _load_faster_whisper(self):
        try:
            from faster_whisper import WhisperModel
            self._fw = WhisperModel("small", device="cpu", compute_type="int8")
            self._whisper = None
            print("[llm] faster-whisper ready")
        except Exception as e:
            print(f"[llm] STT unavailable: {e}")

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            if self._whisper:
                result = self._whisper.transcribe(
                    audio,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    language="es",
                    word_timestamps=False,
                )
                return result["text"].strip()
            elif hasattr(self, "_fw"):
                import tempfile, wave
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    path = f.name
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE)
                    wf.writeframes((audio * 32768).astype(np.int16).tobytes())
                segs, _ = self._fw.transcribe(path, language="es", beam_size=2)
                os.unlink(path)
                return " ".join(s.text for s in segs).strip()
        except Exception as e:
            print(f"[llm] transcribe error: {e}")
        return ""

    # ── Main loop ─────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            try:
                audio = self.speech_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # Don't start a new turn while already speaking
            if self.bus.get("state") in (MascotState.THINKING, MascotState.ANSWER):
                continue

            self.bus.update(state=MascotState.THINKING, heard_text="", response_text="")

            text = self._transcribe(audio)
            if not text or len(text.strip()) < 2:
                print(f"[llm] nothing heard: {text!r}")
                self.bus.update(state=MascotState.IDLE)
                continue

            text = text.strip()
            print(f"[llm] heard: {text!r}")
            self.bus.update(heard_text=text)

            self._answer(text)

    # ── Claude streaming → sentence TTS ───────────────────────────────

    def _answer(self, question: str):
        self._history.append({"role": "user", "content": question})
        if len(self._history) > HISTORY_TURNS * 2:
            self._history = self._history[-HISTORY_TURNS * 2:]

        self.bus.update(stop_speaking=False)

        try:
            sentence_buf = ""
            full_response = ""

            with self._client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=self._history,
            ) as stream:
                self.bus.update(state=MascotState.ANSWER)

                for token in stream.text_stream:
                    if self._stop.is_set() or self.bus.get("stop_speaking"):
                        break

                    sentence_buf  += token
                    full_response += token

                    # Speak as soon as we have a complete sentence
                    if SENTENCE_RE.search(sentence_buf) or \
                       (len(sentence_buf) > 120 and sentence_buf[-1] in ".,"):
                        parts = SENTENCE_RE.split(sentence_buf, maxsplit=1)
                        to_speak      = parts[0].strip()
                        sentence_buf  = parts[1] if len(parts) > 1 else ""
                        if to_speak:
                            self._speak_sentence(to_speak)
                            if self.bus.get("stop_speaking"):
                                break

                # Speak any remaining text
                if sentence_buf.strip() and not self.bus.get("stop_speaking"):
                    self._speak_sentence(sentence_buf.strip())

        except Exception as e:
            print(f"[llm] claude error: {e}")
            full_response = ""

        if full_response:
            self._history.append({"role": "assistant", "content": full_response})
            self.bus.update(response_text=full_response)
            print(f"[llm] response: {full_response!r}")

        self.bus.update(state=MascotState.IDLE, speaking_level=0.0, stop_speaking=False)

    def _speak_sentence(self, text: str):
        if self.bus.get("stop_speaking"):
            return

        # Animation pulse in parallel with TTS
        anim = threading.Thread(target=self._pulse_animation, args=(text,), daemon=True)
        anim.start()

        try:
            self._tts_proc = subprocess.Popen(["say", "-v", "Mónica", text])
            self._tts_proc.wait()
        except Exception as e:
            print(f"[llm] TTS error: {e}")

        anim.join(timeout=1)

    def _pulse_animation(self, text: str):
        duration = max(1.0, len(text) / 14.0)
        start = time.time()
        while time.time() - start < duration and not self._stop.is_set():
            if self.bus.get("stop_speaking"):
                break
            t   = time.time() - start
            lvl = (0.5 + 0.5 * math.sin(t * 4.0 * math.pi)) * \
                  (0.6 + 0.4 * math.sin(t * 1.1 * math.pi))
            self.bus.update(speaking_level=float(lvl))
            time.sleep(0.033)
        self.bus.update(speaking_level=0.0)

    def stop(self):
        self._stop.set()
        if self._tts_proc and self._tts_proc.poll() is None:
            self._tts_proc.kill()


RATE = 16000
