"""
AudioThread — sounddevice callbacks + Silero VAD

16kHz mono stream:
- Publishes volume/bass/mid/high/fft_bars to StateBus for animation
- When recording=True (spacebar held): collects audio + Silero VAD detects
  end-of-speech automatically → puts np.ndarray in speech_q for STT
"""
import threading
import queue
import numpy as np

import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator

from state import StateBus

RATE           = 16000
CHUNK_MS       = 30
CHUNK_SAMPLES  = int(RATE * CHUNK_MS / 1000)   # 480
N_FFT_BANDS    = 16
SILENCE_MS     = 600


def _fft_bands(chunk: np.ndarray, n: int = N_FFT_BANDS):
    win  = chunk * np.hanning(len(chunk))
    spec = np.abs(np.fft.rfft(win)) / (len(chunk) / 2 + 1e-9)
    return [float(np.mean(b)) for b in np.array_split(spec, n)]


class AudioThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict, speech_q: queue.Queue):
        super().__init__(daemon=True, name="audio")
        self.bus      = bus
        self.cfg      = cfg
        self.speech_q = speech_q
        self._stop    = threading.Event()
        self._raw_q   = queue.Queue(maxsize=200)

    def run(self):
        print("[audio] loading Silero VAD...")
        vad_model = load_silero_vad()
        vad = VADIterator(
            vad_model,
            sampling_rate=RATE,
            threshold=0.45,
            min_silence_duration_ms=SILENCE_MS,
            speech_pad_ms=80,
        )
        print("[audio] ready")

        def _cb(indata, frames, t, status):
            self._raw_q.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=RATE, channels=1, dtype="float32",
            blocksize=CHUNK_SAMPLES, callback=_cb,
        )
        stream.start()
        print(f"[audio] stream @ {RATE}Hz, {CHUNK_MS}ms chunks")

        speech_buf    = []
        was_recording = False

        while not self._stop.is_set():
            try:
                chunk = self._raw_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Animation ──────────────────────────────────────────────
            rms  = float(np.sqrt(np.mean(chunk ** 2)))
            bars = _fft_bands(chunk)
            spec = np.abs(np.fft.rfft(chunk)) / (CHUNK_SAMPLES / 2 + 1e-9)
            f    = np.fft.rfftfreq(CHUNK_SAMPLES, 1 / RATE)
            self.bus.update(
                volume   = min(rms * 4, 1.0),
                bass     = min(float(np.mean(spec[(f >= 20)  & (f < 250)])) * 40, 1.0),
                mid      = min(float(np.mean(spec[(f >= 250) & (f < 4000)])) * 20, 1.0),
                high     = min(float(np.mean(spec[(f >= 4000)])) * 20, 1.0),
                fft_bars = bars,
                raw_rms  = rms,
            )

            # ── Speech collection ──────────────────────────────────────
            recording = self.bus.get("recording", False)

            if recording:
                speech_buf.append(chunk)
                was_recording = True
                # Auto-stop when Silero detects end of speech
                result = vad(chunk)
                if result and "end" in result:
                    self._flush(speech_buf, vad)
                    speech_buf = []
                    self.bus.update(recording=False)

            elif was_recording:
                # Space released manually before VAD fired
                self._flush(speech_buf, vad)
                speech_buf    = []
                was_recording = False

        stream.stop()

    def _flush(self, buf: list, vad: VADIterator):
        if not buf:
            return
        audio = np.concatenate(buf)
        if len(audio) > RATE * 0.3:     # ignore clips < 300ms
            self.speech_q.put(audio)
        vad.reset_states()

    def stop(self):
        self._stop.set()
