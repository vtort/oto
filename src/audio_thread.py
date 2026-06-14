import threading
import numpy as np

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False

from state import StateBus


class AudioThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict):
        super().__init__(daemon=True, name="audio")
        self.bus   = bus
        self.cfg   = cfg["audio"]
        self._stop = threading.Event()

    def run(self):
        if not PYAUDIO_OK:
            print("[audio] pyaudio not available — running silent")
            return

        pa     = pyaudio.PyAudio()
        rate   = self.cfg["sample_rate"]
        chunk  = self.cfg["chunk_size"]
        device = self.cfg.get("device_index")  # None = default

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=device,
            frames_per_buffer=chunk,
        )

        bands  = self.cfg["fft_bands"]
        freqs  = np.fft.rfftfreq(chunk, d=1.0 / rate)
        n_bars = 16

        def band_energy(fft_mag, lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.mean(fft_mag[mask])) if mask.any() else 0.0

        peak = 1e-6  # adaptive peak for normalization

        while not self._stop.is_set():
            try:
                raw  = stream.read(chunk, exception_on_overflow=False)
                pcm  = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                pcm  /= 32768.0

                rms  = float(np.sqrt(np.mean(pcm ** 2)))

                fft  = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm))))
                peak = max(peak * 0.995, fft.max() + 1e-9)
                fft_n = fft / peak

                bass = band_energy(fft_n, *bands["bass"])
                mid  = band_energy(fft_n, *bands["mid"])
                high = band_energy(fft_n, *bands["high"])

                # 16 log-spaced bars across full spectrum
                log_edges = np.logspace(np.log10(20), np.log10(20000), n_bars + 1)
                bars = [band_energy(fft_n, log_edges[i], log_edges[i+1]) for i in range(n_bars)]
                # Progressive gain: lows×3, mids×5, highs×12 — compensates mic rolloff
                gains = np.linspace(3.0, 12.0, n_bars)
                bars = [min(1.0, max(0.03, b * gains[i])) for i, b in enumerate(bars)]

                self.bus.update(
                    volume=min(1.0, rms * 20),
                    bass=min(1.0, bass * 4),
                    mid=min(1.0, mid * 4),
                    high=min(1.0, high * 4),
                    fft_bars=bars,
                    raw_audio=raw,
                    raw_rms=rms,
                )

            except Exception as e:
                print(f"[audio] error: {e}")

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def stop(self):
        self._stop.set()
