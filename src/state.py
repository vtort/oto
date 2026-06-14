import threading
from enum import Enum, auto


class MascotState(Enum):
    IDLE   = auto()
    AWARE  = auto()
    LISTEN = auto()
    TOUCH  = auto()


class StateBus:
    """Thread-safe shared state between all sensor threads and renderer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            # Audio
            "volume":    0.0,   # 0.0–1.0 RMS
            "bass":      0.0,   # 0.0–1.0 band energy
            "mid":       0.0,
            "high":      0.0,
            "fft_bars":  [0.0] * 16,  # normalized bar values for EQ display

            # Vision
            "face_detected":  False,
            "face_x_norm":    0.5,    # 0.0=left, 1.0=right

            # Touch
            "touch_active":   False,
            "touch_pos":      (0, 0),

            # Vision debug
            "debug_frame":    None,

            # State machine output
            "state":          MascotState.IDLE,
            "state_color":    "#6366f1",
        }

    def update(self, **kwargs):
        with self._lock:
            self._data.update(kwargs)

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self):
        with self._lock:
            return dict(self._data)
