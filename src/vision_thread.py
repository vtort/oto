import threading
import os
import cv2
import numpy as np

from state import StateBus

# Busca el cascade en varias ubicaciones según distro
def _find_cascade():
    candidates = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    try:
        candidates.insert(0, cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except AttributeError:
        pass
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("haarcascade_frontalface_default.xml no encontrado")

FACE_CASCADE = cv2.CascadeClassifier(_find_cascade())


class VisionThread(threading.Thread):
    def __init__(self, bus: StateBus, cfg: dict):
        super().__init__(daemon=True, name="vision")
        self.bus    = bus
        self.cfg    = cfg.get("vision", {})
        self._stop  = threading.Event()

        self.device     = self.cfg.get("device_index", 0)
        self.width      = self.cfg.get("capture_width", 320)   # capturamos pequeño para ir rápido
        self.height     = self.cfg.get("capture_height", 240)
        self.fps_target = self.cfg.get("fps", 10)              # 10fps es suficiente para detección
        self.scale      = self.cfg.get("detect_scale", 1.2)
        self.neighbors  = self.cfg.get("detect_neighbors", 4)

    def run(self):
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # no acumular frames viejos

        if not cap.isOpened():
            print("[vision] ERROR: no se puede abrir la cámara")
            return

        print(f"[vision] cámara abierta en /dev/video{self.device} @ {self.width}×{self.height}")

        frame_interval = 1.0 / self.fps_target
        import time
        last = 0.0

        while not self._stop.is_set():
            now = time.monotonic()
            if now - last < frame_interval:
                time.sleep(0.005)
                continue
            last = now

            ret, frame = cap.read()
            if not ret:
                continue

            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray   = cv2.equalizeHist(gray)   # mejora detección con poca luz

            faces  = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor  = self.scale,
                minNeighbors = self.neighbors,
                minSize      = (40, 40),
            )

            if len(faces) > 0:
                # La cara más grande = la más cercana
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx_norm = (x + w / 2) / self.width   # 0=izquierda, 1=derecha
                self.bus.update(
                    face_detected=True,
                    face_x_norm=float(cx_norm),
                )
            else:
                self.bus.update(face_detected=False)

        cap.release()
        print("[vision] cámara cerrada")

    def stop(self):
        self._stop.set()
