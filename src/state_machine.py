import time
from state import MascotState, StateBus


STATE_COLORS = {
    MascotState.IDLE:    "#6366f1",
    MascotState.AWARE:   "#38bdf8",
    MascotState.LISTEN:  "#4ade80",
    MascotState.TOUCH:   "#fb923c",
    MascotState.EXCITED: "#f43f5e",
}


class StateMachine:
    def __init__(self, bus: StateBus, cfg: dict):
        self.bus = bus
        self.cfg = cfg
        self._state = MascotState.IDLE
        self._state_since = time.monotonic()
        self._excited_until = 0.0

    def tick(self):
        snap = self.bus.snapshot()
        now  = time.monotonic()

        volume  = snap["volume"]
        bass    = snap["bass"]
        face    = snap["face_detected"]
        touch   = snap["touch_active"]
        idle_t  = self.cfg["states"]["idle_timeout_s"]
        cool_t  = self.cfg["states"]["excited_cooldown_s"]
        idle_th = self.cfg["audio"]["idle_threshold"]

        listen_th = idle_th * 15.0  # umbral alto — solo voz real, no ruido ambiente

        # Priority order (highest first)
        if touch:
            new = MascotState.TOUCH
        elif now < self._excited_until:
            new = MascotState.EXCITED
        elif bass > 0.85 or volume > 0.90:
            new = MascotState.EXCITED
            self._excited_until = now + cool_t
        elif face and volume > listen_th:
            new = MascotState.LISTEN   # cara + voz clara → LISTEN
        elif face:
            new = MascotState.AWARE    # cara detectada → AWARE (sin ruido de fondo)
        elif volume > listen_th:
            new = MascotState.LISTEN   # solo audio fuerte → LISTEN
        else:
            new = MascotState.IDLE

        if new != self._state:
            self._state = new
            self._state_since = now

        self.bus.update(state=self._state, state_color=STATE_COLORS[self._state])
