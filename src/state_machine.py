import time
from state import MascotState, StateBus


STATE_COLORS = {
    MascotState.IDLE:   "#6366f1",
    MascotState.LISTEN: "#4ade80",
    MascotState.TOUCH:  "#fb923c",
}

DEMO_SEQUENCE = [MascotState.IDLE, MascotState.LISTEN]


class StateMachine:
    def __init__(self, bus: StateBus, cfg: dict, demo: bool = False):
        self.bus = bus
        self.cfg = cfg
        self.demo = demo
        self._state = MascotState.IDLE
        self._state_since = time.monotonic()
        self._demo_idx = 0
        self._demo_next = time.monotonic() + 4.0

    def tick(self):
        if self.demo:
            now = time.monotonic()
            if now >= self._demo_next:
                self._demo_idx = (self._demo_idx + 1) % len(DEMO_SEQUENCE)
                self._demo_next = now + 4.0
            state = DEMO_SEQUENCE[self._demo_idx]
            self.bus.update(state=state, state_color=STATE_COLORS[state])
            return

        snap = self.bus.snapshot()
        now  = time.monotonic()

        volume = snap["volume"]
        bass   = snap["bass"]
        touch  = snap["touch_active"]
        idle_th = self.cfg["audio"]["idle_threshold"]
        listen_th = idle_th * 15.0  # solo voz real, no ruido ambiente

        if touch:
            new = MascotState.TOUCH
        elif bass > 0.85 or volume > listen_th:
            new = MascotState.LISTEN
        else:
            new = MascotState.IDLE

        if new != self._state:
            self._state = new
            self._state_since = now

        self.bus.update(state=self._state, state_color=STATE_COLORS[self._state])
