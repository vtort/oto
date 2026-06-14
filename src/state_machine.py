import time
from state import MascotState, StateBus


STATE_COLORS = {
    MascotState.IDLE:     "#6366f1",
    MascotState.AWARE:    "#38bdf8",
    MascotState.LISTEN:   "#4ade80",
    MascotState.TOUCH:    "#fb923c",
    MascotState.THINKING: "#c084fc",
    MascotState.ANSWER:   "#fbbf24",
}

# States driven externally by LLMThread — state machine must not override them
LLM_STATES = {MascotState.THINKING, MascotState.ANSWER}

DEMO_SEQUENCE = [MascotState.IDLE, MascotState.AWARE, MascotState.LISTEN, MascotState.TOUCH, MascotState.THINKING]
DEMO_DURATION = 6.0

# How long a state must be "wanted" before we commit to it (debounce)
# Prevents flickering between states on transient signals
DEBOUNCE = {
    MascotState.IDLE:   0.6,  # stay in active states a bit before going idle
    MascotState.AWARE:  0.3,
    MascotState.LISTEN: 0.0,  # immediate — voice is real-time
    MascotState.TOUCH:  0.0,  # immediate — finger is down
}


class StateMachine:
    def __init__(self, bus: StateBus, cfg: dict, demo: bool = False):
        self.bus  = bus
        self.cfg  = cfg
        self.demo = demo
        self._state        = MascotState.IDLE
        self._state_since  = time.monotonic()
        self._pending      = MascotState.IDLE
        self._pending_since = time.monotonic()
        self._demo_idx  = 0
        self._demo_next = time.monotonic() + DEMO_DURATION

    def tick(self):
        if self.demo:
            now = time.monotonic()
            if now >= self._demo_next:
                self._demo_idx = (self._demo_idx + 1) % len(DEMO_SEQUENCE)
                self._demo_next = now + DEMO_DURATION
            state = DEMO_SEQUENCE[self._demo_idx]
            self.bus.update(state=state, state_color=STATE_COLORS.get(state, "#c084fc"))
            return

        snap = self.bus.snapshot()
        now  = time.monotonic()

        # LLM thread owns these states — don't override
        if snap["state"] in LLM_STATES:
            self._state = snap["state"]
            self.bus.update(state_color=STATE_COLORS[self._state])
            return

        volume   = snap["volume"]
        touch    = snap["touch_active"]
        face     = snap["face_detected"]
        idle_th  = self.cfg["audio"]["idle_threshold"]
        listen_th = idle_th * 35.0

        if snap.get("recording"):
            wanted = MascotState.LISTEN
        elif touch:
            wanted = MascotState.TOUCH
        elif face:
            wanted = MascotState.AWARE
        else:
            wanted = MascotState.IDLE

        # Track pending state change
        if wanted != self._pending:
            self._pending = wanted
            self._pending_since = now

        # Commit only after debounce period
        debounce = DEBOUNCE.get(wanted, 0.3)
        if wanted != self._state and (now - self._pending_since) >= debounce:
            self._state       = wanted
            self._state_since = now

        self.bus.update(state=self._state, state_color=STATE_COLORS.get(self._state, "#6366f1"))
