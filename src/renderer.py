import math
import pygame
from state import MascotState, StateBus


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


class Renderer:
    def __init__(self, bus: StateBus, cfg: dict):
        self.bus  = bus
        self.cfg  = cfg
        self.W    = cfg["display"]["width"]
        self.H    = cfg["display"]["height"]
        self.fps  = cfg["display"]["fps"]

        pygame.init()
        flags = pygame.FULLSCREEN if cfg["display"]["fullscreen"] else 0
        self.screen = pygame.display.set_mode((self.W, self.H), flags)
        pygame.display.set_caption("OTO")
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        self.surf  = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        # Animated state
        self._color  = hex_to_rgb(cfg["mascot"]["color_idle"])
        self._breathe = 0.0
        self._bars   = [0.0] * 16
        self._blink_t = 0.0
        self._t       = 0.0
        self._antenna = [0.0, 0.0]   # phase per antenna

    def _ease(self, current, target, speed=0.1):
        return lerp(current, target, speed)

    def draw_frame(self, snap: dict):
        t      = self._t
        state  = snap["state"]
        col_h  = snap["state_color"]
        col    = hex_to_rgb(col_h)
        volume = snap["volume"]
        bass   = snap["bass"]
        bars   = snap["fft_bars"]

        # Smooth color transition
        self._color = lerp_color(self._color, col, 0.05)
        c = self._color

        # Breathing speed per state
        bspeed = {
            MascotState.IDLE:    0.8,
            MascotState.AWARE:   1.2,
            MascotState.LISTEN:  1.5 + volume,
            MascotState.TOUCH:   2.0,
            MascotState.EXCITED: 3.0 + bass * 2,
        }.get(state, 1.0)
        self._breathe = math.sin(t * bspeed) * 0.5 + 0.5

        # Smooth bars
        self._bars = [lerp(self._bars[i], bars[i], 0.2) for i in range(16)]

        bg = (8, 8, 10)
        self.screen.fill(bg)

        cx, cy = self.W // 2, self.H // 2
        mcfg   = self.cfg["mascot"]
        base_r = mcfg["body_radius"]

        # Body radius pulses with bass in excited/listen states
        pulse = 0.0
        if state in (MascotState.LISTEN, MascotState.EXCITED):
            pulse = self._bars[1] * 18 + bass * 12
        body_r = int(base_r + self._breathe * 4 + pulse)

        # ── Glow ──────────────────────────────────────────────
        glow_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for ring in range(5, 0, -1):
            alpha = int(18 * ring * (0.5 + bass * 0.5))
            r_glow = body_r + ring * 14
            pygame.draw.circle(glow_surf, (*c, alpha), (cx, cy), r_glow)
        self.screen.blit(glow_surf, (0, 0))

        # ── EQ bars ───────────────────────────────────────────
        n_bars  = 16
        bar_w   = 12
        gap     = 6
        total_w = n_bars * bar_w + (n_bars - 1) * gap
        bx0     = (self.W - total_w) // 2
        max_bh  = 60
        bar_y   = self.H - 50

        for i, v in enumerate(self._bars):
            bh  = int(v * max_bh)
            bx  = bx0 + i * (bar_w + gap)
            # Background track
            pygame.draw.rect(self.screen, (20, 20, 24), (bx, bar_y - max_bh, bar_w, max_bh), border_radius=3)
            if bh > 0:
                alpha_bar = int(80 + v * 140)
                bar_col   = (*c, alpha_bar)
                s = pygame.Surface((bar_w, bh), pygame.SRCALPHA)
                s.fill(bar_col)
                self.screen.blit(s, (bx, bar_y - bh))
            # Top pip
            pygame.draw.rect(self.screen, c, (bx, bar_y - bh - 2, bar_w, 2), border_radius=1)

        # ── Body circle ───────────────────────────────────────
        pygame.draw.circle(self.screen, (18, 18, 22), (cx, cy), body_r)
        pygame.draw.circle(self.screen, c, (cx, cy), body_r, 2)

        # ── Eyes ──────────────────────────────────────────────
        self._blink_t += 1 / self.fps
        blinking = math.sin(self._blink_t * 0.25) > 0.97

        eye_spread = 28 if state == MascotState.AWARE else 24
        ey = cy - 10

        for side in (-1, 1):
            ex = cx + side * eye_spread

            if blinking and state != MascotState.EXCITED:
                pygame.draw.line(self.screen, c, (ex - 9, ey), (ex + 9, ey), 3)
            else:
                eye_h = {
                    MascotState.IDLE:    8,
                    MascotState.AWARE:   13,
                    MascotState.LISTEN:  11,
                    MascotState.TOUCH:   10,
                    MascotState.EXCITED: 15,
                }.get(state, 10)

                # Eye white
                pygame.draw.ellipse(self.screen, c,
                    (ex - 9, ey - eye_h, 18, eye_h * 2))
                # Pupil
                pygame.draw.ellipse(self.screen, (10, 10, 14),
                    (ex - 4, ey - eye_h // 2, 8, eye_h))
                # Shine
                pygame.draw.circle(self.screen, (255, 255, 255), (ex + 3, ey - eye_h // 3), 2)

        # ── Mouth ─────────────────────────────────────────────
        my = cy + 18
        if state == MascotState.EXCITED:
            pts = [(cx + int(math.cos(a) * 18), my + int(math.sin(a) * 10))
                   for a in [math.radians(x) for x in range(10, 171, 10)]]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, c, False, pts, 3)
        else:
            curve = 4 if state in (MascotState.LISTEN, MascotState.AWARE) else 2
            pts = [(cx + int(math.cos(a) * 16), my + int(math.sin(a) * curve + 2))
                   for a in [math.radians(x) for x in range(10, 171, 15)]]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, c, False, pts, 2)

        # ── Antennae (LISTEN / EXCITED) ───────────────────────
        if state in (MascotState.LISTEN, MascotState.EXCITED):
            for i, side in enumerate((-1, 1)):
                self._antenna[i] += 0.08 * (2 if state == MascotState.EXCITED else 1)
                wave = math.sin(self._antenna[i] + side * 1.5) * (8 if state == MascotState.EXCITED else 5)
                ax   = cx + side * (body_r - 20)
                ay   = cy - body_r + 2
                tx   = ax + side * 14 + int(wave)
                ty   = ay - 28
                pygame.draw.line(self.screen, (*c, 160), (ax, ay), (tx, ty), 2)
                pygame.draw.circle(self.screen, c, (tx, ty), 5)
                pygame.draw.circle(self.screen, (10, 10, 14), (tx, ty), 3)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                    pos = getattr(event, "pos", None) or (
                        int(event.x * self.W), int(event.y * self.H))
                    self.bus.update(touch_active=True, touch_pos=pos)
                elif event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                    self.bus.update(touch_active=False)

            self._t += 1.0 / self.fps
            snap = self.bus.snapshot()
            self.draw_frame(snap)
            self.clock.tick(self.fps)

        pygame.quit()
