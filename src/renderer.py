import math
import random
import pygame
import cv2
import numpy as np
from state import MascotState, StateBus


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if   h < 60:  r,g,b = c,x,0
    elif h < 120: r,g,b = x,c,0
    elif h < 180: r,g,b = 0,c,x
    elif h < 240: r,g,b = 0,x,c
    elif h < 300: r,g,b = x,0,c
    else:          r,g,b = c,0,x
    return (int((r+m)*255), int((g+m)*255), int((b+m)*255))


class Particle:
    def __init__(self, x, y, vx, vy, color, life):
        self.x, self.y   = x, y
        self.vx, self.vy = vx, vy
        self.color        = color
        self.life         = life
        self.max_life     = life

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vx *= 0.96
        self.vy *= 0.96
        self.life -= 1

    @property
    def alpha(self):
        return self.life / self.max_life

    @property
    def alive(self):
        return self.life > 0


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
        self.clock  = pygame.time.Clock()

        # Blob shape
        self.N      = 128          # puntos del blob
        self.radii  = [0.0] * self.N   # radio actual suavizado por punto

        # Color
        self._hue    = 260.0      # arranca en violeta
        self._sat    = 0.8
        self._val    = 0.85

        # Partículas
        self.particles = []

        # Tiempo
        self._t = 0.0

        # Smooth bars
        self._bars    = [0.0] * 16
        self._volume  = 0.0
        self._bass    = 0.0
        self._mid     = 0.0
        self._high    = 0.0

        # Surface para alpha blending
        self.glow_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

    def _blob_points(self, cx, cy, t, bars, bass, mid, high, volume):
        pts = []
        base_r = min(self.W, self.H) * 0.22

        for i in range(self.N):
            angle = (i / self.N) * math.pi * 2
            bar_i = bars[int(i / self.N * len(bars))]

            # Capas de deformación: cada banda afecta a una frecuencia espacial distinta
            d  = 0.0
            d += bass  * 55  * math.sin(angle * 2  + t * 0.9)
            d += bass  * 30  * math.sin(angle * 3  - t * 0.7 + 0.5)
            d += mid   * 28  * math.sin(angle * 5  + t * 1.4)
            d += mid   * 18  * math.sin(angle * 7  - t * 1.8 + 1.0)
            d += high  * 14  * math.sin(angle * 11 + t * 2.8)
            d += high  * 8   * math.sin(angle * 17 - t * 3.5)
            d += bar_i * 22  * math.sin(angle * 4  + t * 1.1 + i * 0.1)
            d += volume * 10 * math.sin(angle * 9  + t * 2.0)

            # Radio target
            target_r = base_r + d

            # Smooth por punto
            self.radii[i] = lerp(self.radii[i], target_r, 0.12)
            r = max(10, self.radii[i])

            pts.append((cx + math.cos(angle) * r,
                        cy + math.sin(angle) * r))
        return pts

    def _emit_particles(self, pts, color, density=0.15):
        for pt in pts:
            if random.random() < density:
                speed = random.uniform(0.3, 2.5)
                angle = random.uniform(0, math.pi * 2)
                self.particles.append(Particle(
                    pt[0], pt[1],
                    math.cos(angle) * speed,
                    math.sin(angle) * speed,
                    color,
                    random.randint(20, 50)
                ))

    def draw_frame(self, snap):
        t       = self._t
        state   = snap["state"]
        bars    = snap["fft_bars"]
        volume  = snap["volume"]
        bass    = snap["bass"]
        mid     = snap["mid"]
        high    = snap["high"]
        face    = snap.get("face_detected", False)

        # Smooth audio
        sp = 0.15
        self._bars   = [lerp(self._bars[i],   bars[i], sp) for i in range(16)]
        self._volume = lerp(self._volume, volume, sp)
        self._bass   = lerp(self._bass,   bass,   sp)
        self._mid    = lerp(self._mid,    mid,    sp)
        self._high   = lerp(self._high,   high,   sp)

        # ── Color según estado y audio ─────────────────────────────────
        target_hue = {
            MascotState.IDLE:    260,   # violeta
            MascotState.AWARE:   200,   # cian
            MascotState.LISTEN:  160,   # verde-cian
            MascotState.TOUCH:   30,    # naranja
            MascotState.EXCITED: 0,     # rojo/rosa
        }.get(state, 260)

        # Audio empuja el hue: bass → rojo, high → azul
        target_hue += self._bass * 40 - self._high * 30
        self._hue    = lerp(self._hue, target_hue, 0.03)
        self._sat    = lerp(self._sat,  0.6 + self._volume * 0.4, 0.05)
        self._val    = lerp(self._val,  0.7 + self._volume * 0.3, 0.05)

        col_main  = hsv_to_rgb(self._hue,        self._sat, self._val)
        col_inner = hsv_to_rgb(self._hue + 30,   self._sat * 0.6, self._val * 0.5)
        col_outer = hsv_to_rgb(self._hue - 20,   self._sat * 0.4, self._val * 0.3)

        # ── Fondo: trail oscuro con persistencia ──────────────────────
        bg_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        bg_surf.fill((6, 6, 8, 180))   # alpha trail — cuanto más bajo más trail
        self.screen.blit(bg_surf, (0, 0))

        cx = self.W // 2
        cy = self.H // 2 + 10

        # ── Blob principal ────────────────────────────────────────────
        pts = self._blob_points(cx, cy, t, self._bars,
                                self._bass, self._mid, self._high, self._volume)

        # Glow exterior (3 capas)
        self.glow_surf.fill((0, 0, 0, 0))
        for layer in range(3, 0, -1):
            alpha = int(18 + self._volume * 30) * layer // 3
            offset = layer * 12
            off_pts = []
            for i, (px, py) in enumerate(pts):
                angle = (i / self.N) * math.pi * 2
                off_pts.append((px + math.cos(angle) * offset,
                                py + math.sin(angle) * offset))
            if len(off_pts) > 2:
                pygame.draw.polygon(self.glow_surf,
                    (*col_outer, alpha), off_pts)
        self.screen.blit(self.glow_surf, (0, 0))

        # Blob relleno
        if len(pts) > 2:
            pygame.draw.polygon(self.screen, col_inner, pts)

        # Contorno brillante
        pygame.draw.polygon(self.screen, col_main, pts, 2)

        # ── Partículas ────────────────────────────────────────────────
        if state == MascotState.EXCITED or self._bass > 0.5:
            self._emit_particles(pts[::8], col_main, density=0.3)
        elif self._volume > 0.1:
            self._emit_particles(pts[::16], col_main, density=0.05)

        self.particles = [p for p in self.particles if p.alive]
        for p in self.particles:
            p.update()
            alpha = int(p.alpha * 180)
            r     = max(1, int(p.alpha * 3))
            s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p.color, alpha), (r, r), r)
            self.screen.blit(s, (int(p.x)-r, int(p.y)-r))

        # ── Ondas interiores (alta frecuencia) ────────────────────────
        if self._high > 0.15:
            inner_r = min(self.W, self.H) * 0.08
            for ring in range(3):
                rr  = inner_r * (ring + 1) * (1 + self._high * 0.3)
                pts_r = []
                for i in range(48):
                    a   = (i / 48) * math.pi * 2
                    dr  = self._high * 8 * math.sin(a * 6 + t * 4 + ring)
                    pts_r.append((cx + math.cos(a) * (rr + dr),
                                  cy + math.sin(a) * (rr + dr)))
                alpha_r = int(self._high * 80)
                s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
                pygame.draw.polygon(s, (*col_main, alpha_r), pts_r, 1)
                self.screen.blit(s, (0, 0))

        # ── HUD minimal ───────────────────────────────────────────────
        font = pygame.font.SysFont(None, 22)
        hud = [
            (f"{state.name}", col_main),
            (f"FACE: {'YES' if face else 'NO'}", (80,80,80) if not face else (100,200,100)),
            (f"VOL {int(self._volume*100):3d}%", (60,60,60)),
        ]
        for i, (txt, c) in enumerate(hud):
            self.screen.blit(font.render(txt, True, c), (10, 10 + i*20))

        # ── Camera preview ────────────────────────────────────────────
        debug_frame = snap.get("debug_frame")
        if debug_frame is not None:
            try:
                h, w  = debug_frame.shape[:2]
                dw, dh = 140, int(h * 140 / w)
                small = cv2.resize(debug_frame, (dw, dh))
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(
                    np.transpose(small_rgb, (1, 0, 2)))
                dx = self.W - dw - 6
                dy = self.H - dh - 6
                pygame.draw.rect(self.screen, (25,25,25),
                    (dx-2, dy-2, dw+4, dh+4))
                self.screen.blit(surf, (dx, dy))
            except Exception:
                pass

        pygame.display.flip()

    def run(self):
        import time as _time
        running = True
        start   = _time.monotonic()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if _time.monotonic() - start > 2.0:
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
