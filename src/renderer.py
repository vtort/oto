import math
import numpy as np
import pygame
import moderngl
import cv2
from state import MascotState, StateBus


# ── Shaders ───────────────────────────────────────────────────────────────────

VERT = """
#version 300 es
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAG = """
#version 300 es
precision highp float;
in  vec2  v_uv;
out vec4  fragColor;

uniform vec2  u_res;

uniform vec4  u_ep0; uniform float u_ea0; uniform vec3 u_ec0; uniform float u_eph0;
uniform vec4  u_ep1; uniform float u_ea1; uniform vec3 u_ec1; uniform float u_eph1;
uniform vec4  u_ep2; uniform float u_ea2; uniform vec3 u_ec2; uniform float u_eph2;

// Blob orgánico: distorsión armónica en coordenadas polares.
// sin(θ×3) → 3 esquinas suaves; sin(θ×5) añade textura secundaria.
// El resultado rota con la elipse via `phase`.
float blobMask(vec4 ep, float ea, vec2 p, float phase) {
    vec2 d = p - ep.xy;
    float c = cos(ea), s = sin(ea);
    vec2 q = vec2(c*d.x + s*d.y, -s*d.x + c*d.y);
    vec2 n = q / ep.zw;
    float theta = atan(n.y, n.x);
    float r = length(n);
    float warp = 1.0
        + 0.11 * sin(theta * 3.0 + phase)
        + 0.04 * sin(theta * 5.0 + phase * 1.5);
    return 1.0 - smoothstep(-0.022, 0.022, r / warp - 1.0);
}

void main() {
    float asp = u_res.x / u_res.y;
    vec2 p = (v_uv * 2.0 - 1.0) * vec2(asp, 1.0);

    vec3 col = vec3(0.04, 0.04, 0.07);

    float m0 = blobMask(u_ep0, u_ea0, p, u_eph0);
    float m1 = blobMask(u_ep1, u_ea1, p, u_eph1);
    float m2 = blobMask(u_ep2, u_ea2, p, u_eph2);

    // Screen blend: blanco donde los 3 se solapan, colores mixtos donde 2 se solapan
    vec3 scr = vec3(0.0);
    scr = 1.0 - (1.0 - scr) * (1.0 - u_ec0 * m0);
    scr = 1.0 - (1.0 - scr) * (1.0 - u_ec1 * m1);
    scr = 1.0 - (1.0 - scr) * (1.0 - u_ec2 * m2);

    float any_ellipse = max(m0, max(m1, m2));
    col = mix(col, scr, any_ellipse);

    // Vignette suave
    vec2 vd = v_uv - 0.5;
    col *= 1.0 - dot(vd, vd) * 1.2;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# Overlay quad for HUD texture
HUD_VERT = """
#version 300 es
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

HUD_FRAG = """
#version 300 es
precision mediump float;
in  vec2      v_uv;
out vec4      fragColor;
uniform sampler2D u_tex;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""


# ── State configurations ───────────────────────────────────────────────────────

# RGB puro — screen blend de los 3 genera blanco en la intersección central
# R+G → amarillo, R+B → magenta, G+B → cian, R+G+B → blanco
ELLIPSE_COLORS = np.array([
    [1.00, 0.00, 0.00],   # Rojo
    [0.00, 1.00, 0.00],   # Verde
    [0.00, 0.00, 1.00],   # Azul
], dtype=np.float32)

N_ELLIPSES = 3

# Per state: 3 × (cx, cy, rx, ry, angle_base)
# Cada elipse separada ~120° del centro, muy juntas para que se solapen mucho
# Cada fila: (cx, cy, rx, ry, angle_base)
# Formas base compactas y casi redondas — el estiramiento solo viene del audio
_R = 0.30   # radio base
_O = 0.065  # offset centro desde origen

_S = {
    MascotState.IDLE: np.array([
        [ 0.000,  _O,    _R,    _R,    0.20],
        [-_O*0.87, -_O*0.50, _R, _R,  2.30],
        [ _O*0.87, -_O*0.50, _R, _R,  4.00],
    ], dtype=np.float32),

    MascotState.AWARE: np.array([
        [ 0.000,  _O,    _R+.01, _R+.01, 0.20],
        [-_O*0.87, -_O*0.50, _R+.01, _R+.01, 2.30],
        [ _O*0.87, -_O*0.50, _R+.01, _R+.01, 4.00],
    ], dtype=np.float32),

    MascotState.LISTEN: np.array([
        [ 0.000,  _O,    _R+.02, _R+.02, 0.20],
        [-_O*0.87, -_O*0.50, _R+.02, _R+.02, 2.30],
        [ _O*0.87, -_O*0.50, _R+.02, _R+.02, 4.00],
    ], dtype=np.float32),

    MascotState.TOUCH: np.array([
        [ 0.000,  _O,    _R+.02, _R+.02, 0.70],
        [-_O*0.87, -_O*0.50, _R+.02, _R+.02, 2.79],
        [ _O*0.87, -_O*0.50, _R+.02, _R+.02, 4.89],
    ], dtype=np.float32),

    MascotState.EXCITED: np.array([
        [ 0.000,  _O,    _R+.04, _R+.04, 0.00],
        [-_O*0.87, -_O*0.50, _R+.04, _R+.04, 2.09],
        [ _O*0.87, -_O*0.50, _R+.04, _R+.04, 4.19],
    ], dtype=np.float32),
}

def _lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, bus: StateBus, cfg: dict):
        self.bus = bus
        self.W   = cfg["display"]["width"]
        self.H   = cfg["display"]["height"]
        self.fps = cfg["display"]["fps"]

        # ── pygame + OpenGL window ─────────────────────────────────────
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if cfg["display"]["fullscreen"]:
            flags |= pygame.FULLSCREEN
        pygame.display.set_mode((self.W, self.H), flags)
        pygame.display.set_caption("OTO")
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        # ── ModernGL context ──────────────────────────────────────────
        self.ctx = moderngl.create_context(require=300)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── Main scene program ────────────────────────────────────────
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
        self.vbo  = self.ctx.buffer(quad)
        self.vao  = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

        # ── HUD overlay program ───────────────────────────────────────
        self.hud_prog = self.ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
        hud_verts = np.array([
            -1,-1, 0,1,
             1,-1, 1,1,
            -1, 1, 0,0,
             1, 1, 1,0,
        ], dtype=np.float32)
        self.hud_vbo = self.ctx.buffer(hud_verts)
        self.hud_vao = self.ctx.simple_vertex_array(
            self.hud_prog, self.hud_vbo, 'in_vert', 'in_uv')
        self.hud_tex = self.ctx.texture((self.W, self.H), 4)
        self.hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.hud_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self.hud_font = pygame.font.SysFont(None, 22)

        # Static uniform
        self.prog['u_res'].value = (float(self.W), float(self.H))

        # ── Smooth state ──────────────────────────────────────────────
        self._t        = 0.0
        self._ep       = _S[MascotState.IDLE].copy()
        self._n_ep     = N_ELLIPSES
        self._vol      = 0.0
        self._bass     = 0.0
        self._mid      = 0.0
        self._high     = 0.0
        self._bars     = [0.0] * 16
        self._rot_offs = np.zeros(N_ELLIPSES, dtype=np.float32)

    def _update_hud(self, state, face, volume, debug_frame):
        self.hud_surf.fill((0, 0, 0, 0))

        state_col = {
            MascotState.IDLE:    (100, 100, 180),
            MascotState.AWARE:   (80,  180, 240),
            MascotState.LISTEN:  (80,  220, 140),
            MascotState.TOUCH:   (240, 160, 60),
            MascotState.EXCITED: (240, 80,  80),
        }.get(state, (140, 140, 140))

        lines = [
            (state.name,                             state_col),
            (f"FACE: {'YES' if face else 'NO'}",     (80,200,80) if face else (60,60,60)),
            (f"VOL  {int(volume*100):3d}%",          (60,60,60)),
        ]
        for i, (txt, col) in enumerate(lines):
            surf = self.hud_font.render(txt, True, col)
            self.hud_surf.blit(surf, (10, 10 + i * 20))

        if debug_frame is not None:
            try:
                h, w   = debug_frame.shape[:2]
                dw, dh = 140, int(h * 140 / w)
                small  = cv2.resize(debug_frame, (dw, dh))
                rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                cam_s  = pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))
                pygame.draw.rect(self.hud_surf, (20,20,20,200),
                    (self.W-dw-8, self.H-dh-8, dw+4, dh+4))
                self.hud_surf.blit(cam_s, (self.W-dw-6, self.H-dh-6))
            except Exception:
                pass

        data = pygame.image.tostring(self.hud_surf, 'RGBA', False)
        self.hud_tex.write(data)

    def draw_frame(self, snap):
        t      = self._t
        state  = snap["state"]
        bars   = snap["fft_bars"]
        vol    = snap["volume"]
        bass   = snap["bass"]
        mid    = snap["mid"]
        high   = snap["high"]
        face   = snap.get("face_detected", False)
        dframe = snap.get("debug_frame")

        # Smooth audio
        sp = 0.12
        self._vol  = _lerp(self._vol,  vol,  sp)
        self._bass = _lerp(self._bass, bass, sp)
        self._mid  = _lerp(self._mid,  mid,  sp)
        self._high = _lerp(self._high, high, sp)
        self._bars = [_lerp(self._bars[i], bars[i], sp) for i in range(16)]

        # Target ellipse config
        target_ep = _S[state].copy()

        # Estiramiento solo cuando hay audio: bass escala, y la dirección varía por elipse
        bass_scale = 1.0 + self._bass * 0.18 + self._vol * 0.06
        stretch_dirs = [(1.0, 1.15), (1.15, 1.0), (1.0, 1.10)]  # rx/ry stretch distinto
        for i in range(N_ELLIPSES):
            sx, sy = stretch_dirs[i]
            target_ep[i, 2] *= 1.0 + (bass_scale - 1.0) * sx
            target_ep[i, 3] *= 1.0 + (bass_scale - 1.0) * sy

        # Lerp current → target
        lp = 0.06
        self._ep = self._ep + (target_ep - self._ep) * lp

        # Rotation: base angle per ellipse + time drift, faster when excited
        rot_speed = {
            MascotState.IDLE:    0.008,
            MascotState.AWARE:   0.014,
            MascotState.LISTEN:  0.022,
            MascotState.TOUCH:   0.030,
            MascotState.EXCITED: 0.055,
        }.get(state, 0.010)
        rot_speed += self._bass * 0.02
        self._rot_offs += rot_speed / self.fps

        # ── GL render ─────────────────────────────────────────────────
        self.ctx.clear(0.04, 0.04, 0.07, 1.0)

        asp = self.W / self.H
        for i in range(N_ELLIPSES):
            angle = self._ep[i, 4] + self._rot_offs[i]
            self.prog[f'u_ep{i}'].value = (
                float(self._ep[i,0]) * asp,
                float(self._ep[i,1]),
                float(self._ep[i,2]) * asp,
                float(self._ep[i,3]),
            )
            self.prog[f'u_ea{i}'].value = float(angle)
            self.prog[f'u_ec{i}'].value = tuple(ELLIPSE_COLORS[i].tolist())
            # Phase para la distorsión armónica — cada elipse usa una offset distinta
            self.prog[f'u_eph{i}'].value = float(angle * 1.0 + i * 2.094)

        self.vao.render(moderngl.TRIANGLE_STRIP)

        # ── HUD overlay ───────────────────────────────────────────────
        self._update_hud(state, face, self._vol, dframe)
        self.hud_tex.use(0)
        self.hud_prog['u_tex'].value = 0
        self.hud_vao.render(moderngl.TRIANGLE_STRIP)

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
                    pos = getattr(event, 'pos', None) or (
                        int(event.x * self.W), int(event.y * self.H))
                    self.bus.update(touch_active=True, touch_pos=pos)
                elif event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                    self.bus.update(touch_active=False)

            self._t += 1.0 / self.fps
            self.draw_frame(self.bus.snapshot())
            self.clock.tick(self.fps)

        pygame.quit()
