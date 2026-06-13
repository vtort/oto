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

uniform vec4  u_ep0; uniform float u_ea0; uniform vec3 u_ec0;
uniform vec4  u_ep1; uniform float u_ea1; uniform vec3 u_ec1;
uniform vec4  u_ep2; uniform float u_ea2; uniform vec3 u_ec2;
uniform vec4  u_ep3; uniform float u_ea3; uniform vec3 u_ec3;
uniform vec4  u_ep4; uniform float u_ea4; uniform vec3 u_ec4;
uniform vec4  u_ep5; uniform float u_ea5; uniform vec3 u_ec5;

uniform vec4  u_blob;
uniform float u_bn;

float sdEllipse(vec2 p, vec2 cen, vec2 r, float ang) {
    vec2 d = p - cen;
    float c = cos(ang), s = sin(ang);
    vec2 q = vec2(c*d.x + s*d.y, -s*d.x + c*d.y);
    return length(q / r) - 1.0;
}

vec3 screenBlend(vec3 acc, vec4 ep, float ea, vec3 ec, vec2 p) {
    float d = sdEllipse(p, ep.xy, ep.zw, ea);
    float m = 1.0 - smoothstep(-0.025, 0.025, d);
    return 1.0 - (1.0 - acc) * (1.0 - ec * m);
}

void main() {
    float asp = u_res.x / u_res.y;
    vec2 p = (v_uv * 2.0 - 1.0) * vec2(asp, 1.0);

    vec3 col = vec3(0.04, 0.04, 0.07);

    vec3 scr = vec3(0.0);
    scr = screenBlend(scr, u_ep0, u_ea0, u_ec0, p);
    scr = screenBlend(scr, u_ep1, u_ea1, u_ec1, p);
    scr = screenBlend(scr, u_ep2, u_ea2, u_ec2, p);
    scr = screenBlend(scr, u_ep3, u_ea3, u_ec3, p);
    scr = screenBlend(scr, u_ep4, u_ea4, u_ec4, p);
    scr = screenBlend(scr, u_ep5, u_ea5, u_ec5, p);

    float presence = (scr.r + scr.g + scr.b) * 0.333;
    col = mix(col, scr, clamp(presence * 3.0, 0.0, 1.0));

    // White central superellipse blob
    vec2 bq = (p - u_blob.xy) / u_blob.zw;
    float bf = pow(abs(bq.x), u_bn) + pow(abs(bq.y), u_bn) - 1.0;
    float bm = 1.0 - smoothstep(-0.04, 0.04, bf);
    col = mix(col, vec3(1.0), bm);

    vec2 vd = v_uv - 0.5;
    col *= 1.0 - dot(vd, vd) * 1.4;

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

# Colors: blue, red, green, orange, cyan, pink — matching the Figma palette
ELLIPSE_COLORS = np.array([
    [0.00, 0.47, 1.00],
    [1.00, 0.15, 0.12],
    [0.10, 0.82, 0.30],
    [1.00, 0.55, 0.00],
    [0.20, 0.85, 0.95],
    [1.00, 0.10, 0.55],
], dtype=np.float32)

# Per state: 6 × (cx, cy, rx, ry, angle_base)
# cx/cy in aspect-corrected space (aspect ≈ 1.667 for 800×480)
_S = {
    MascotState.IDLE: np.array([
        [ 0.00,  0.08, 0.40, 0.30,  0.00],
        [-0.18, -0.06, 0.38, 0.26,  1.20],
        [ 0.15,  0.06, 0.34, 0.32,  2.50],
        [-0.10,  0.14, 0.32, 0.24,  3.80],
        [ 0.12, -0.10, 0.30, 0.28,  5.10],
        [-0.05,  0.00, 0.28, 0.22,  0.60],
    ], dtype=np.float32),

    MascotState.AWARE: np.array([
        [ 0.00,  0.04, 0.50, 0.36,  0.00],
        [-0.25, -0.04, 0.46, 0.30,  1.00],
        [ 0.22,  0.08, 0.42, 0.38,  2.20],
        [-0.12,  0.18, 0.38, 0.26,  3.50],
        [ 0.16, -0.14, 0.36, 0.32,  4.80],
        [-0.06,  0.00, 0.32, 0.24,  0.30],
    ], dtype=np.float32),

    MascotState.LISTEN: np.array([
        [ 0.00,  0.00, 0.60, 0.22,  0.00],
        [-0.30,  0.00, 0.55, 0.20,  0.80],
        [ 0.26,  0.02, 0.52, 0.24,  2.00],
        [-0.12,  0.04, 0.48, 0.18,  3.30],
        [ 0.18, -0.04, 0.42, 0.22,  4.60],
        [-0.06,  0.00, 0.36, 0.16,  0.20],
    ], dtype=np.float32),

    MascotState.TOUCH: np.array([
        [ 0.05,  0.10, 0.44, 0.38,  0.50],
        [-0.35,  0.12, 0.20, 0.42,  1.80],
        [ 0.32,  0.06, 0.20, 0.40,  3.00],
        [-0.10,  0.20, 0.40, 0.24,  4.20],
        [ 0.14, -0.18, 0.32, 0.30,  5.50],
        [-0.04,  0.02, 0.26, 0.34,  0.80],
    ], dtype=np.float32),

    MascotState.EXCITED: np.array([
        [ 0.00,  0.00, 0.55, 0.42,  0.00],
        [-0.35, -0.12, 0.50, 0.36,  1.50],
        [ 0.32,  0.14, 0.46, 0.38,  2.80],
        [-0.14,  0.22, 0.44, 0.30,  4.00],
        [ 0.20, -0.20, 0.40, 0.34,  5.20],
        [-0.08,  0.05, 0.36, 0.28,  0.60],
    ], dtype=np.float32),
}

# (cx, cy, rx, ry, superellipse_n)
_BLOB = {
    MascotState.IDLE:    (0.0, 0.0, 0.22, 0.22, 2.5),
    MascotState.AWARE:   (0.0, 0.0, 0.26, 0.26, 3.0),
    MascotState.LISTEN:  (0.0, 0.0, 0.34, 0.17, 2.0),
    MascotState.TOUCH:   (0.0, 0.0, 0.24, 0.30, 3.5),
    MascotState.EXCITED: (0.0, 0.0, 0.30, 0.30, 2.0),
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
        self._ep       = _S[MascotState.IDLE].copy()   # current ellipse params
        self._blob     = list(_BLOB[MascotState.IDLE])
        self._vol      = 0.0
        self._bass     = 0.0
        self._mid      = 0.0
        self._high     = 0.0
        self._bars     = [0.0] * 16
        self._rot_offs = np.zeros(6, dtype=np.float32)  # individual rotation drift

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
        target_ep   = _S[state].copy()
        target_blob = list(_BLOB[state])

        # Audio deformation: bass scales ellipses, high adds jitter
        scale = 1.0 + self._bass * 0.25 + self._vol * 0.10
        for i in range(6):
            target_ep[i, 2] *= scale
            target_ep[i, 3] *= scale
            # High frequencies add asymmetric wobble
            target_ep[i, 2] += self._high * 0.04 * math.sin(t * 3.0 + i * 1.1)
            target_ep[i, 3] += self._high * 0.04 * math.cos(t * 2.7 + i * 0.9)

        # Blob reacts to bass
        blob_scale = 1.0 + self._bass * 0.15
        target_blob[2] *= blob_scale
        target_blob[3] *= blob_scale

        # Lerp current → target
        lp = 0.06
        self._ep   = self._ep   + (target_ep - self._ep) * lp
        self._blob = [_lerp(self._blob[i], target_blob[i], lp) for i in range(5)]

        # Rotation: base angle per ellipse + time drift, faster when excited
        rot_speed = {
            MascotState.IDLE:    0.15,
            MascotState.AWARE:   0.25,
            MascotState.LISTEN:  0.40,
            MascotState.TOUCH:   0.60,
            MascotState.EXCITED: 1.20,
        }.get(state, 0.2)
        rot_speed += self._bass * 0.8
        self._rot_offs += rot_speed / self.fps

        # ── GL render ─────────────────────────────────────────────────
        self.ctx.clear(0.04, 0.04, 0.07, 1.0)

        asp = self.W / self.H
        for i in range(6):
            angle = self._ep[i, 4] + self._rot_offs[i]
            self.prog[f'u_ep{i}'].value = (
                float(self._ep[i,0]) * asp,
                float(self._ep[i,1]),
                float(self._ep[i,2]) * asp,
                float(self._ep[i,3]),
            )
            self.prog[f'u_ea{i}'].value = float(angle)
            self.prog[f'u_ec{i}'].value = tuple(ELLIPSE_COLORS[i].tolist())

        self.prog['u_blob'].value = (
            float(self._blob[0]) * asp,
            float(self._blob[1]),
            float(self._blob[2]) * asp,
            float(self._blob[3]),
        )
        self.prog['u_bn'].value = float(self._blob[4])

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
