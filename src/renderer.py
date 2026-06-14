import math
import numpy as np
import pygame
import moderngl
import cv2
from state import MascotState, StateBus


# ── Shaders ───────────────────────────────────────────────────────────────────

VERT = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAG = """
#version 330 core
in  vec2  v_uv;
out vec4  fragColor;

uniform vec2  u_res;

uniform vec4  u_ep0; uniform float u_ea0; uniform vec3 u_ec0; uniform float u_eph0;
uniform vec4  u_ep1; uniform float u_ea1; uniform vec3 u_ec1; uniform float u_eph1;
uniform vec4  u_ep2; uniform float u_ea2; uniform vec3 u_ec2; uniform float u_eph2;
uniform float u_h2, u_h3, u_h4;

float blobMask(vec4 ep, float ea, vec2 p, float phase) {
    vec2 d = p - ep.xy;
    float c = cos(ea), s = sin(ea);
    vec2 q = vec2(c*d.x + s*d.y, -s*d.x + c*d.y);
    vec2 n = q / ep.zw;
    float theta = atan(n.y, n.x);
    float r = length(n);
    float warp = 1.0
        + u_h2 * sin(theta * 2.0 + phase)
        + u_h3 * sin(theta * 3.0 + phase * 1.2)
        + u_h4 * sin(theta * 4.0 + phase * 0.7);
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
#version 330 core
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

HUD_FRAG = """
#version 330 core
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
_R = 0.30   # radio base — rx=ry siempre para no estirar en reposo
_O = 0.065  # offset normal
_ANGLES = [0.20, 2.30, 4.00]

def _pos(o): return [(0.000, o), (-o*0.87, -o*0.50), (o*0.87, -o*0.50)]
def _state(o, r): return np.array([[*p, r, r, a] for p, a in zip(_pos(o), _ANGLES)], dtype=np.float32)

_S = {
    MascotState.IDLE:     _state(_O,     _R),
    MascotState.AWARE:    _state(_O,     _R),
    MascotState.LISTEN:   _state(_O,     _R+.02),
    MascotState.TOUCH:    _state(0.008,  _R+.02),
    MascotState.THINKING: _state(_O,     _R),
    MascotState.ANSWER:   _state(_O,     _R+.02),
}

_H = {
    MascotState.IDLE:     (0.07, 0.05, 0.03),
    MascotState.AWARE:    (0.07, 0.05, 0.03),
    MascotState.LISTEN:   (0.04, 0.12, 0.09),
    MascotState.TOUCH:    (0.03, 0.02, 0.01),
    MascotState.THINKING: (0.04, 0.02, 0.01),
    MascotState.ANSWER:   (0.04, 0.12, 0.09),
}

def _lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, bus: StateBus, cfg: dict, demo: bool = False):
        self.bus  = bus
        self.demo = demo
        self.W    = cfg["display"]["width"]
        self.H    = cfg["display"]["height"]
        self.fps  = cfg["display"]["fps"]
        # ── pygame + OpenGL window ─────────────────────────────────────
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if cfg["display"]["fullscreen"]:
            flags |= pygame.FULLSCREEN
        pygame.display.set_mode((self.W, self.H), flags)
        pygame.display.set_caption("OTO")
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        # ── ModernGL context ──────────────────────────────────────────
        self.ctx = moderngl.create_context(require=330)
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
        self.hud_font      = pygame.font.SysFont(None, 22)
        self.hud_font_big  = pygame.font.SysFont(None, 72)
        self.hud_font_conv = pygame.font.SysFont(None, 28)  # cached — never create in draw loop

        # Static uniform
        self.prog['u_res'].value = (float(self.W), float(self.H))

        # ── Smooth state ──────────────────────────────────────────────
        self._t        = 0.0
        self._ep       = _S[MascotState.IDLE].copy()
        self._n_ep     = N_ELLIPSES
        self._harmonics   = np.array(_H[MascotState.IDLE], dtype=np.float32)
        self._face_offset = np.array([0.0, 0.0], dtype=np.float32)
        self._drag        = np.array([0.0, 0.0], dtype=np.float32)
        self._touch_start = None   # posición píxel donde empezó el toque
        self._drag_origin = np.array([0.0, 0.0], dtype=np.float32)  # drag al inicio
        self._vol      = 0.0
        self._bass     = 0.0
        self._mid      = 0.0
        self._high     = 0.0
        self._bars     = [0.0] * 16
        self._rot_offs  = np.zeros(N_ELLIPSES, dtype=np.float32)
        self._ea        = np.zeros(N_ELLIPSES, dtype=np.float32)  # rendered angle, lerped
        self._prev_state = MascotState.IDLE
        self._rot_speed = 0.008   # lerped rotation speed for smooth transitions
        self._squish_x  = 1.0    # lerped squish for THINKING
        self._squish_y  = 1.0
        self._speak_lvl = 0.0    # lerped speaking_level for ANSWER

    def _wrap_text(self, text, font, max_w):
        """Split text into lines that fit within max_w pixels."""
        words, lines, line = text.split(), [], []
        for w in words:
            test = " ".join(line + [w])
            if font.size(test)[0] <= max_w:
                line.append(w)
            else:
                if line:
                    lines.append(" ".join(line))
                line = [w]
        if line:
            lines.append(" ".join(line))
        return lines

    def _update_hud(self, snap):
        state      = snap["state"]
        face       = snap.get("face_detected", False)
        volume     = self._vol
        debug_frame = snap.get("debug_frame")
        heard      = snap.get("heard_text", "")
        response   = snap.get("response_text", "")

        self.hud_surf.fill((0, 0, 0, 0))

        state_col = {
            MascotState.IDLE:     (100, 100, 180),
            MascotState.AWARE:    (80,  180, 240),
            MascotState.LISTEN:   (80,  220, 140),
            MascotState.TOUCH:    (240, 160, 60),
            MascotState.THINKING: (200, 140, 240),
            MascotState.ANSWER:   (240, 200, 80),
        }.get(state, (140, 140, 140))

        # Corner HUD (small, top-left)
        for i, (txt, col) in enumerate([
            (state.name, state_col),
            (f"VOL {int(volume*100):3d}%", (60, 60, 60)),
        ]):
            surf = self.hud_font.render(txt, True, col)
            self.hud_surf.blit(surf, (10, 10 + i * 20))

        if self.demo:
            big = self.hud_font_big.render(state.name, True, (*state_col, 220))
            bx  = (self.W - big.get_width()) // 2
            self.hud_surf.blit(big, (bx, self.H - 90))

        # Conversation text (bottom of screen)
        pad   = 18
        max_w = self.W - pad * 2
        y     = self.H - pad

        font_conv = self.hud_font_conv

        if response and state in (MascotState.ANSWER, MascotState.IDLE):
            lines = self._wrap_text(response, font_conv, max_w)
            for line in reversed(lines):
                s = font_conv.render(line, True, (240, 200, 80))
                y -= s.get_height() + 2
                self.hud_surf.blit(s, (pad, y))
            label = font_conv.render("OTO:", True, (180, 150, 60))
            y -= label.get_height() + 4
            self.hud_surf.blit(label, (pad, y))
            y -= 8

        if heard and state in (MascotState.LISTEN, MascotState.THINKING, MascotState.ANSWER, MascotState.IDLE):
            lines = self._wrap_text(heard, font_conv, max_w)
            for line in reversed(lines):
                s = font_conv.render(line, True, (180, 230, 180))
                y -= s.get_height() + 2
                self.hud_surf.blit(s, (pad, y))
            label = font_conv.render("TÚ:", True, (100, 160, 100))
            y -= label.get_height() + 4
            self.hud_surf.blit(label, (pad, y))

        if debug_frame is not None:
            try:
                h, w   = debug_frame.shape[:2]
                dw, dh = 120, int(h * 120 / w)
                small  = cv2.resize(debug_frame, (dw, dh))
                rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                cam_s  = pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))
                pygame.draw.rect(self.hud_surf, (20,20,20,200),
                    (self.W-dw-8, 8, dw+4, dh+4))
                self.hud_surf.blit(cam_s, (self.W-dw-6, 10))
            except Exception:
                pass

        data = pygame.image.tostring(self.hud_surf, 'RGBA', False)
        self.hud_tex.write(data)

    def draw_frame(self, snap):
        t     = self._t
        state = snap["state"]
        bars  = snap["fft_bars"]
        vol   = snap["volume"]
        bass  = snap["bass"]
        mid   = snap["mid"]

        # ── Smooth audio ───────────────────────────────────────────────
        sp = 0.12
        self._vol  = _lerp(self._vol,  vol,  sp)
        self._bass = _lerp(self._bass, bass, sp)
        self._mid  = _lerp(self._mid,  mid,  sp)
        self._bars = [_lerp(self._bars[i], bars[i], sp) for i in range(16)]

        # speaking_level from bus (set by LLMThread during ANSWER)
        self._speak_lvl = _lerp(self._speak_lvl, snap.get("speaking_level", 0.0), 0.15)

        # ── Audio level used for animation (mic or TTS depending on state) ─
        anim_vol = self._speak_lvl if state == MascotState.ANSWER else self._vol
        anim_mid = self._speak_lvl * 0.5 if state == MascotState.ANSWER else self._mid

        # ── Target ellipse config ──────────────────────────────────────
        target_ep = _S[state].copy()
        bass_scale  = 1.0 + self._bass * 0.16 + anim_vol * 0.05
        center_pull = 1.0 - self._bass * 0.30 - anim_vol * 0.10
        for i in range(N_ELLIPSES):
            target_ep[i, 0] *= center_pull
            target_ep[i, 1] *= center_pull
            target_ep[i, 2] *= bass_scale
            target_ep[i, 3] *= bass_scale

        lp = 0.18 if state == MascotState.TOUCH else 0.05
        self._ep += (target_ep - self._ep) * lp

        # ── Harmonics — slow lerp for organic transitions ──────────────
        target_h = np.array(_H[state], dtype=np.float32)
        if state in (MascotState.LISTEN, MascotState.ANSWER):
            target_h[2] = _H[state][2] + anim_vol * 0.20 + anim_mid * 0.12
        self._harmonics += (target_h - self._harmonics) * 0.04  # 0.04 = slow morph

        # ── Face tracking ──────────────────────────────────────────────
        face_x = snap.get("face_x_norm", 0.5)
        if state == MascotState.AWARE:
            target_fx = (face_x - 0.5) * 0.30
            self._face_offset[0] += (target_fx - self._face_offset[0]) * 0.015
        else:
            self._face_offset[0] += (0.0 - self._face_offset[0]) * 0.03

        # ── Drag (TOUCH) ───────────────────────────────────────────────
        touch_pos = snap.get("touch_pos", (self.W/2, self.H/2))
        if state == MascotState.TOUCH:
            if self._touch_start is None:
                self._touch_start = touch_pos
                self._drag_origin = self._drag.copy()
            dx = (touch_pos[0] - self._touch_start[0]) / self.W * 2.0
            dy = -((touch_pos[1] - self._touch_start[1]) / self.H * 2.0)
            target_drag = np.clip(
                self._drag_origin + np.array([dx, dy], dtype=np.float32), -0.7, 0.7)
            self._drag += (target_drag - self._drag) * 0.30
        else:
            self._touch_start = None
            self._drag += (np.zeros(2) - self._drag) * 0.025

        # ── Rotation speed — lerped for smooth state transitions ───────
        target_speed = {
            MascotState.IDLE:     0.008,
            MascotState.AWARE:    0.012,
            MascotState.LISTEN:   0.06,
            MascotState.TOUCH:    0.030,
            MascotState.THINKING: 0.004,
            MascotState.ANSWER:   0.06,
        }.get(state, 0.010)
        target_speed += anim_vol * 0.18 + anim_mid * 0.10 + self._bass * 0.02
        self._rot_speed = _lerp(self._rot_speed, target_speed, 0.04)

        # Wobble amplitude also lerps
        target_wobble = 0.12 if state in (MascotState.LISTEN, MascotState.ANSWER) else 0.012
        if not hasattr(self, '_wobble_amp'):
            self._wobble_amp = target_wobble
        self._wobble_amp = _lerp(self._wobble_amp, target_wobble, 0.04)

        _speed_mult  = [1.00, -1.41, 1.73]
        _wobble_freqs = [0.11, 0.17, 0.13]
        for i in range(N_ELLIPSES):
            wobble = self._wobble_amp * math.sin(t * _wobble_freqs[i] * 2 * math.pi + i * 2.1)
            self._rot_offs[i] += (self._rot_speed * _speed_mult[i] + wobble) / self.fps

        # ── THINKING squish — lerped ───────────────────────────────────
        if state == MascotState.THINKING:
            pulse          = math.sin(t * 1.2)
            target_squish_x = 1.55 + pulse * 0.30
            target_squish_y = 0.45 - pulse * 0.12
        else:
            target_squish_x = target_squish_y = 1.0
        self._squish_x = _lerp(self._squish_x, target_squish_x, 0.06)
        self._squish_y = _lerp(self._squish_y, target_squish_y, 0.06)

        # ── GL render ──────────────────────────────────────────────────
        self.ctx.clear(0.04, 0.04, 0.07, 1.0)
        asp = self.W / self.H

        # On exit from THINKING: sync rot_offs to _ea so angle continues smoothly
        if self._prev_state == MascotState.THINKING and state != MascotState.THINKING:
            for j in range(N_ELLIPSES):
                self._rot_offs[j] = self._ea[j] - self._ep[j, 4]

        for i in range(N_ELLIPSES):
            angle = self._ep[i, 4] + self._rot_offs[i]
            if state == MascotState.THINKING:
                self._ea[i] = _lerp(self._ea[i], 0.0, 0.05)
            else:
                self._ea[i] = angle
            r  = (float(self._ep[i,2]) + float(self._ep[i,3])) * 0.5
            self.prog[f'u_ep{i}'].value = (
                float(self._ep[i,0]) * asp + (self._drag[0] + self._face_offset[0]) * asp,
                float(self._ep[i,1]) + self._drag[1],
                r * self._squish_x,
                r * self._squish_y,
            )
            self.prog[f'u_ea{i}'].value = float(self._ea[i])
            self.prog[f'u_ec{i}'].value = tuple(ELLIPSE_COLORS[i].tolist())
            self.prog[f'u_eph{i}'].value = float(angle + i * 2.094)

        self.prog['u_h2'].value = float(self._harmonics[0])
        self.prog['u_h3'].value = float(self._harmonics[1])
        self.prog['u_h4'].value = float(self._harmonics[2])
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self._prev_state = state

        # ── HUD overlay ────────────────────────────────────────────────
        self._update_hud(snap)
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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        snap = self.bus.snapshot()
                        if snap["state"] == MascotState.ANSWER:
                            self.bus.update(stop_speaking=True)
                        else:
                            self.bus.update(recording=True)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        self.bus.update(recording=False)
                elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                    pos = getattr(event, 'pos', None) or (
                        int(event.x * self.W), int(event.y * self.H))
                    self.bus.update(touch_active=True, touch_pos=pos, recording=True)
                elif event.type in (pygame.MOUSEMOTION, pygame.FINGERMOTION):
                    if snap := self.bus.snapshot():
                        if snap.get("touch_active"):
                            pos = getattr(event, 'pos', None) or (
                                int(event.x * self.W), int(event.y * self.H))
                            self.bus.update(touch_pos=pos)
                elif event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                    self.bus.update(touch_active=False, recording=False)

            self._t += 1.0 / self.fps
            self.draw_frame(self.bus.snapshot())
            self.clock.tick(self.fps)

        pygame.quit()
