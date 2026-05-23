"""
3D iskelet render motoru — PyOpenGL + Pygame
Pygame yüzeyine (Surface) offscreen render eder, radar_ui.py bunu blit eder.
"""

import math
import numpy as np
import pygame
from typing import Optional

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from core.skeleton import Skeleton, BONES, BONE_GROUPS

# ─── Renk Paleti ──────────────────────────────────────────────────────────────
_COLORS = {
    "spine":       (0.0,  0.90, 0.46),   # yeşil
    "r_arm":       (0.0,  0.75, 1.0),    # mavi
    "l_arm":       (0.0,  0.75, 1.0),
    "r_leg":       (1.0,  0.14, 0.40),   # kırmızı
    "l_leg":       (1.0,  0.14, 0.40),
    "face":        (1.0,  0.85, 0.0),    # sarı
    "joint":       (1.0,  1.0,  1.0),    # beyaz
    "grid_line":   (0.13, 0.17, 0.22),
    "grid_active": (0.0,  0.90, 0.46, 0.15),
    "background":  (0.04, 0.04, 0.06),
}

_BONE_GROUP_MAP: dict[tuple, str] = {}
for group_name, pairs in BONE_GROUPS.items():
    for pair in pairs:
        _BONE_GROUP_MAP[pair] = group_name


class Renderer3D:
    """
    Tek sorumluluğu: verilen Skeleton listesini 3D olarak çizmek.
    OpenGL yoksa otomatik olarak 2D fallback moduna geçer.
    """

    def __init__(self, viewport_w: int, viewport_h: int):
        self.vp_w = viewport_w
        self.vp_h = viewport_h
        self.use_opengl = False

        # Kamera
        self.cam_distance = 6.0
        self.cam_azimuth  = 30.0   # derece — yatay döndürme
        self.cam_elevation = 20.0  # derece — dikey açı

        # Mouse sürükleme durumu
        self._drag_start: Optional[tuple] = None
        self._az_start  = self.cam_azimuth
        self._el_start  = self.cam_elevation

        if OPENGL_AVAILABLE:
            surface = pygame.display.get_surface()
            if surface is not None and (surface.get_flags() & pygame.OPENGL):
                self.use_opengl = True
                self._init_opengl()
            else:
                print("⚠️  OpenGL context bulunamadı — 2D fallback modunda çalışıyor.")
        else:
            print("⚠️  PyOpenGL bulunamadı — 2D fallback modunda çalışıyor.")

    # ── Başlatma ────────────────────────────────────────────────────────────────
    def _init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glPointSize(6.0)

    # ── Kamera Kontrolü ─────────────────────────────────────────────────────────
    def handle_mouse_event(self, event: pygame.event.Event, offset_x: int = 0):
        """Fare olaylarını alır. offset_x: viewport'un ekran içindeki x başlangıcı."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if mx >= offset_x:
                self._drag_start = (mx, my)
                self._az_start   = self.cam_azimuth
                self._el_start   = self.cam_elevation
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_start = None
        elif event.type == pygame.MOUSEMOTION and self._drag_start:
            dx = event.pos[0] - self._drag_start[0]
            dy = event.pos[1] - self._drag_start[1]
            self.cam_azimuth   = self._az_start  + dx * 0.4
            self.cam_elevation = max(-89, min(89, self._el_start - dy * 0.4))
        elif event.type == pygame.MOUSEWHEEL:
            self.cam_distance = max(2.0, min(15.0, self.cam_distance - event.y * 0.3))

    # ── Ana Render ──────────────────────────────────────────────────────────────
    def render(self, skeletons: list[Skeleton], active_grid: tuple[int, int]):
        """OpenGL çağrılarını yapar. Pygame display OpenGL modundaysa direkt ekrana yazar."""
        if not self.use_opengl:
            return  # fallback için render_fallback_to_surface kullan

        glViewport(0, 0, self.vp_w, self.vp_h)
        bg = _COLORS["background"]
        glClearColor(*bg, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self._set_camera()
        self._draw_grid(active_grid)

        for sk in skeletons:
            self._draw_skeleton(sk)

    def _set_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.vp_w / self.vp_h, 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        az  = math.radians(self.cam_azimuth)
        el  = math.radians(self.cam_elevation)
        eye_x = self.cam_distance * math.cos(el) * math.sin(az)
        eye_y = self.cam_distance * math.sin(el)
        eye_z = self.cam_distance * math.cos(el) * math.cos(az)

        gluLookAt(eye_x, eye_y, eye_z,  # göz
                  0, 0.8, 0,             # baktığı nokta (insan boyunun ortası)
                  0, 1, 0)               # yukarı yönü

    # ── Grid Zemini ─────────────────────────────────────────────────────────────
    def _draw_grid(self, active_cell: tuple[int, int]):
        glLineWidth(1.0)
        r, g, b = _COLORS["grid_line"]
        glColor3f(r, g, b)
        glBegin(GL_LINES)
        for i in range(-2, 4):
            glVertex3f(i * 1.5 - 0.75, 0, -2.25)
            glVertex3f(i * 1.5 - 0.75, 0,  2.25)
            glVertex3f(-2.25, 0, i * 1.5 - 0.75)
            glVertex3f( 2.25, 0, i * 1.5 - 0.75)
        glEnd()

        # Aktif hücreyi vurgula
        ar, ac = active_cell
        cx = (ac - 1) * 1.5
        cz = (ar - 1) * 1.5
        glColor4f(0.0, 0.90, 0.46, 0.18)
        glBegin(GL_QUADS)
        glVertex3f(cx - 0.75, 0.01, cz - 0.75)
        glVertex3f(cx + 0.75, 0.01, cz - 0.75)
        glVertex3f(cx + 0.75, 0.01, cz + 0.75)
        glVertex3f(cx - 0.75, 0.01, cz + 0.75)
        glEnd()

        glLineWidth(2.0)

    # ── İskelet Çizimi ──────────────────────────────────────────────────────────
    def _draw_skeleton(self, sk: Skeleton):
        kps = sk.get_world_keypoints()  # (17, 3)

        # Kemikler
        glBegin(GL_LINES)
        for i, j in BONES:
            group = _BONE_GROUP_MAP.get((i, j)) or _BONE_GROUP_MAP.get((j, i), "spine")
            r, g, b = _COLORS[group]
            glColor3f(r, g, b)
            glVertex3fv(kps[i])
            glVertex3fv(kps[j])
        glEnd()

        # Eklem noktaları
        glBegin(GL_POINTS)
        glColor3f(*_COLORS["joint"])
        for kp in kps:
            glVertex3fv(kp)
        glEnd()

    # ── 2D Fallback (PyOpenGL yoksa) ────────────────────────────────────────────
    def render_fallback_to_surface(
        self,
        surface: pygame.Surface,
        skeletons: list[Skeleton],
        active_grid: tuple[int, int],
        grid_dims: tuple[int, int] = (3, 3),
    ):
        """PyOpenGL olmadan Pygame 2D ile basit projeksiyon çizer."""
        surface.fill((10, 11, 14))

        az  = math.radians(self.cam_azimuth)
        el  = math.radians(self.cam_elevation)

        def project(p3: np.ndarray) -> tuple[int, int]:
            x =  p3[0] * math.cos(az) + p3[2] * math.sin(az)
            z = -p3[0] * math.sin(az) + p3[2] * math.cos(az)
            y =  p3[1] * math.cos(el) - z      * math.sin(el)
            d = 6.0 + z * math.sin(el) + p3[1] * math.sin(el) * 0.3
            d = max(d, 0.5)
            scale = 280 / d
            sx = int(self.vp_w // 2 + x * scale)
            sy = int(self.vp_h // 2 - (y - 0.8) * scale)
            return sx, sy

        # Dinamik grid — dünya koordinatları [-1.5, +1.5]
        total_rows, total_cols = grid_dims
        step_x = 3.0 / total_cols
        step_z = 3.0 / total_rows
        for ci in range(total_cols + 1):
            wx = -1.5 + ci * step_x
            p1 = project(np.array([wx, 0, -1.5]))
            p2 = project(np.array([wx, 0,  1.5]))
            pygame.draw.line(surface, (34, 43, 56), p1, p2, 1)
        for ri in range(total_rows + 1):
            wz = -1.5 + ri * step_z
            p1 = project(np.array([-1.5, 0, wz]))
            p2 = project(np.array([ 1.5, 0, wz]))
            pygame.draw.line(surface, (34, 43, 56), p1, p2, 1)

        # Aktif hücre vurgusu
        ar, ac = active_grid
        ax_min = -1.5 + ac * step_x
        ax_max = ax_min + step_x
        az_min = -1.5 + ar * step_z
        az_max = az_min + step_z
        corners = [
            project(np.array([ax_min, 0.01, az_min])),
            project(np.array([ax_max, 0.01, az_min])),
            project(np.array([ax_max, 0.01, az_max])),
            project(np.array([ax_min, 0.01, az_max])),
        ]
        active_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(active_surf, (0, 230, 118, 35), corners)
        surface.blit(active_surf, (0, 0))
        pygame.draw.polygon(surface, (0, 180, 90), corners, 1)

        _COLOR_MAP_2D = {
            "spine": (0, 230, 118), "r_arm": (0, 191, 255), "l_arm": (0, 191, 255),
            "r_leg": (255, 35, 102), "l_leg": (255, 35, 102), "face": (255, 215, 0),
        }

        for sk in skeletons:
            kps = sk.get_world_keypoints()
            proj = [project(kps[i]) for i in range(len(kps))]

            for i, j in BONES:
                group = _BONE_GROUP_MAP.get((i, j)) or _BONE_GROUP_MAP.get((j, i), "spine")
                color = _COLOR_MAP_2D.get(group, (255, 255, 255))
                pygame.draw.line(surface, color, proj[i], proj[j], 2)

            for pt in proj:
                pygame.draw.circle(surface, (255, 255, 255), pt, 3)
