import pygame
import sys
import os
import random
import math
import time
import numpy as np
from typing import Optional, List, Tuple

from core.radar_ml import RadarML
from core.skeleton import Skeleton
from ui.renderer_3d import Renderer3D
from core.esp32_serial_reader import ESP32SerialReader
from core.csi_features import build_feature_vector, N_SUBCARRIERS as _N_SC
import joblib

os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEO_CENTERED'] = '1'

# ─── Ekran Düzeni ─────────────────────────────────────────────────────────────
TOTAL_W   = 1820
TOTAL_H   = 980
LEFT_W    = 860    # Sol panel (grid + grafik + config)
VIEW3D_W  = 540    # Orta 3D viewport
RIGHT_W   = TOTAL_W - LEFT_W - VIEW3D_W   # ~420 px — telemetri
FPS       = 60

# ─── Renk Paleti ──────────────────────────────────────────────────────────────
BG_COLOR          = (6,    8,   12)
PANEL_COLOR       = (12,  16,   24)
PANEL_DARK        = (8,   11,   17)
PANEL_MID         = (16,  22,   34)
TEXT_COLOR        = (195, 210,  225)
TEXT_DIM          = (75,  90,  108)
TEXT_BRIGHT       = (230, 240,  255)
ACTIVE_COLOR      = (0,   230,  118)
ACTIVE_DIM        = (0,   140,   72)
PASSIVE_COLOR     = (28,  38,   52)
AVATAR_COLOR      = (255,  35,  102)
GRAPH_COLOR       = (0,   191,  255)
GRAPH_DIM         = (0,    80,  130)
VITAL_COLOR       = (255,  80,  120)
VITAL_GREEN       = (0,   230,  118)
WARN_RED          = (255,  55,   55)
WARN_YELLOW       = (255, 210,    0)
WARN_ORANGE       = (255, 140,    0)
DIVIDER           = (18,  24,   36)
BUTTON_NORMAL     = (18,  26,   40)
BUTTON_HOVER      = (0,   170,   80)
BUTTON_RESET_NRM  = (30,  14,   14)
BUTTON_RESET_HOV  = (170,  36,   36)
BORDER_COLOR      = (38,  52,   72)
BORDER_BRIGHT     = (60,  80,  110)
NEON_CYAN         = (0,   255,  240)
NEON_PURPLE       = (180,  50,  255)
GRID_LINE_COL     = (22,  32,   48)
SCANLINE_COL      = (0,    0,    0, 10)
HEATMAP_ACTIVE    = (0,   230,  118)

# ─── Dinamik Grid Yardımcıları ────────────────────────────────────────────────
MIN_CELL_M = 2.0   # Hücre başına minimum metre (donanım hata payı ±1m)

def compute_grid_dims(room_w: float, room_l: float) -> Tuple[int, int]:
    """
    Fiziksel kural: donanım hata payı ±1m → 2m belirsizlik çapı.
    Hiçbir hücre 2.0m'nin altına DÜŞMEMELİ.

    Algoritma (tam bölme):
        cols = max(1, int(room_w // 2.0))
        rows = max(1, int(room_l // 2.0))

    Örnekler:
        9m × 3m  → 4 × 1  (hücre: 2.25m × 3.0m)
        6m × 6m  → 3 × 3  (hücre: 2.0m  × 2.0m)
        3m × 4m  → 1 × 2  (hücre: 3.0m  × 2.0m)
    """
    cols = max(1, int(room_w // MIN_CELL_M))
    rows = max(1, int(room_l // MIN_CELL_M))
    return cols, rows

def build_grid_labels(cols: int, rows: int) -> List[List[str]]:
    row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labels = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(f"{row_letters[r]}{c+1}")
        labels.append(row)
    return labels

def _rssi_label(v: int) -> str:
    if v >= -55: return "EXCELLENT"
    if v >= -65: return "GOOD"
    if v >= -75: return "FAIR"
    return "WEAK"

def _rssi_color(v: int) -> tuple:
    if v >= -55: return (0, 230, 118)
    if v >= -65: return (0, 191, 255)
    if v >= -75: return (255, 210, 0)
    return (255, 60, 60)

def _heatmap_fill(rssi: int) -> tuple:
    t = max(0.0, min(1.0, (rssi + 85) / 45.0))
    r = int(180 * (1 - t))
    g = int(230 * t)
    b = int(255 * (1 - t) * 0.5 + 80 * t)
    a = int(20 + 55 * t)
    return (r, g, b, a)


# ═══════════════════════════════════════════════════════════════════════════════
class Radar3x3UI:
# ═══════════════════════════════════════════════════════════════════════════════

    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((TOTAL_W, TOTAL_H))
        pygame.display.set_caption(
            "Wi-Fi CSI Radar  //  TACTICAL COMMAND v3.1  //  CLASSIFIED"
        )
        self.clock      = pygame.time.Clock()
        self.last_time  = time.time()

        # ── Alt sistemler ──────────────────────────────────────────────────────
        self.ml_engine  = RadarML()
        self.skeleton   = Skeleton(grid_position=(0, 0), grid_dims=(1, 1))
        self.renderer3d = Renderer3D(VIEW3D_W, TOTAL_H)

        # ── Fontlar (sadece __init__'te yüklenir) ─────────────────────────────
        self.f_title   = pygame.font.SysFont("Courier", 16, bold=True)
        self.f_title_lg= pygame.font.SysFont("Courier", 18, bold=True)
        self.f_body    = pygame.font.SysFont("Courier", 13, bold=True)
        self.f_sm      = pygame.font.SysFont("Courier", 11)
        self.f_xs      = pygame.font.SysFont("Courier", 10)
        self.f_grid    = pygame.font.SysFont("Arial",   18, bold=True)
        self.f_grid_sm = pygame.font.SysFont("Arial",   13, bold=True)
        self.f_mono    = pygame.font.SysFont("Courier", 12)
        self.f_warn    = pygame.font.SysFont("Courier", 11, bold=True)
        self.f_input   = pygame.font.SysFont("Courier", 13)
        self.f_vital   = pygame.font.SysFont("Courier", 12, bold=True)
        self.f_vital_sm= pygame.font.SysFont("Courier", 10)
        self.f_big     = pygame.font.SysFont("Courier", 22, bold=True)

        # ── Oda boyutları ve dinamik grid ──────────────────────────────────────
        self.cfg_w_str  = "3.0"
        self.cfg_l_str  = "4.0"
        self.cfg_w_val  = 3.0
        self.cfg_l_val  = 4.0
        self.cfg_warn   = False
        self.cfg_active_field: Optional[str] = None
        self.cfg_w_rect  = pygame.Rect(0, 0, 1, 1)
        self.cfg_l_rect  = pygame.Rect(0, 0, 1, 1)
        self.cfg_btn_rect = pygame.Rect(0, 0, 1, 1)

        # Dinamik grid hesaplama
        self.grid_cols: int = 0
        self.grid_rows: int = 0
        self.grid_labels: List[List[str]] = []
        self.regions: dict = {}
        self._rebuild_grid()

        # ── Durum değişkenleri ─────────────────────────────────────────────────
        self.current_row    = 0
        self.current_col    = 0
        self.current_region = self.grid_labels[0][0]
        self.avatar_x       = float(self.regions[self.current_region].centerx)
        self.avatar_y       = float(self.regions[self.current_region].centery)
        self.is_ml_mode     = True   # ESP32 + model varsa direkt ML modunda başla
        self.rssi_active    = -48
        self.rssi_cells: dict = {}
        self._init_rssi_cells()

        # ── Hysteresis filtresi ────────────────────────────────────────────────
        self._hyst_candidate = self.current_region
        self._hyst_count     = 0
        self._HYST_THRESHOLD = 3

        # ── ML probability smoothing buffer ────────────────────────────────────
        # Asimetrik karar: yeni zone'a gecmek icin mevcuttan MARGIN kadar
        # daha yuksek proba lazim. Bu hem A->B'yi hizlandirir hem de
        # gurultuden gelen geri-donusleri engeller.
        self._ml_proba_buffer: list = []          # her eleman: np.ndarray
        self._ML_PROBA_WINDOW   = 12              # ~0.4s @ 30Hz — hizli tepki
        self._ML_ENTER_THRESHOLD = 0.50           # yeni zone icin minimum proba
        self._ML_SWITCH_MARGIN  = 0.15            # mevcuttan ne kadar fazla olmali
        self._ml_classes: list = []
        self._dead_sc: list = []                  # train_model.py'den yuklenir

        # ── İskelet yürüme hedefi (dünya koordinatı) ──────────────────────────
        self._skel_walk_timer = 0.0
        self._skel_is_walking = False

        # ── CSI sinyal dalga grafiği ──────────────────────────────────────────
        self._graph_rect      = pygame.Rect(0, 0, 1, 1)
        self.wave_history     = [0.0] * 200
        self._wave_phase      = 0.0

        # ── Spektral band geçmişi (60 frame smooth) ───────────────────────────
        self._band_low   = [0.0] * 60   # 0.1–0.5 Hz solunum
        self._band_mid   = [0.0] * 60   # 0.5–2 Hz yürüyüş
        self._band_high  = [0.0] * 60   # 2–10 Hz hızlı hareket
        self._band_rect  = pygame.Rect(0, 0, 1, 1)

        # ── Alt CSI heatmap (sol panelde bağımsız grafik) ─────────────────────
        self._csi_heatmap_data = np.zeros((8, 16), dtype=np.float32)
        self._csi_hmap_rect    = pygame.Rect(0, 0, 1, 1)

        # ── Telemetri / Diagnostics ────────────────────────────────────────────
        self.cqi        = 72
        self.csi_rate   = 320
        self.diag_log: List[str] = []
        self._blink_t   = 0.0
        self._reset_hover = False
        self.reset_btn_rect = pygame.Rect(0, 0, 1, 1)

        # ── Vital Signs ────────────────────────────────────────────────────────
        self._bpm            = 72.0
        self._bpm_target     = 72.0
        self._bpm_display    = 72
        self._resp_rate      = 16.0
        self._resp_target    = 16.0
        self._ekg_history    = [0.0] * 80
        self._ekg_phase      = 0.0
        self._vital_blink    = 0.0
        self._bpm_state      = "STABLE"
        self._resp_state     = "NORMAL"

        # ── Signal Quality CQI sinüs fazı ─────────────────────────────────────
        self._cqi_phase      = 0.0

        # ── Kare sayacı ───────────────────────────────────────────────────────
        self._frame = 0

        # ── ESP32 serial reader ────────────────────────────────────────────────
        self._esp32: Optional[ESP32SerialReader] = None
        self._esp32_port: Optional[str] = None
        self._csi_buffer: List[np.ndarray] = []
        self._csi_lock = __import__("threading").Lock()
        # Vital signs için sliding window (30sn × 50Hz = 1500 sample)
        self._vital_csi_window: List[float] = []
        self._vital_window_size = 1500
        self._vital_update_counter = 0
        self._connect_esp32()

        # ── ML model (wifi_radar_model.pkl) ────────────────────────────────────
        self._ml_model  = None
        self._ml_labels: List[str] = []
        self._load_model()

        # ── Başlangıç logu ─────────────────────────────────────────────────────
        self._log(">> SYSTEM BOOT: Tactical Radar v3.1 online.")
        self._log(f">> GRID: {self.grid_cols}x{self.grid_rows} initialized.")
        self._log(">> STANDBY: Awaiting ESP32 serial handshake...")

    # ─── ESP32 bağlantı / model yükleme ──────────────────────────────────────
    def _connect_esp32(self):
        ports = ESP32SerialReader.find_esp32_ports()
        if not ports:
            return
        port = ports[0]
        self._esp32 = ESP32SerialReader(port, self._on_csi_packet)
        if self._esp32.start():
            self._esp32_port = port
        else:
            self._esp32 = None

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "..", "core", "wifi_radar_model.pkl")
        if not os.path.exists(model_path):
            return
        data = joblib.load(model_path)
        self._ml_model  = data["model"]
        self._ml_labels = data["labels"]
        # v2 model: dead_sc listesi de yuklu
        self._dead_sc   = data.get("dead_sc", [])
        version = data.get("version", 1)
        self._log(f">> MODEL: v{version} loaded ({data.get('name','?')}, {len(self._ml_labels)} class, {len(self._dead_sc)} dead SC)")

    def _on_csi_packet(self, arr: np.ndarray):
        with self._csi_lock:
            self._csi_buffer.append(arr.copy())

    def _process_csi_buffer(self):
        with self._csi_lock:
            packets = self._csi_buffer[:]
            self._csi_buffer.clear()
        if not packets:
            return
        for arr in packets:
            mean_amp = float(arr.mean())
            self.wave_history.pop(0)
            self.wave_history.append(float(np.clip(mean_amp * 2.0, 20, 180)))
            padded = np.resize(arr, 128).reshape(8, 16)
            self._csi_heatmap_data = np.clip(padded / padded.max() * 100 if padded.max() > 0 else padded, 0, 100).astype(np.float32)
            # Vital signs window
            self._vital_csi_window.append(mean_amp)
            if len(self._vital_csi_window) > self._vital_window_size:
                self._vital_csi_window.pop(0)
        # CSI rate güncelle
        if self._esp32:
            self.csi_rate = int(self._esp32.csi_rate_hz)
        # ── ML tahmin — TUM gelen paketleri pencereye ekle, asimetrik karar ──
        if self._ml_model is not None and self.is_ml_mode and packets:
            try:
                # Tum batch icin feature matrisi (tek predict_proba cagrisi)
                feats = np.array([self._build_features(p) for p in packets])
                probs_batch = self._ml_model.predict_proba(feats)  # (B, C)
                if not self._ml_classes:
                    self._ml_classes = list(self._ml_model.classes_)

                for row in probs_batch:
                    self._ml_proba_buffer.append(row)
                # Sliding window: en eskileri at
                while len(self._ml_proba_buffer) > self._ML_PROBA_WINDOW:
                    self._ml_proba_buffer.pop(0)

                # En az pencere yarisi dolmadan karar verme
                if len(self._ml_proba_buffer) >= max(5, self._ML_PROBA_WINDOW // 2):
                    avg = np.mean(np.stack(self._ml_proba_buffer), axis=0)
                    best_i = int(np.argmax(avg))
                    best_label = self._ml_classes[best_i]
                    best_p = float(avg[best_i])

                    # Su anki gosterilen zone'un (label) probasi
                    cur_label = self._current_zone_label()
                    cur_p = float(avg[self._ml_classes.index(cur_label)]) if cur_label in self._ml_classes else 0.0

                    # Karar kurali:
                    #  - Mevcut zone hala en yuksekse: hicbir sey yapma (stabil kal)
                    #  - Yeni zone en yuksek VE mevcuttan MARGIN kadar fazlaysa: gec
                    #  - Yeni zone en yuksek ama mevcut zone hala ciddi (>=0.30):
                    #    ekstra esik (0.55) iste — kararsiz sinirda gec sallanma
                    should_switch = False
                    if best_label != cur_label and best_p >= self._ML_ENTER_THRESHOLD:
                        if (best_p - cur_p) >= self._ML_SWITCH_MARGIN:
                            should_switch = True
                        elif cur_p < 0.30 and best_p >= 0.55:
                            should_switch = True

                    if should_switch:
                        self._apply_zone_prediction(best_label)
            except (AttributeError, ValueError):
                feat = self._build_features(packets[-1])
                label = self._ml_model.predict([feat])[0]
                self._apply_zone_prediction(label)

    def _current_zone_label(self) -> str:
        """Su anki zone'un ML label karsiligi (zone_A / zone_B / empty)."""
        total = self.grid_rows * self.grid_cols
        first_r, first_c = 0, 0
        last_r = (total - 1) // self.grid_cols
        last_c = (total - 1) % self.grid_cols
        if (self.current_row, self.current_col) == (first_r, first_c):
            return "zone_A"
        if (self.current_row, self.current_col) == (last_r, last_c):
            return "zone_B"
        return "empty"

    def _build_features(self, arr: np.ndarray) -> np.ndarray:
        # Egitim ile birebir ayni feature pipeline'i (core/csi_features.py).
        N = _N_SC
        if len(arr) >= N:
            row = arr[:N].astype(np.float32)
        else:
            row = np.concatenate([arr, np.zeros(N - len(arr))]).astype(np.float32)
        return build_feature_vector(row, self._dead_sc)

    def _apply_zone_prediction(self, label: str):
        # zone_A → ilk hücre, zone_B → son hücre, empty → mevcut yerinde kal
        total = self.grid_rows * self.grid_cols
        first_r, first_c = 0, 0
        last_r = (total - 1) // self.grid_cols
        last_c = (total - 1) % self.grid_cols
        zone_map = {
            "zone_A": (first_r, first_c),
            "zone_B": (last_r,  last_c),
        }
        pos = zone_map.get(label)
        if pos is None:
            # 'empty' veya bilinmeyen label: hicbir sey yapma (mevcut zone'da kal)
            return
        # Zaten bu zone'daysak histerezis sayacini bosuna arttirma
        if (self.current_row, self.current_col) == pos:
            self._hyst_candidate = self.grid_labels[pos[0]][pos[1]]
            self._hyst_count = 0
            return
        self._request_zone_change(pos[0], pos[1])

    # ─── Dinamik Grid Kurma ───────────────────────────────────────────────────
    def _rebuild_grid(self):
        cols, rows = compute_grid_dims(self.cfg_w_val, self.cfg_l_val)
        self.grid_cols   = cols
        self.grid_rows   = rows
        self.grid_labels = build_grid_labels(cols, rows)

        grid_area_h = int(TOTAL_H * 0.54)
        margin_x, margin_y = 16, 28
        avail_w = LEFT_W - margin_x * 2
        avail_h = grid_area_h - margin_y - 8

        GAP   = max(4, min(10, 200 // (cols + rows)))
        box_w = (avail_w - GAP * (cols - 1)) // cols
        box_h = (avail_h - GAP * (rows - 1)) // rows

        self.regions = {}
        for r in range(rows):
            for c in range(cols):
                label = self.grid_labels[r][c]
                rx = margin_x + c * (box_w + GAP)
                ry = margin_y + r * (box_h + GAP)
                self.regions[label] = pygame.Rect(rx, ry, box_w, box_h)

        cw = self.cfg_w_val / cols
        cl = self.cfg_l_val / rows
        self.cfg_warn = (cw < MIN_CELL_M or cl < MIN_CELL_M)

        if not hasattr(self, 'current_row'):
            self.current_row = 0
            self.current_col = 0
        self.current_row = min(self.current_row, rows - 1)
        self.current_col = min(self.current_col, cols - 1)
        self.current_region = self.grid_labels[self.current_row][self.current_col]

        if self.current_region in self.regions:
            rect = self.regions[self.current_region]
            if hasattr(self, 'avatar_x'):
                self.avatar_x = float(rect.centerx)
                self.avatar_y = float(rect.centery)

        self._hyst_candidate = self.current_region
        self._hyst_count     = 0

        self._init_rssi_cells()

        if hasattr(self, 'skeleton'):
            self.skeleton.set_grid_position(
                self.current_row, self.current_col,
                grid_dims=(rows, cols)
            )

        self._rebuild_keymap()

    def _rebuild_keymap(self):
        num_keys = [
            pygame.K_1, pygame.K_2, pygame.K_3,
            pygame.K_4, pygame.K_5, pygame.K_6,
            pygame.K_7, pygame.K_8, pygame.K_9,
        ]
        self._key_map: dict = {}
        idx = 0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if idx < len(num_keys):
                    self._key_map[num_keys[idx]] = (r, c)
                    idx += 1
        self._arrow_map = {
            pygame.K_UP:    (-1,  0),
            pygame.K_DOWN:  ( 1,  0),
            pygame.K_LEFT:  ( 0, -1),
            pygame.K_RIGHT: ( 0,  1),
        }

    # ─── RSSI Başlatma ────────────────────────────────────────────────────────
    def _init_rssi_cells(self):
        self.rssi_cells = {}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                label = self.grid_labels[r][c]
                self.rssi_cells[label] = random.randint(-80, -55)

    # ═══════════════════════════════════════════════════════════════════════════
    # ANA DÖNGÜ
    # ═══════════════════════════════════════════════════════════════════════════
    def run(self):
        running = True
        while running:
            now = time.time()
            dt  = min(now - self.last_time, 0.05)
            self.last_time = now
            self.clock.tick(FPS)
            self._frame += 1

            mx, my = pygame.mouse.get_pos()
            self._reset_hover = self.reset_btn_rect.collidepoint(mx, my)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                self.renderer3d.handle_mouse_event(event, offset_x=LEFT_W)

            self._update(dt)
            self._draw()

        pygame.quit()
        sys.exit()

    # ═══════════════════════════════════════════════════════════════════════════
    # GÜNCELLEME
    # ═══════════════════════════════════════════════════════════════════════════
    def _update(self, dt: float):
        self._blink_t   += dt
        self._vital_blink += dt
        self._cqi_phase += dt * 0.4

        # Gerçek ESP32 varsa buffer işle, yoksa simüle et
        esp32_live = self._esp32 is not None and self._esp32.connected
        if esp32_live:
            self._process_csi_buffer()
            csi_data = np.array(self.wave_history[-64:], dtype=np.float32)
        else:
            csi_data = self._simulate_csi() if self.is_ml_mode else None
        self.skeleton.update(dt, csi_amplitude=csi_data)

        if not self.is_ml_mode:
            if self._skel_walk_timer > 0:
                self._skel_walk_timer -= dt
                self._skel_is_walking = True
                self.skeleton.activity = "walking"
                self.skeleton.anim_phase += dt * 4.0
            else:
                self._skel_is_walking = False
                if self.skeleton.activity == "walking":
                    self.skeleton.activity = "standing"

        if self.current_region in self.regions:
            rect = self.regions[self.current_region]
            self.avatar_x += (rect.centerx - self.avatar_x) * 0.14
            self.avatar_y += (rect.centery - self.avatar_y) * 0.14

        self._update_wave(dt)

        if self._frame % 3 == 0:
            self._update_csi_heatmap()

        if self._frame % 8 == 0:
            self._update_rssi_cells()

        if self._frame % 6 == 0:
            self.cqi      = max(0, min(100, self.cqi + random.randint(-3, 3)))
            if not esp32_live:
                self.csi_rate = random.randint(314, 326)

        self._update_vitals(dt)

    def _simulate_csi(self) -> np.ndarray:
        t     = pygame.time.get_ticks() * 0.001
        base  = 60 + math.sin(t * 2.1) * 20
        noise = np.random.normal(0, 4, 64).astype(np.float32)
        return (np.full(64, base, dtype=np.float32) + noise).clip(0, 150)

    def _update_wave(self, dt: float):
        self._wave_phase += dt * 5.0
        self.wave_history.pop(0)
        t       = pygame.time.get_ticks() * 0.001
        base    = math.sin(t * 3.1) * 22 + math.sin(t * 7.3) * 10
        carrier = math.cos(t * 0.9) * 18

        if self._skel_is_walking or self.skeleton.activity == "walking":
            noise = random.gauss(0, 14)
            amp   = 95 + base + carrier + noise
        elif self.skeleton.activity == "sitting":
            noise = random.gauss(0, 4)
            amp   = 70 + base * 0.4 + noise
        else:
            noise = random.gauss(0, 6)
            amp   = 82 + base * 0.65 + carrier * 0.3 + noise

        if random.random() > 0.93:
            amp += random.choice([-1, 1]) * random.randint(20, 45)

        self.wave_history.append(float(np.clip(amp, 20, 180)))
        self._update_spectral_bands(dt)

    def _update_spectral_bands(self, dt: float):
        activity = self.skeleton.activity
        t = pygame.time.get_ticks() * 0.001
        # Simülasyonda her bant aktiviteye göre farklı enerji seviyesi alır.
        # Gerçek ESP32 verisinde bunlar Butterworth band-pass çıktısından gelir.
        if activity == "walking":
            low  = 15 + math.sin(t * 0.3) * 8 + random.gauss(0, 3)
            mid  = 72 + math.sin(t * 1.2) * 18 + random.gauss(0, 8)
            high = 45 + math.sin(t * 4.5) * 20 + random.gauss(0, 10)
        elif activity == "sitting":
            low  = 42 + math.sin(t * 0.22) * 14 + random.gauss(0, 4)
            mid  = 18 + math.sin(t * 0.6) * 8  + random.gauss(0, 4)
            high = 8  + random.gauss(0, 3)
        else:  # standing
            low  = 28 + math.sin(t * 0.18) * 10 + random.gauss(0, 3)
            mid  = 22 + math.sin(t * 0.5) * 7   + random.gauss(0, 4)
            high = 12 + random.gauss(0, 3)

        # Smooth geçiş: mevcut son değere doğru lerp
        alpha = min(1.0, dt * 6.0)
        last_low  = self._band_low[-1]
        last_mid  = self._band_mid[-1]
        last_high = self._band_high[-1]
        self._band_low.pop(0);  self._band_low.append(float(np.clip(last_low  + (low  - last_low)  * alpha, 0, 100)))
        self._band_mid.pop(0);  self._band_mid.append(float(np.clip(last_mid  + (mid  - last_mid)  * alpha, 0, 100)))
        self._band_high.pop(0); self._band_high.append(float(np.clip(last_high + (high - last_high) * alpha, 0, 100)))

    def _update_csi_heatmap(self):
        t = pygame.time.get_ticks() * 0.001
        for row in range(8):
            for col in range(16):
                base = 40 + 30 * math.sin(t * 1.2 + col * 0.4) * math.cos(t * 0.7 + row * 0.5)
                if self.skeleton.activity == "walking":
                    base += 25 * math.sin(t * 4.0 + col * 0.3)
                base += random.gauss(0, 5)
                self._csi_heatmap_data[row, col] = float(np.clip(base, 0, 100))

    def _update_rssi_cells(self):
        ar, ac = self.current_row, self.current_col
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                label = self.grid_labels[r][c]
                dist  = math.sqrt((r - ar) ** 2 + (c - ac) ** 2)
                base  = -42 - int(dist * 11)
                noise = random.randint(-5, 5)
                self.rssi_cells[label] = max(-87, min(-38, base + noise))
        self.rssi_active = self.rssi_cells.get(self.current_region, -70)

    def _update_vitals(self, dt: float):
        activity = self.skeleton.activity
        if activity == "walking":
            self._bpm_target  = random.gauss(95, 3)
            self._resp_target = random.gauss(20, 1)
            self._bpm_state   = "ELEVATED"
            self._resp_state  = "ELEVATED"
        elif activity == "sitting":
            self._bpm_target  = random.gauss(65, 2)
            self._resp_target = random.gauss(14, 0.5)
            self._bpm_state   = "RESTING"
            self._resp_state  = "NORMAL"
        else:
            self._bpm_target  = random.gauss(72, 2)
            self._resp_target = random.gauss(16, 0.5)
            self._bpm_state   = "STABLE"
            self._resp_state  = "NORMAL"

        self._bpm      += (self._bpm_target  - self._bpm)      * dt * 0.8
        self._resp_rate += (self._resp_target - self._resp_rate) * dt * 0.6
        self._bpm_display = int(round(self._bpm))

        self._ekg_phase += dt * (self._bpm / 60.0) * 2 * math.pi
        self._ekg_history.pop(0)

        phase_mod = self._ekg_phase % (2 * math.pi)
        if 0.0 < phase_mod < 0.12:
            ekv = math.sin(phase_mod / 0.12 * math.pi) * 0.3
        elif 0.12 <= phase_mod < 0.22:
            ekv = -0.2
        elif 0.22 <= phase_mod < 0.36:
            peak_phase = (phase_mod - 0.22) / 0.14
            ekv = math.sin(peak_phase * math.pi) * 1.0
        elif 0.36 <= phase_mod < 0.5:
            ekv = -0.15
        elif 0.5 <= phase_mod < 0.7:
            ekv = math.sin((phase_mod - 0.5) / 0.2 * math.pi) * 0.25
        else:
            ekv = 0.0
        self._ekg_history.append(ekv + random.gauss(0, 0.02))

    # ═══════════════════════════════════════════════════════════════════════════
    # KLAVYE & FARE
    # ═══════════════════════════════════════════════════════════════════════════
    def _handle_key(self, event: pygame.event.Event):
        if self.cfg_active_field is not None:
            field   = self.cfg_active_field
            target  = "cfg_w_str" if field == "w" else "cfg_l_str"
            current = getattr(self, target)
            if event.key == pygame.K_BACKSPACE:
                setattr(self, target, current[:-1])
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._apply_config()
                self.cfg_active_field = None
            elif event.key == pygame.K_TAB:
                self.cfg_active_field = "l" if field == "w" else "w"
            elif event.unicode in "0123456789.":
                if len(current) < 6:
                    setattr(self, target, current + event.unicode)
            return

        if event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.key == pygame.K_m:
            self.is_ml_mode = not self.is_ml_mode
            mode = "ML PREDICT" if self.is_ml_mode else "MANUAL_SIM"
            self._log(f">> MODE SWITCH: {mode}")
        elif event.key == pygame.K_t:
            self.ml_engine.train_model()
            self._log(">> ML ENGINE: Training initiated...")
        elif not self.is_ml_mode:
            if event.key in self._key_map:
                r, c = self._key_map[event.key]
                self._request_zone_change(r, c)
            elif event.key in self._arrow_map:
                dr, dc = self._arrow_map[event.key]
                nr = max(0, min(self.grid_rows - 1, self.current_row + dr))
                nc = max(0, min(self.grid_cols - 1, self.current_col + dc))
                self._request_zone_change(nr, nc)

    def _handle_click(self, pos: Tuple[int, int]):
        mx, my = pos

        if self.cfg_w_rect.collidepoint(mx, my):
            self.cfg_active_field = "w"
            return
        if self.cfg_l_rect.collidepoint(mx, my):
            self.cfg_active_field = "l"
            return
        if self.cfg_btn_rect.collidepoint(mx, my):
            self._apply_config()
            return
        if self.reset_btn_rect.collidepoint(mx, my):
            self._log("!! CRITICAL: ESP32 Serial Interface resetting...")
            if self._esp32:
                self._esp32.stop()
                self._esp32 = None
            self._connect_esp32()
            if self._esp32 and self._esp32.connected:
                self._log(f"   Reconnected: {self._esp32_port}")
            else:
                self._log("   Port re-initialized. Awaiting handshake...")
            return

        for label, rect in self.regions.items():
            if rect.collidepoint(mx, my):
                for r in range(self.grid_rows):
                    for c in range(self.grid_cols):
                        if self.grid_labels[r][c] == label:
                            self._request_zone_change(r, c)
                            return

        self.cfg_active_field = None

    # ── Hysteresis: bölge değişim isteği ──────────────────────────────────────
    def _request_zone_change(self, r: int, c: int):
        if r < 0 or r >= self.grid_rows or c < 0 or c >= self.grid_cols:
            return
        candidate = self.grid_labels[r][c]
        if candidate == self._hyst_candidate:
            self._hyst_count += 1
        else:
            self._hyst_candidate = candidate
            self._hyst_count     = 1

        # ML modunda proba katmanı zaten histerezis yapıyor — burayı yumuşat.
        # Manuel modda kullanıcı tıklamasi/tusu icin 3 ardisik bekle.
        threshold = 2 if self.is_ml_mode else self._HYST_THRESHOLD
        if self._hyst_count >= threshold:
            if self.current_region != candidate:
                old = self.current_region
                self.current_region = candidate
                self.current_row    = r
                self.current_col    = c
                self.skeleton.set_grid_position(
                    r, c, grid_dims=(self.grid_rows, self.grid_cols))
                self._skel_walk_timer = 1.2
                self._log(f">> ZONE: {old} → {candidate}")
                print(f"[ZONE CHANGE] {old} → {candidate}", flush=True)
                # NOT: _ml_proba_buffer.clear() YAPMIYORUZ — sliding window
                # dogal olarak yeni zone'a kayacak. Clear edersek 5-10 paket
                # boyunca az veriyle karar verme zorunda kaliriz ve geri-donus
                # gerceklesir.
            self._hyst_count = 0

    # ── Config Apply ──────────────────────────────────────────────────────────
    def _apply_config(self):
        try:
            w = float(self.cfg_w_str)
            l = float(self.cfg_l_str)
            if w <= 0.5 or l <= 0.5:
                raise ValueError
        except ValueError:
            self._log("!! CONFIG ERROR: Invalid metric values.")
            return

        self.cfg_w_val = w
        self.cfg_l_val = l
        self._rebuild_grid()

        cols, rows = self.grid_cols, self.grid_rows
        cw = w / cols
        cl = l / rows
        self._log(f">> CONFIG: {w:.1f}m x {l:.1f}m → {cols}x{rows} grid")
        self._log(f"   Cell size: {cw:.2f}m x {cl:.2f}m")
        if self.cfg_warn:
            self._log("!! WARN: LOW SPATIAL RESOLUTION detected.")

    # ── Terminal Log ──────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.diag_log.append(f"[{ts}] {msg}")
        if len(self.diag_log) > 12:
            self.diag_log.pop(0)

    # ═══════════════════════════════════════════════════════════════════════════
    # ÇİZİM — ANA ORKESTRA
    # ═══════════════════════════════════════════════════════════════════════════
    def _draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_scanline_bg()
        self._draw_left_panel()
        self._draw_3d_viewport()
        self._draw_right_panel()
        self._draw_warn_overlay()
        pygame.display.flip()

    # ── Scanline Arka Plan ────────────────────────────────────────────────────
    def _draw_scanline_bg(self):
        scan = pygame.Surface((TOTAL_W, TOTAL_H), pygame.SRCALPHA)
        for y in range(0, TOTAL_H, 3):
            pygame.draw.line(scan, SCANLINE_COL, (0, y), (TOTAL_W, y))
        self.screen.blit(scan, (0, 0))

    # ═══════════════════════════════════════════════════════════════════════════
    # SOL PANEL — Grid + Grafikler + Config
    # ═══════════════════════════════════════════════════════════════════════════
    def _draw_left_panel(self):
        panel_bg = pygame.Rect(0, 0, LEFT_W, TOTAL_H)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_bg)
        pygame.draw.line(self.screen, BORDER_COLOR, (LEFT_W, 0), (LEFT_W, TOTAL_H), 2)

        self._draw_panel_header()
        self._draw_heatmap()
        self._draw_grid_panel()

        grid_bottom = self._get_grid_bottom() + 8

        # ── CSI Dalga Grafiği (sol alt, tam genişlik) ─────────────────────────
        graph_h  = 120
        graph_x  = 12
        graph_w  = LEFT_W - 24
        graph_y  = grid_bottom
        self._graph_rect = pygame.Rect(graph_x, graph_y, graph_w, graph_h)
        self._draw_wave_graph()

        # ── Config paneli (grafik altında, sol yarı) ──────────────────────────
        cfg_w   = LEFT_W // 2 - 18
        cfg_h   = 138
        cfg_x   = 12
        cfg_y   = graph_y + graph_h + 6
        if cfg_y + cfg_h <= TOTAL_H - 6:
            self._cfg_panel_rect = pygame.Rect(cfg_x, cfg_y, cfg_w, cfg_h)
            self.draw_config_panel(self._cfg_panel_rect)

        # ── CSI Heatmap (grafik ve config altında, sağ yarı veya tam) ─────────
        hmap_x = cfg_x + cfg_w + 8
        hmap_w = LEFT_W - hmap_x - 12
        hmap_y = graph_y + graph_h + 6
        hmap_h = TOTAL_H - hmap_y - 8
        if hmap_h > 50:
            self._csi_hmap_rect = pygame.Rect(hmap_x, hmap_y, hmap_w, hmap_h)
            self._draw_csi_heatmap_panel()

        # ── Warn uyarısı altına yazı (config altı) ────────────────────────────
        if self.cfg_warn:
            warn_y = cfg_y + cfg_h + 4
            if warn_y + 28 < TOTAL_H:
                self._draw_low_res_warning_inline(12, warn_y, cfg_w)

    def _draw_panel_header(self):
        t = pygame.time.get_ticks() * 0.001
        header_rect = pygame.Rect(0, 0, LEFT_W, 24)
        pygame.draw.rect(self.screen, (8, 20, 14), header_rect)
        title = self.f_title_lg.render(
            f"// FLOOR PLAN  [{self.cfg_w_val:.1f}m × {self.cfg_l_val:.1f}m]"
            f"  GRID:{self.grid_cols}×{self.grid_rows}",
            True, ACTIVE_COLOR
        )
        self.screen.blit(title, (12, 3))
        dot_x = LEFT_W - 30 + int(math.sin(t * 3) * 8)
        pygame.draw.circle(self.screen, ACTIVE_COLOR, (dot_x, 12), 4)
        pygame.draw.circle(self.screen, ACTIVE_DIM,   (dot_x, 12), 6, 1)

    def _get_grid_bottom(self) -> int:
        max_y = 28
        for rect in self.regions.values():
            if rect.bottom > max_y:
                max_y = rect.bottom
        return max_y

    # ── Heatmap Dolgu ────────────────────────────────────────────────────────
    def _draw_heatmap(self):
        for label, rect in self.regions.items():
            rssi      = self.rssi_cells.get(label, -70)
            is_active = (label == self.current_region)
            fill      = _heatmap_fill(rssi)
            surf      = pygame.Surface((rect.width - 4, rect.height - 4), pygame.SRCALPHA)

            if is_active:
                pulse  = abs(math.sin(self._blink_t * 2.8))
                base_a = 28 + int(48 * pulse)
                surf.fill((0, 230, 118, base_a))
                glow   = pygame.Surface((rect.width + 8, rect.height + 8), pygame.SRCALPHA)
                glow.fill((0, 230, 118, int(22 * pulse)))
                self.screen.blit(glow, (rect.x - 4, rect.y - 4))
            else:
                surf.fill(fill)
            self.screen.blit(surf, (rect.x + 2, rect.y + 2))

    def _zone_confidence(self, label: str) -> Optional[float]:
        """ML proba buffer'dan bu zone'un ortalama olasılığını döndürür (0-1), yoksa None."""
        if not self._ml_proba_buffer or not self._ml_classes:
            return None
        avg = np.mean(np.stack(self._ml_proba_buffer), axis=0)
        total = self.grid_rows * self.grid_cols
        zone_map = {}
        for idx in range(total):
            r = idx // self.grid_cols
            c = idx % self.grid_cols
            lbl = self.grid_labels[r][c]
            if idx == 0:
                zone_map[lbl] = "zone_A"
            elif idx == total - 1:
                zone_map[lbl] = "zone_B"
        ml_label = zone_map.get(label)
        if ml_label and ml_label in self._ml_classes:
            return float(avg[self._ml_classes.index(ml_label)])
        return None

    # ── 2D Grid Kutuları ──────────────────────────────────────────────────────
    def _draw_grid_panel(self):
        for label, rect in self.regions.items():
            is_active  = (label == self.current_region)
            is_pending = (label == self._hyst_candidate and
                          self._hyst_candidate != self.current_region and
                          self._hyst_count > 0)

            bw    = 3 if is_active else (2 if is_pending else 1)
            color = (ACTIVE_COLOR if is_active
                     else (WARN_YELLOW if is_pending else PASSIVE_COLOR))
            pygame.draw.rect(self.screen, color, rect, bw, border_radius=6)

            font = self.f_grid if self.grid_cols <= 4 else self.f_grid_sm
            lbl_surf = font.render(label, True, color)
            self.screen.blit(lbl_surf, lbl_surf.get_rect(
                centerx=rect.centerx, centery=rect.centery - 6))

            if rect.height > 55:
                rssi    = self.rssi_cells.get(label, -70)
                rc      = _rssi_color(rssi)
                rssi_s  = self.f_xs.render(f"{rssi}dBm", True, rc)
                self.screen.blit(rssi_s, (rect.x + 4, rect.bottom - 18))

            if is_pending and self._hyst_count > 0:
                bar_w = int(rect.width * self._hyst_count / self._HYST_THRESHOLD)
                pygame.draw.rect(self.screen, WARN_YELLOW,
                                 pygame.Rect(rect.x + 2, rect.bottom - 6,
                                             bar_w - 4, 4), border_radius=2)

            if rect.width > 90 and rect.height > 65:
                cw = self.cfg_w_val / self.grid_cols
                cl = self.cfg_l_val / self.grid_rows
                dim_s = self.f_xs.render(f"{cw:.1f}×{cl:.1f}m", True, TEXT_DIM)
                self.screen.blit(dim_s, (rect.x + 4, rect.y + 4))

            # Zone confidence — ML olasılığı varsa sağ alt köşeye yaz
            conf = self._zone_confidence(label)
            if conf is not None and rect.height > 40:
                conf_pct = int(conf * 100)
                conf_col = (ACTIVE_COLOR if conf_pct > 60
                            else (WARN_YELLOW if conf_pct > 30 else TEXT_DIM))
                conf_s = self.f_xs.render(f"{conf_pct}%", True, conf_col)
                self.screen.blit(conf_s, (rect.right - conf_s.get_width() - 4, rect.y + 4))

        # Avatar (2D grid üzerinde)
        ax, ay = int(self.avatar_x), int(self.avatar_y)
        halo = pygame.Surface((56, 56), pygame.SRCALPHA)
        pulse = abs(math.sin(self._blink_t * 3.2))
        pygame.draw.circle(halo, (255, 35, 102, int(35 * pulse)), (28, 28), 28)
        self.screen.blit(halo, (ax - 28, ay - 28))
        pygame.draw.circle(self.screen, AVATAR_COLOR, (ax, ay), 9)
        pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), 3)

        if self._skel_walk_timer > 0.3:
            ar = self.regions.get(self.current_region)
            if ar:
                arrow_alpha = int(200 * (self._skel_walk_timer / 1.2))
                asurf = pygame.Surface((20, 20), pygame.SRCALPHA)
                pygame.draw.polygon(asurf, (*ACTIVE_COLOR, arrow_alpha),
                                    [(10, 0), (20, 20), (10, 14), (0, 20)])
                self.screen.blit(asurf, (ax - 10, ay - 28))

    # ── CSI Dalga Grafiği ─────────────────────────────────────────────────────
    def _signal_morphology(self) -> tuple:
        """Son 30 frame'den varyans ve trend hesaplayarak sinyal durumunu döndürür."""
        if len(self.wave_history) < 30:
            return "ANALYZING...", TEXT_DIM
        recent = self.wave_history[-30:]
        variance = float(np.var(recent))
        mean_amp = float(np.mean(recent))
        trend = recent[-1] - recent[0]

        if variance > 280:
            return "HIGH TURBULENCE  — MOTION DETECTED", WARN_ORANGE
        if variance > 120:
            if trend > 8:
                return "RISING AMPLITUDE — APPROACHING", WARN_YELLOW
            if trend < -8:
                return "FALLING AMPLITUDE — RECEDING", NEON_CYAN
            return "MODERATE VARIANCE — ACTIVE PRESENCE", GRAPH_COLOR
        if mean_amp < 55:
            return "LOW AMPLITUDE — AREA CLEAR", TEXT_DIM
        return "STABLE SIGNAL — STATIONARY TARGET", ACTIVE_COLOR

    def _draw_wave_graph(self):
        r = self._graph_rect
        # Paneli wave bölgesi + band bölgesi olarak ikiye böl
        band_h   = 32
        wave_h   = r.height - band_h - 4
        wave_r   = pygame.Rect(r.x, r.y, r.width, wave_h)
        band_r   = pygame.Rect(r.x, r.y + wave_h + 4, r.width, band_h)
        self._band_rect = band_r

        pygame.draw.rect(self.screen, PANEL_DARK, r, border_radius=6)
        pygame.draw.rect(self.screen, BORDER_COLOR, r, 1, border_radius=6)

        # Başlık
        hdr = self.f_sm.render("// LIVE CSI SIGNAL  [SUB-CARRIER AMPLITUDE]", True, GRAPH_COLOR)
        self.screen.blit(hdr, (wave_r.x + 8, wave_r.y + 5))

        # Morfoloji etiketi — sağ üst
        morph_label, morph_col = self._signal_morphology()
        morph_surf = self.f_xs.render(morph_label, True, morph_col)
        self.screen.blit(morph_surf, (wave_r.right - morph_surf.get_width() - 8, wave_r.y + 6))

        # Yatay ızgara çizgileri
        for i, pct in enumerate((0.25, 0.5, 0.75)):
            gy = wave_r.y + int(wave_h * pct)
            pygame.draw.line(self.screen, (22, 32, 46), (wave_r.x + 2, gy), (wave_r.right - 2, gy))
            val = int(180 * (1 - pct) + 20 * pct)
            lbl = self.f_xs.render(str(val), True, TEXT_DIM)
            self.screen.blit(lbl, (wave_r.x + 2, gy - 8))

        # Dalga çizimi
        pts = []
        step = wave_r.width / max(len(self.wave_history) - 1, 1)
        for i, amp in enumerate(self.wave_history):
            x = wave_r.x + i * step
            y = wave_r.y + wave_h * 0.82 - amp * (wave_h * 0.55 / 160)
            y = max(wave_r.y + 18, min(y, wave_r.bottom - 4))
            pts.append((int(x), int(y)))

        if len(pts) > 1:
            fill_pts = [(wave_r.x, wave_r.bottom - 4)] + pts + [(wave_r.right, wave_r.bottom - 4)]
            fill_surf = pygame.Surface((wave_r.width, wave_h), pygame.SRCALPHA)
            offset_pts = [(p[0] - wave_r.x, p[1] - wave_r.y) for p in fill_pts]
            pygame.draw.polygon(fill_surf, (0, 191, 255, 22), offset_pts)
            self.screen.blit(fill_surf, (wave_r.x, wave_r.y))

            shadow_pts = [(x, y + 2) for x, y in pts]
            pygame.draw.lines(self.screen, GRAPH_DIM, False, shadow_pts, 1)
            pygame.draw.lines(self.screen, GRAPH_COLOR, False, pts, 2)

            lx, ly = pts[-1]
            pygame.draw.circle(self.screen, GRAPH_COLOR, (lx, ly), 5)
            pygame.draw.circle(self.screen, (255, 255, 255), (lx, ly), 2)
            live_v = self.f_sm.render(f"{self.wave_history[-1]:.0f}", True, TEXT_BRIGHT)
            self.screen.blit(live_v, (lx - 20, ly - 16))

        last_jump = abs(self.wave_history[-1] - self.wave_history[-2]) if len(self.wave_history) >= 2 else 0
        if last_jump > 25:
            jlbl = self.f_xs.render(f"PHASE JUMP +{last_jump:.0f}", True, WARN_ORANGE)
            self.screen.blit(jlbl, (wave_r.x + wave_r.width - 110, wave_r.y + wave_h - 14))

        # ── Spektral Band Çubukları ────────────────────────────────────────────
        self._draw_spectral_bands(band_r)

    def _draw_spectral_bands(self, r: pygame.Rect):
        pygame.draw.line(self.screen, BORDER_COLOR,
                         (r.x + 2, r.y), (r.right - 2, r.y))

        bands = [
            ("RESP", self._band_low[-1],  (80, 200, 255)),   # mavi — solunum
            ("WALK", self._band_mid[-1],  (255, 210, 0)),    # sarı — yürüyüş
            ("MOVE", self._band_high[-1], (255, 80, 120)),   # kırmızı — hızlı
        ]

        label_w = 32
        gap      = 6
        bar_area_w = (r.width - label_w * len(bands) - gap * (len(bands) + 1)) // len(bands)
        bar_max_h  = r.height - 14

        for i, (name, val, col) in enumerate(bands):
            bx = r.x + gap + i * (bar_area_w + label_w + gap)
            # Etiket
            lbl = self.f_xs.render(name, True, col)
            self.screen.blit(lbl, (bx, r.y + 2))
            # Çubuk arka planı
            bar_bg = pygame.Rect(bx, r.y + 12, bar_area_w + label_w - 2, bar_max_h)
            pygame.draw.rect(self.screen, (14, 20, 30), bar_bg, border_radius=2)
            # Dolu kısım
            fill_h = int(bar_max_h * val / 100.0)
            if fill_h > 0:
                fill_rect = pygame.Rect(bx, r.bottom - fill_h - 2, bar_area_w + label_w - 2, fill_h)
                # Gradient efekti: üst kısım parlak, alt kısım sönük
                glow_surf = pygame.Surface((fill_rect.width, fill_rect.height), pygame.SRCALPHA)
                for gy in range(fill_rect.height):
                    alpha = int(80 + 160 * (1 - gy / max(fill_rect.height, 1)))
                    pygame.draw.line(glow_surf, (*col, alpha),
                                     (0, gy), (fill_rect.width, gy))
                self.screen.blit(glow_surf, (fill_rect.x, fill_rect.y))
            # Değer metni
            pct_s = self.f_xs.render(f"{int(val)}", True, col)
            self.screen.blit(pct_s, (bx + bar_area_w + label_w - pct_s.get_width() - 2, r.y + 2))

    # ── CSI Heatmap Paneli ────────────────────────────────────────────────────
    def _draw_csi_heatmap_panel(self):
        r = self._csi_hmap_rect
        if r.height < 50:
            return
        pygame.draw.rect(self.screen, PANEL_DARK, r, border_radius=6)
        pygame.draw.rect(self.screen, BORDER_COLOR, r, 1, border_radius=6)

        hdr = self.f_sm.render(
            "// CSI HEATMAP  [8×16 SUBCARRIERS]", True, NEON_CYAN)
        self.screen.blit(hdr, (r.x + 8, r.y + 4))

        rows_h, cols_h = self._csi_heatmap_data.shape
        cell_w = (r.width - 22) / cols_h
        cell_h = max(4, (r.height - 20) / rows_h)
        start_y = r.y + 18

        for row in range(rows_h):
            for col in range(cols_h):
                val  = self._csi_heatmap_data[row, col]
                t    = val / 100.0
                rr   = int(10  + 245 * t)
                gg   = int(100 * (1 - abs(t - 0.5) * 2) + 40 * t)
                bb   = int(200 * (1 - t) + 10 * t)
                cell = pygame.Rect(
                    int(r.x + 8 + col * cell_w),
                    int(start_y + row * cell_h),
                    max(1, int(cell_w) - 1),
                    max(1, int(cell_h) - 1)
                )
                pygame.draw.rect(self.screen, (rr, gg, bb), cell)

        legend_x = r.x + r.width - 14
        legend_h  = r.height - 24
        for i in range(legend_h):
            t  = 1 - i / legend_h
            rr = int(10  + 245 * t)
            gg = int(100 * (1 - abs(t - 0.5) * 2) + 40 * t)
            bb = int(200 * (1 - t))
            pygame.draw.line(self.screen, (rr, gg, bb),
                             (legend_x, r.y + 20 + i), (legend_x + 8, r.y + 20 + i))
        high_l = self.f_xs.render("HI", True, (255, 200, 50))
        low_l  = self.f_xs.render("LO", True, (50, 100, 200))
        self.screen.blit(high_l, (legend_x - 2, r.y + 20))
        self.screen.blit(low_l,  (legend_x - 2, r.bottom - 16))

    # ── Inline düşük çözünürlük uyarısı ──────────────────────────────────────
    def _draw_low_res_warning_inline(self, x: int, y: int, w: int):
        blink = int(self._blink_t * 2.8) % 2 == 0
        col   = WARN_RED if blink else WARN_YELLOW
        bg    = pygame.Surface((w, 26), pygame.SRCALPHA)
        bg.fill((80, 12, 12, 160))
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, col, pygame.Rect(x, y, w, 26), 1, border_radius=3)
        s1 = self.f_warn.render("⚠  LOW SPATIAL RESOLUTION", True, col)
        self.screen.blit(s1, (x + 6, y + 4))

    # ═══════════════════════════════════════════════════════════════════════════
    # KONFİGÜRASYON PANELİ
    # ═══════════════════════════════════════════════════════════════════════════
    def draw_config_panel(self, r: pygame.Rect):
        pygame.draw.rect(self.screen, PANEL_DARK, r, border_radius=6)
        pygame.draw.rect(self.screen, BORDER_BRIGHT, r, 1, border_radius=6)

        for corner in [(r.x, r.y), (r.right - 8, r.y),
                       (r.x, r.bottom - 8), (r.right - 8, r.bottom - 8)]:
            pygame.draw.rect(self.screen, ACTIVE_COLOR,
                             pygame.Rect(corner[0], corner[1], 8, 2))
            pygame.draw.rect(self.screen, ACTIVE_COLOR,
                             pygame.Rect(corner[0], corner[1], 2, 8))

        title = self.f_body.render("// AREA CONFIG", True, ACTIVE_DIM)
        self.screen.blit(title, (r.x + 10, r.y + 6))

        cw = self.cfg_w_val / max(self.grid_cols, 1)
        cl = self.cfg_l_val / max(self.grid_rows, 1)
        area_col  = WARN_YELLOW if self.cfg_warn else ACTIVE_COLOR
        info_strs = [
            f"AREA: {self.cfg_w_val*self.cfg_l_val:.1f} m\xb2",
            f"CELL: {cw:.2f}x{cl:.2f}m",
            f"GRID: {self.grid_cols}col x {self.grid_rows}row",
        ]
        for i, s in enumerate(info_strs):
            surf = self.f_xs.render(s, True, area_col)
            self.screen.blit(surf, (r.x + 10, r.y + 22 + i * 13))

        def draw_input(label: str, value: str, field_id: str, y_off: int) -> pygame.Rect:
            lbl_s = self.f_xs.render(label, True, TEXT_DIM)
            self.screen.blit(lbl_s, (r.x + 10, r.y + y_off))
            box = pygame.Rect(r.x + 10, r.y + y_off + 10, r.width - 20, 16)
            is_focused = (self.cfg_active_field == field_id)
            pygame.draw.rect(self.screen,
                             (0, 55, 38) if is_focused else PANEL_COLOR,
                             box, border_radius=3)
            pygame.draw.rect(self.screen,
                             ACTIVE_COLOR if is_focused else BORDER_COLOR,
                             box, 1, border_radius=3)
            cursor = "|" if is_focused and int(self._blink_t * 2) % 2 == 0 else ""
            val_s = self.f_input.render(value + cursor, True, TEXT_COLOR)
            self.screen.blit(val_s, (box.x + 5, box.y + 2))
            return box

        self.cfg_w_rect = draw_input("WIDTH (m):",  self.cfg_w_str, "w", 62)
        self.cfg_l_rect = draw_input("LENGTH (m):", self.cfg_l_str, "l", 90)

        btn = pygame.Rect(r.x + 10, r.bottom - 24, r.width - 20, 20)
        mx, my = pygame.mouse.get_pos()
        btn_hov = btn.collidepoint(mx, my)
        pygame.draw.rect(self.screen,
                         BUTTON_HOVER if btn_hov else BUTTON_NORMAL,
                         btn, border_radius=4)
        pygame.draw.rect(self.screen,
                         ACTIVE_COLOR if btn_hov else BORDER_COLOR,
                         btn, 1, border_radius=4)
        lbl = self.f_sm.render("[ APPLY & RESCALE ]", True,
                                (8, 8, 8) if btn_hov else TEXT_COLOR)
        self.screen.blit(lbl, lbl.get_rect(center=btn.center))
        self.cfg_btn_rect = btn

    # ═══════════════════════════════════════════════════════════════════════════
    # ORTA PANEL — 3D VIEWPORT
    # ═══════════════════════════════════════════════════════════════════════════
    def _draw_3d_viewport(self):
        vp_rect = pygame.Rect(LEFT_W, 0, VIEW3D_W, TOTAL_H)
        pygame.draw.rect(self.screen, DIVIDER, vp_rect)
        pygame.draw.line(self.screen, BORDER_COLOR, (LEFT_W, 0), (LEFT_W, TOTAL_H), 2)
        pygame.draw.line(self.screen, BORDER_COLOR,
                         (LEFT_W + VIEW3D_W, 0), (LEFT_W + VIEW3D_W, TOTAL_H), 2)

        hdr_band = pygame.Rect(LEFT_W, 0, VIEW3D_W, 44)
        pygame.draw.rect(self.screen, (8, 14, 20), hdr_band)
        hdr = self.f_title.render("// 3D SKELETON FIELD  [TACTICAL VIEW]", True, ACTIVE_COLOR)
        self.screen.blit(hdr, (LEFT_W + 10, 5))
        hint = self.f_xs.render(
            "drag:rotate   scroll:zoom   [M]:ML   arrows/1-9:zone", True, TEXT_DIM)
        self.screen.blit(hint, (LEFT_W + 10, 24))

        zone_badge = self.f_body.render(
            f"ZONE: {self.current_region}  |  {self.cfg_w_val/self.grid_cols:.1f}x"
            f"{self.cfg_l_val/self.grid_rows:.1f}m cell",
            True, WARN_YELLOW)
        self.screen.blit(zone_badge, (LEFT_W + 10, 46))

        sub_h = TOTAL_H - 90
        sub   = pygame.Surface((VIEW3D_W, sub_h))
        self.renderer3d.render_fallback_to_surface(
            sub, [self.skeleton],
            (self.current_row, self.current_col),
            grid_dims=(self.grid_rows, self.grid_cols))
        self.screen.blit(sub, (LEFT_W, 66))

        act     = self.skeleton.activity.upper()
        act_col = (ACTIVE_COLOR if act in ("STANDING", "WALKING")
                   else (GRAPH_COLOR if act == "SITTING" else TEXT_DIM))
        badge   = self.f_body.render(f"[ {act} ]", True, act_col)
        self.screen.blit(badge, badge.get_rect(
            centerx=LEFT_W + VIEW3D_W // 2, y=TOTAL_H - 32))

        if self._skel_walk_timer > 0:
            alpha = int(180 * (self._skel_walk_timer / 1.2))
            t     = pygame.time.get_ticks() * 0.002
            for i in range(6):
                fx  = LEFT_W + VIEW3D_W // 2 + int(math.sin(t + i) * 30)
                fy  = TOTAL_H - 90 - i * 14
                dot = pygame.Surface((8, 4), pygame.SRCALPHA)
                dot.fill((*ACTIVE_COLOR, alpha // (i + 1)))
                self.screen.blit(dot, (fx - 4, fy - 2))

        # Vital Signs mini panel — viewport sol alt köşesine
        self._draw_vital_signs_mini(LEFT_W + 6, TOTAL_H - 210)

    # ─── Vital Signs Mini Panel ───────────────────────────────────────────────
    def _draw_vital_signs_mini(self, px: int, py: int):
        panel_w, panel_h = 200, 130
        if py < 70 or px + panel_w > LEFT_W + VIEW3D_W:
            return
        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((6, 12, 20, 215))
        self.screen.blit(bg, (px, py))
        border_col = VITAL_COLOR if int(self._vital_blink * 1.5) % 2 == 0 else ACTIVE_DIM
        pygame.draw.rect(self.screen, border_col,
                         pygame.Rect(px, py, panel_w, panel_h), 1, border_radius=4)

        hdr = self.f_vital.render("// VITAL SIGNS", True, VITAL_COLOR)
        self.screen.blit(hdr, (px + 6, py + 4))

        bpm_col = WARN_ORANGE if self._bpm > 88 else VITAL_GREEN
        bpm_lbl = self.f_vital.render(
            f"HR: {self._bpm_display} BPM  [{self._bpm_state}]", True, bpm_col)
        self.screen.blit(bpm_lbl, (px + 6, py + 20))

        resp_val = int(round(self._resp_rate))
        resp_lbl = self.f_vital_sm.render(
            f"RESP: {resp_val} Br/m  [{self._resp_state}]", True, GRAPH_COLOR)
        self.screen.blit(resp_lbl, (px + 6, py + 35))

        # Mini EKG çizgisi
        ekg_rect = pygame.Rect(px + 4, py + 50, panel_w - 8, 70)
        pygame.draw.rect(self.screen, (6, 8, 14), ekg_rect)
        pygame.draw.rect(self.screen, PASSIVE_COLOR, ekg_rect, 1)

        # İzgara çizgisi (EKG arka plan)
        mid_line = ekg_rect.centery
        pygame.draw.line(self.screen, (22, 32, 46),
                         (ekg_rect.x, mid_line), (ekg_rect.right, mid_line))

        ekg_pts = []
        step    = ekg_rect.width / max(len(self._ekg_history) - 1, 1)
        for i, v in enumerate(self._ekg_history):
            x = int(ekg_rect.x + i * step)
            y = int(mid_line - v * 24)
            y = max(ekg_rect.y + 2, min(y, ekg_rect.bottom - 3))
            ekg_pts.append((x, y))

        if len(ekg_pts) > 1:
            # Gölge çizgisi
            shadow = [(x, y + 1) for x, y in ekg_pts]
            pygame.draw.lines(self.screen, (80, 20, 35), False, shadow, 1)
            pygame.draw.lines(self.screen, VITAL_COLOR, False, ekg_pts, 2)

        if ekg_pts:
            lx, ly = ekg_pts[-1]
            pulse  = abs(math.sin(self._vital_blink * self._bpm / 60 * math.pi))
            r_size = int(2 + pulse * 3)
            glow_s = pygame.Surface((r_size * 4, r_size * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_s, (*VITAL_COLOR, 60), (r_size * 2, r_size * 2), r_size * 2)
            self.screen.blit(glow_s, (lx - r_size * 2, ly - r_size * 2))
            pygame.draw.circle(self.screen, VITAL_COLOR, (lx, ly), r_size)

    # ═══════════════════════════════════════════════════════════════════════════
    # SAĞ PANEL — TELEMETRİ + DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════════
    def _draw_right_panel(self):
        px  = LEFT_W + VIEW3D_W
        pw  = RIGHT_W
        panel = pygame.Rect(px, 0, pw, TOTAL_H)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel)
        pygame.draw.line(self.screen, BORDER_COLOR, (px, 0), (px, TOTAL_H), 2)

        title_band = pygame.Rect(px, 0, pw, 44)
        pygame.draw.rect(self.screen, (8, 22, 14), title_band)
        t_surf = self.f_title_lg.render("SYSTEM STATUS", True, ACTIVE_COLOR)
        self.screen.blit(t_surf, t_surf.get_rect(centerx=px + pw // 2, centery=22))

        y    = 50
        xpad = px + 10
        w_inner = pw - 20

        def clamp_text(font: pygame.font.Font, text: str, max_w: int, color: tuple) -> pygame.Surface:
            surf = font.render(text, True, color)
            if surf.get_width() > max_w:
                while len(text) > 2:
                    text = text[:-1]
                    surf = font.render(text + "..", True, color)
                    if surf.get_width() <= max_w:
                        break
            return surf

        def sep():
            nonlocal y
            pygame.draw.line(self.screen, PASSIVE_COLOR,
                             (px + 6, y + 3), (px + pw - 6, y + 3))
            y += 10

        def row(label: str, value: str, vcol: tuple = TEXT_COLOR):
            nonlocal y
            ls = self.f_xs.render(label, True, TEXT_DIM)
            vs = clamp_text(self.f_body, value, w_inner, vcol)
            self.screen.blit(ls, (xpad, y))
            self.screen.blit(vs, (xpad, y + 11))
            y += 28

        def section(title: str):
            nonlocal y
            pygame.draw.rect(self.screen, (14, 20, 30),
                             pygame.Rect(px + 4, y, pw - 8, 16))
            ts = self.f_warn.render(f"── {title} ──", True, ACTIVE_DIM)
            self.screen.blit(ts, ts.get_rect(centerx=px + pw // 2, y=y + 1))
            y += 18

        # ── COMMS ─────────────────────────────────────────────────────────────
        section("COMMS")
        esp32_live = self._esp32 is not None and self._esp32.connected
        if esp32_live and self.is_ml_mode:
            lnk_col  = ACTIVE_COLOR
            lnk_txt  = "[OPERATIONAL / CONNECTED]"
            mode_txt = "ML PREDICT  [LIVE]"
        elif esp32_live:
            lnk_col  = NEON_CYAN
            lnk_txt  = "[CONNECTED / SIM MODE]"
            mode_txt = "MANUAL_SIM  [ESP32 ON]"
        else:
            lnk_col  = WARN_YELLOW
            lnk_txt  = "[STANDBY / NO ESP32]"
            mode_txt = "MANUAL_SIM  [OFFLINE]"

        ls_surf = clamp_text(self.f_body, lnk_txt, w_inner, lnk_col)
        if esp32_live:
            glow_bg = pygame.Surface((ls_surf.get_width() + 8, ls_surf.get_height() + 4), pygame.SRCALPHA)
            glow_bg.fill((*ACTIVE_COLOR, 18))
            self.screen.blit(glow_bg, (xpad - 2, y + 11 - 1))
        self.f_xs.render("ESP32 LINK STATUS", True, TEXT_DIM)
        row("ESP32 LINK STATUS", lnk_txt, lnk_col)
        row("OPERATION MODE",    mode_txt, lnk_col)
        sep()

        # ── TARGET TRACKING ───────────────────────────────────────────────────
        section("TARGET TRACKING")
        row("ACTIVE SECTOR",  self.current_region, ACTIVE_COLOR)
        row("ACTIVITY STATE", self.skeleton.activity.upper(), ACTIVE_COLOR)
        row("RSSI SIGNAL",    f"{self.rssi_active}dBm [{_rssi_label(self.rssi_active)}]",
            _rssi_color(self.rssi_active))
        cw = self.cfg_w_val / max(self.grid_cols, 1)
        cl = self.cfg_l_val / max(self.grid_rows, 1)
        row("SECTOR SIZE",    f"{cw:.2f}m x {cl:.2f}m", TEXT_DIM)
        sep()

        # ── SIGNAL QUALITY ────────────────────────────────────────────────────
        section("SIGNAL QUALITY")
        cqi_disp = int(self.cqi + math.sin(self._cqi_phase) * 2)
        cqi_disp = max(0, min(100, cqi_disp))
        filled   = cqi_disp // 10
        bar      = "█" * filled + "░" * (10 - filled)
        cqi_col  = (ACTIVE_COLOR if cqi_disp > 60
                    else (WARN_YELLOW if cqi_disp > 30 else WARN_RED))

        # CQI satırı
        ls = self.f_xs.render("CQI (LINK QUALITY)", True, TEXT_DIM)
        self.screen.blit(ls, (xpad, y))
        bar_surf = clamp_text(self.f_body, f"[{bar}]", w_inner, cqi_col)
        self.screen.blit(bar_surf, (xpad, y + 11))
        pct_surf = self.f_xs.render(f"{cqi_disp}%", True, cqi_col)
        self.screen.blit(pct_surf, (xpad + w_inner - pct_surf.get_width(), y + 11))
        y += 28

        row("CSI PACKET RATE", f"{self.csi_rate} Hz", GRAPH_COLOR)
        row("RENDER FPS",      f"{int(self.clock.get_fps())} fps", TEXT_COLOR)
        sep()

        # ── VITAL SIGNS (sağ panel özet) ──────────────────────────────────────
        section("VITAL SIGNS")
        bpm_col = WARN_ORANGE if self._bpm > 88 else VITAL_GREEN
        row("HEART RATE",  f"{self._bpm_display} BPM  [{self._bpm_state}]", bpm_col)
        resp_val = int(round(self._resp_rate))
        row("RESP. RATE",  f"{resp_val} Br/m  [{self._resp_state}]", GRAPH_COLOR)

        # Mini flat EKG şeridi (sağ panel versiyonu)
        ekg_strip_h = 28
        ekg_strip = pygame.Rect(xpad, y, w_inner, ekg_strip_h)
        pygame.draw.rect(self.screen, PANEL_DARK, ekg_strip, border_radius=3)
        pygame.draw.rect(self.screen, PASSIVE_COLOR, ekg_strip, 1, border_radius=3)
        strip_pts = []
        step_s = ekg_strip.width / max(len(self._ekg_history) - 1, 1)
        mid_s  = ekg_strip.centery
        for i, v in enumerate(self._ekg_history):
            sx = int(ekg_strip.x + i * step_s)
            sy = int(mid_s - v * 10)
            sy = max(ekg_strip.y + 2, min(sy, ekg_strip.bottom - 3))
            strip_pts.append((sx, sy))
        if len(strip_pts) > 1:
            pygame.draw.lines(self.screen, VITAL_COLOR, False, strip_pts, 1)
        y += ekg_strip_h + 6
        sep()

        # ── GRID DIAGNOSTICS ──────────────────────────────────────────────────
        section("GRID DIAGNOSTICS")
        row("GRID MATRIX",   f"{self.grid_cols} x {self.grid_rows} cells", TEXT_COLOR)
        row("TOTAL SECTORS", f"{self.grid_cols * self.grid_rows}", TEXT_COLOR)
        row("ROOM AREA",     f"{self.cfg_w_val*self.cfg_l_val:.2f} m\xb2", TEXT_COLOR)
        hyst_str = (f"{self._hyst_count}/{self._HYST_THRESHOLD} "
                    f"[{self._hyst_candidate}]")
        row("HYSTERESIS",    hyst_str, WARN_YELLOW if self._hyst_count > 0 else TEXT_DIM)
        sep()

        # ── TERMINAL LOG ──────────────────────────────────────────────────────
        section("TERMINAL LOG")
        log_lines = self.diag_log[-8:] if self.diag_log else ["  [IDLE — NO EVENTS]"]
        for line in log_lines:
            col_l = (WARN_RED    if "!!" in line
                     else ACTIVE_COLOR if ">>" in line
                     else TEXT_DIM)
            surf = clamp_text(self.f_xs, line, w_inner, col_l)
            if y + 12 < TOTAL_H - 50:
                self.screen.blit(surf, (xpad, y))
                y += 12
        sep()

        # ── CONTROLS ──────────────────────────────────────────────────────────
        section("CONTROLS")
        ctrl_lines = [
            "[M] Toggle ML  [T] Train",
            "[1-9] Zone  [ESC] Exit",
            "[↑↓←→] Navigate zones",
            "3D: drag=rot  wheel=zoom",
        ]
        for ln in ctrl_lines:
            if y + 12 < TOTAL_H - 46:
                s = self.f_xs.render(ln, True, TEXT_DIM)
                self.screen.blit(s, (xpad, y))
                y += 12
        y += 4

        # ── LINK RESET BUTONU ─────────────────────────────────────────────────
        btn_h   = 32
        btn_y   = TOTAL_H - btn_h - 8
        btn     = pygame.Rect(px + 8, btn_y, pw - 16, btn_h)
        btn_col = BUTTON_RESET_HOV if self._reset_hover else BUTTON_RESET_NRM
        bdr_col = WARN_RED if self._reset_hover else (70, 18, 18)
        pygame.draw.rect(self.screen, btn_col, btn, border_radius=5)
        pygame.draw.rect(self.screen, bdr_col, btn, 1, border_radius=5)
        lbl_col = (255, 255, 255) if self._reset_hover else (180, 75, 75)

        if self._reset_hover:
            pulse = abs(math.sin(self._blink_t * 5))
            glow  = pygame.Surface((btn.width + 8, btn.height + 8), pygame.SRCALPHA)
            glow.fill((255, 50, 50, int(30 * pulse)))
            self.screen.blit(glow, (btn.x - 4, btn.y - 4))

        lbl = self.f_body.render("[ LINK RESET ]", True, lbl_col)
        self.screen.blit(lbl, lbl.get_rect(center=btn.center))
        self.reset_btn_rect = btn

    # ═══════════════════════════════════════════════════════════════════════════
    # UYARI OVERLAY
    # ═══════════════════════════════════════════════════════════════════════════
    def _draw_warn_overlay(self):
        if not self.cfg_warn:
            return
        blink = int(self._blink_t * 2.8) % 2 == 0
        col   = WARN_RED if blink else WARN_YELLOW

        grid_bottom = self._get_grid_bottom()
        warn_y = grid_bottom + 2
        warn_w = LEFT_W // 2 - 4

        lines = [
            "⚠  WARNING: LOW SPATIAL RESOLUTION",
            f"   Cell < {MIN_CELL_M:.0f}m — ±1m error margin violated!",
        ]

        warn_bg = pygame.Surface((warn_w, 30), pygame.SRCALPHA)
        warn_bg.fill((80, 12, 12, 175))
        self.screen.blit(warn_bg, (12, warn_y))
        pygame.draw.rect(self.screen, col,
                         pygame.Rect(12, warn_y, warn_w, 30), 1, border_radius=4)

        for i, ln in enumerate(lines):
            s = self.f_warn.render(ln, True, col)
            self.screen.blit(s, (20, warn_y + 4 + i * 13))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        os.system("osascript -e 'tell application \"Python\" to activate' &")
    except Exception:
        pass
    ui = Radar3x3UI()
    ui.run()
