import numpy as np
import math
from typing import Optional

# ─── İskelet Keypoint Tanımları ───────────────────────────────────────────────
# MediaPipe / Wi-Pose uyumlu 17 keypoint (CSI entegrasyonuna hazır)
KEYPOINTS = [
    "nose",          # 0
    "neck",          # 1
    "r_shoulder",    # 2
    "r_elbow",       # 3
    "r_wrist",       # 4
    "l_shoulder",    # 5
    "l_elbow",       # 6
    "l_wrist",       # 7
    "r_hip",         # 8
    "r_knee",        # 9
    "r_ankle",       # 10
    "l_hip",         # 11
    "l_knee",        # 12
    "l_ankle",       # 13
    "r_eye",         # 14
    "l_eye",         # 15
    "r_ear",         # 16
]

# Kemik bağlantıları (keypoint index çiftleri)
BONES = [
    (0, 1),   # burun → boyun
    (1, 2),   # boyun → sağ omuz
    (2, 3),   # sağ omuz → sağ dirsek
    (3, 4),   # sağ dirsek → sağ bilek
    (1, 5),   # boyun → sol omuz
    (5, 6),   # sol omuz → sol dirsek
    (6, 7),   # sol dirsek → sol bilek
    (1, 8),   # boyun → sağ kalça
    (8, 9),   # sağ kalça → sağ diz
    (9, 10),  # sağ diz → sağ ayak bileği
    (1, 11),  # boyun → sol kalça
    (11, 12), # sol kalça → sol diz
    (12, 13), # sol diz → sol ayak bileği
    (0, 14),  # burun → sağ göz
    (0, 15),  # burun → sol göz
    (14, 16), # sağ göz → sağ kulak
]

# Kemik renk grupları (render'da kullanılır)
BONE_GROUPS = {
    "spine":  [(0,1),(1,2),(1,5),(1,8),(1,11)],
    "r_arm":  [(2,3),(3,4)],
    "l_arm":  [(5,6),(6,7)],
    "r_leg":  [(8,9),(9,10)],
    "l_leg":  [(11,12),(12,13)],
    "face":   [(0,14),(0,15),(14,16)],
}

# ─── Duruş Şablonları (T-pose merkezli, normalize edilmiş) ────────────────────
def _pose_standing() -> np.ndarray:
    """Ayakta duran insan — tüm pozlar bu temelden türetilir."""
    p = np.zeros((17, 3), dtype=np.float32)
    p[0]  = [0.00,  1.70,  0]   # burun
    p[1]  = [0.00,  1.50,  0]   # boyun
    p[2]  = [ 0.20, 1.50,  0]   # sağ omuz
    p[3]  = [ 0.38, 1.20,  0]   # sağ dirsek
    p[4]  = [ 0.38, 0.90,  0]   # sağ bilek
    p[5]  = [-0.20, 1.50,  0]   # sol omuz
    p[6]  = [-0.38, 1.20,  0]   # sol dirsek
    p[7]  = [-0.38, 0.90,  0]   # sol bilek
    p[8]  = [ 0.10, 0.95,  0]   # sağ kalça
    p[9]  = [ 0.10, 0.50,  0]   # sağ diz
    p[10] = [ 0.10, 0.05,  0]   # sağ ayak bileği
    p[11] = [-0.10, 0.95,  0]   # sol kalça
    p[12] = [-0.10, 0.50,  0]   # sol diz
    p[13] = [-0.10, 0.05,  0]   # sol ayak bileği
    p[14] = [ 0.05, 1.73,  0]   # sağ göz
    p[15] = [-0.05, 1.73,  0]   # sol göz
    p[16] = [ 0.10, 1.70,  0]   # sağ kulak
    return p

def _pose_sitting() -> np.ndarray:
    p = _pose_standing().copy()
    # Kalça aşağı, bacaklar yatay
    p[8]  = [ 0.10, 0.90,  0]
    p[9]  = [ 0.35, 0.90,  0]
    p[10] = [ 0.35, 0.88,  0.30]
    p[11] = [-0.10, 0.90,  0]
    p[12] = [-0.35, 0.90,  0]
    p[13] = [-0.35, 0.88,  0.30]
    # Tüm vücut aşağı kaydır
    p -= [0, 0.85, 0]
    return p

def _pose_walking(phase: float) -> np.ndarray:
    """phase: 0..2π arası yürüyüş fazı"""
    p = _pose_standing().copy()
    swing = math.sin(phase)
    p[9]  += [0,  swing * 0.15, -swing * 0.12]
    p[10] += [0,  swing * 0.08, -swing * 0.20]
    p[12] += [0, -swing * 0.15,  swing * 0.12]
    p[13] += [0, -swing * 0.08,  swing * 0.20]
    p[3]  += [0,  swing * 0.10,  swing * 0.08]
    p[6]  += [0, -swing * 0.10, -swing * 0.08]
    return p

POSE_TEMPLATES = {
    "standing": _pose_standing,
    "sitting":  _pose_sitting,
}

# ─── CSI → Hareket Tahmini Hook ───────────────────────────────────────────────
# ESP32 gelince bu fonksiyon gerçek ML çıktısını alacak.
# Şimdilik sadece sinyal gücüne (amplitude_mean) bakarak basit karar veriyor.
def csi_to_activity(amplitude_vector: Optional[np.ndarray]) -> str:
    """
    Döndürdüğü değerler: 'idle' | 'walking' | 'sitting' | 'standing'
    amplitude_vector: CSI subcarrier genlik dizisi (None = donanım yok)
    """
    if amplitude_vector is None:
        return "idle"

    mean_amp = float(np.mean(amplitude_vector))
    variance  = float(np.var(amplitude_vector))

    if variance < 5:
        return "idle"
    elif mean_amp > 80:
        return "walking"
    elif mean_amp > 50:
        return "standing"
    else:
        return "sitting"


# ─── Skeleton Sınıfı ──────────────────────────────────────────────────────────
class Skeleton:
    """
    Tek bir insanı temsil eden 3D iskelet.
    CSI verisi yokken animasyon motoruyla çalışır,
    ESP32 gelince csi_to_activity() hook'u gerçek veriyi kullanır.
    """

    def __init__(self, grid_position: tuple[int, int] = (0, 0),
                 grid_dims: tuple[int, int] = (3, 3)):
        self.grid_pos   = grid_position      # (satır, sütun)
        self.grid_dims  = grid_dims          # (toplam_satır, toplam_sütun)
        self.world_pos  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_pos = self.world_pos.copy()

        self.activity    = "standing"
        self.anim_phase  = 0.0               # yürüyüş animasyonu fazı
        self.keypoints   = _pose_standing()  # (17, 3) float32

        # Geçiş için lerp tamponu
        self._lerp_from  = self.keypoints.copy()
        self._lerp_to    = self.keypoints.copy()
        self._lerp_t     = 1.0               # 1.0 = geçiş bitti

        # Grid hücre merkezlerini dünya koordinatlarına çevir
        self._update_world_pos()

    # ── Güncelleme ─────────────────────────────────────────────────────────────
    def update(self, dt: float, csi_amplitude: Optional[np.ndarray] = None):
        """Her frame çağrılır. dt: saniye cinsinden geçen süre."""
        new_activity = csi_to_activity(csi_amplitude)

        if new_activity != self.activity:
            self._start_transition(new_activity)
            self.activity = new_activity

        # Lerp geçişini ilerlet
        if self._lerp_t < 1.0:
            self._lerp_t = min(1.0, self._lerp_t + dt * 3.0)  # 0.33s geçiş
            self.keypoints = self._lerp_from + (self._lerp_to - self._lerp_from) * self._lerp_t

        # Yürüyüş animasyonu
        if self.activity == "walking":
            self.anim_phase += dt * 4.0
            self.keypoints = _pose_walking(self.anim_phase)

        # Dünya konumu yumuşak geçiş
        self.world_pos += (self.target_pos - self.world_pos) * min(1.0, dt * 5.0)

    def set_grid_position(self, row: int, col: int,
                          grid_dims: Optional[tuple] = None):
        """Kullanıcı veya ML tahmini yeni grid hücresi atadığında çağrılır."""
        self.grid_pos = (row, col)
        if grid_dims is not None:
            self.grid_dims = grid_dims
        self._update_world_pos()

    # ── İç Yardımcılar ─────────────────────────────────────────────────────────
    def _update_world_pos(self):
        row, col = self.grid_pos
        total_rows, total_cols = self.grid_dims
        # Grid'i [-1.5, +1.5] dünya aralığına normalize et (dinamik boyut)
        step_x = 3.0 / max(total_cols, 1)
        step_z = 3.0 / max(total_rows, 1)
        cx = (col + 0.5) * step_x - 1.5
        cz = (row + 0.5) * step_z - 1.5
        self.target_pos = np.array([cx, 0.0, cz], dtype=np.float32)

    def _start_transition(self, new_activity: str):
        self._lerp_from = self.keypoints.copy()
        if new_activity == "standing":
            self._lerp_to = _pose_standing()
        elif new_activity == "sitting":
            self._lerp_to = _pose_sitting()
        elif new_activity in ("walking", "idle"):
            self._lerp_to = _pose_standing()
        self._lerp_t = 0.0

    def get_world_keypoints(self) -> np.ndarray:
        """Keypoint'leri dünya koordinatlarına taşıyarak döndürür."""
        return self.keypoints + self.world_pos
