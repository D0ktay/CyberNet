---
name: project-overview
description: Wi-Fi CSI Radar projesinin genel yapısı, teknik altyapı ve mevcut özellikler
metadata:
  type: project
---

Wi-Fi CSI tabanlı insan konum/aktivite tespiti sistemi. ESP32 donanımı, Python/Pygame UI.

**Teknik Yığın:**
- `ui/radar_ui.py` — Ana UI (Radar3x3UI sınıfı), 1720×960px, 3 panel
- `ui/renderer_3d.py` — PyOpenGL + 2D Fallback iskelet renderer (macOS'ta OpenGL crash yüzünden hep 2D fallback çalışır)
- `core/skeleton.py` — 17 keypoint, LERP geçiş animasyonu, csi_to_activity()
- `core/radar_ml.py` — ML motoru

**Panel Düzeni (1720×960):**
- Sol (820px): Dinamik grid + heatmap dolgu + CSI dalga grafiği + CSI subcarrier heatmap + area config
- Orta (520px): 3D viewport (2D fallback), vital signs mini panel
- Sağ (380px): Telemetri, diagnostics, terminal log, link reset butonu

**Dinamik Grid Motoru (v3.0):**
- Algoritma: `cols = max(1, int(room_w // 2.0))`, `rows = max(1, int(room_l // 2.0))`
- Fiziksel kural: hücre ≥ 2.0m (donanım hata payı ±1m → 2m belirsizlik çapı)
- 9×3m → 4×1, 6×6m → 3×3, 3×4m → 1×2
- Warn: hücre < 2m ise "⚠ LOW SPATIAL RESOLUTION" yanıp söner

**Yeni Özellikler (v3.0):**
- Dinamik grid: Apply & Rescale ile anlık yeniden hesaplama
- Vital Signs mini panel: BPM + EKG çizgisi + solunum hızı (aktiviteye göre dinamik)
- CSI subcarrier heatmap (8×16 alt taşıyıcı renk matrisi)
- Hysteresis filtresi (5 frame eşiği)
- Diagnostics terminal log
- Yön tuşları + 1-9 klavye navigasyonu
- İskelet yürüme animasyonu (zone değişiminde 1.2 sn)
- skeleton.py grid_dims parametresi ile dinamik 3D konum normalizasyonu

**Why:** ESP32 gerçek donanımı gelene kadar simülasyon modu ile çalışır.
**How to apply:** Tüm grid değişikliklerinde `_rebuild_grid()` çağrılmalı; skeleton.set_grid_position'a grid_dims geçilmeli.
