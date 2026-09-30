"""Birlesik 3xESP32 RX modulu - fiziksel yerlesim ve baglanti semasi (PDF cikti)."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

OUT = "/Users/oktay/Desktop/WiFi_Radar/docs/birlesik_modul_baglanti_semasi.pdf"

NEON = "#39ff14"
CYAN = "#00d8ff"
ORANGE = "#ff9100"
DIM = "#888888"
BG = "#0d1117"
FG = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": DIM,
})


def board(ax, cx, cy, w, h, angle_deg, label, color):
    """ESP32 kartini merkez (cx,cy) etrafinda angle_deg kadar dondurerek ciz."""
    rect = mpatches.FancyBboxPatch(
        (-w / 2, -h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=color, facecolor="#161b22"
    )
    t = matplotlib.transforms.Affine2D().rotate_deg(angle_deg).translate(cx, cy) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    lx = cx + math.cos(math.radians(angle_deg)) * 0
    ly = cy + math.sin(math.radians(angle_deg)) * 0
    ax.text(cx, cy - 0.02, label, ha="center", va="center",
            fontsize=8.5, color=color, fontweight="bold", rotation=0)


def antenna(ax, cx, cy, length, angle_deg, color, label=None):
    """Antenni cubuk + ucunda yuvarlak olarak ciz, angle_deg = ekseninden sapma."""
    rad = math.radians(angle_deg)
    x2 = cx + length * math.sin(rad)
    y2 = cy + length * math.cos(rad)
    ax.plot([cx, x2], [cy, y2], color=color, linewidth=2.4, solid_capstyle="round")
    ax.add_patch(mpatches.Circle((x2, y2), 0.045, facecolor=color, edgecolor="none", zorder=5))
    if label:
        ax.text(x2 + 0.08 * math.sin(rad), y2 + 0.08 * math.cos(rad), label,
                ha="center", va="center", fontsize=7.5, color=color)
    return x2, y2


def angle_arc(ax, cx, cy, r, a1, a2, color, text):
    """Iki acı (derece, 0=yukari, saat yonu) arasinda yay ciz ve etiketle."""
    arc = mpatches.Arc((cx, cy), r * 2, r * 2,
                       theta1=90 - a2, theta2=90 - a1,
                       edgecolor=color, linewidth=1.3, linestyle="--")
    ax.add_patch(arc)
    mid = math.radians((a1 + a2) / 2)
    tx = cx + (r + 0.16) * math.sin(mid)
    ty = cy + (r + 0.16) * math.cos(mid)
    ax.text(tx, ty, text, ha="center", va="center", fontsize=8, color=color)


# ----------------------------------------------------------------------------
# SAYFA 1 — Fiziksel yerlesim (kus bakisi): 3 ESP32 + anten fan dizilimi
# ----------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(11.7, 8.3))
ax1.set_xlim(-3.4, 3.4)
ax1.set_ylim(-2.05, 4.6)
ax1.set_aspect("equal")
ax1.axis("off")

ax1.text(0, 4.25, "BİRLEŞİK RX MODÜLÜ — FİZİKSEL YERLEŞİM (KUŞ BAKIŞI)",
         ha="center", fontsize=14, fontweight="bold", color=NEON)
ax1.text(0, 3.92, "3× ESP32-WROOM-32U  +  3× 3 dBi anten (fan dizilimi, ortak taban üzerinde)",
         ha="center", fontsize=9.5, color=DIM)

# Ortak taban (pertinaks/karton plaka)
base = mpatches.FancyBboxPatch((-2.35, -0.42), 4.7, 1.15,
                               boxstyle="round,pad=0.02,rounding_size=0.12",
                               linewidth=1.4, edgecolor=DIM, facecolor="#11161d", linestyle="-")
ax1.add_patch(base)
ax1.text(0, -0.62, "Ortak taban: 6×8 cm pertinaks plaka veya karton/akrilik altlık",
         ha="center", fontsize=8, color=DIM)
ax1.text(0, -1.45, "(3 kart üst üste değil, yan yana dizilir — ısınma ve anten girişimini önlemek için)",
         ha="center", fontsize=7.6, color=DIM)

ANT_LEN = 1.55
fan_angles = [-45, 0, 45]   # antenlerin merkez eksenden sapmasi
positions = [(-1.55, 0.0), (0.0, 0.0), (1.55, 0.0)]
labels = ["RX #1\nESP32-WROOM-32U", "RX #2 — MERKEZ\nESP32-WROOM-32U", "RX #3\nESP32-WROOM-32U"]
colors = [CYAN, NEON, ORANGE]

for (cx, cy), ang, lbl, col in zip(positions, fan_angles, labels, colors):
    board(ax1, cx, cy, 0.95, 0.34, 0, lbl, col)
    ax_, ay_ = antenna(ax1, cx, cy + 0.18, ANT_LEN, ang, col, label=f"3 dBi  ({ang:+d}°)")

# Acık alan / kapsama yonu oku
ax1.annotate("", xy=(0, 4.0), xytext=(0, 1.9),
             arrowprops=dict(arrowstyle="-|>", color=DIM, lw=1.2, alpha=0.55))
ax1.text(0.18, 3.55, "ODA / SALON YÖNÜ\n(yansımaların geleceği bölge)",
         fontsize=8, color=DIM, ha="left")

# Acı yaylari (RX#2 merkezinden referans alinarak)
angle_arc(ax1, 0.0, 0.18, 0.95, -45, 0, CYAN, "≈45°")
angle_arc(ax1, 0.0, 0.18, 1.25, 0, 45, ORANGE, "≈45°")
ax1.text(0, 2.7, "Toplam tarama açısı ≈ 90°\n(üç anten birlikte odanın geniş bir kesitini kapsar)",
         ha="center", fontsize=8.2, color=FG)

# Mevcut malzemeler kutusu
mat_text = (
    "MEVCUT MALZEMELER\n"
    "──────────────────────\n"
    "• 4× ESP32-WROOM-32U  (3'ü RX, 1'i sonraki TX denemesi için yedek)\n"
    "• 1× 12 dBi anten      (şimdilik kullanılmıyor — tek-kart sürümünde TX içindi)\n"
    "• 3× 3 dBi anten       (RX modülü — fan dizilimi)\n"
    "• IPEX→SMA pigtail kablo ×3  (anten ↔ kart bağlantısı için ZORUNLU)\n"
    "• 4 portlu USB hub (harici adaptörlü/powered)\n"
    "• Ev Wi-Fi router'ı     (illuminator — sinyal kaynağı olarak kullanılacak)"
)
ax1.text(-3.25, -1.62, mat_text, fontsize=7.6, color=FG, family="monospace",
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=DIM))

ax1.text(3.25, -1.62,
         "SABİTLEME ÖNERİSİ\n"
         "──────────────────\n"
         "Şu an sabit aparat yok.\n"
         "Geçici çözüm:\n"
         "  1) Üç kartı pertinaks/karton\n"
         "     plakaya çift taraflı bantla\n"
         "     45°'lik şablon üzerine yapıştır\n"
         "  2) Plakayı oda köşesine, duvara\n"
         "     yaslayarak dik konumda yerleştir\n"
         "  3) Antenler yukarı/odaya bakacak\n"
         "     şekilde pigtail kabloyu yönlendir\n"
         "Kalıcı çözüm: 3D baskı/lazer kesim\n"
         "  fan-mount (ileride)",
         fontsize=7.4, color=FG, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=DIM))

# ----------------------------------------------------------------------------
# SAYFA 2 — Bağlantı / kablolama şeması (blok diyagram)
# ----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11.7, 8.3))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 9)
ax2.axis("off")

ax2.text(6, 8.6, "BİRLEŞİK RX MODÜLÜ — BAĞLANTI ŞEMASI (BLOK DİYAGRAM)",
         ha="center", fontsize=14, fontweight="bold", color=NEON)
ax2.text(6, 8.2, "Ev Wi-Fi ile pasif algılama  →  3 paralel CSI akışı  →  USB Hub  →  PC (tek kablo)",
         ha="center", fontsize=9.5, color=DIM)


def box(ax, x, y, w, h, text, color, fontsize=9, fc="#161b22"):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=color, facecolor=fc)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold")
    return x, y, w, h


def arrow(ax, x1, y1, x2, y2, color=FG, label=None, lstyle="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, linestyle=lstyle))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center",
                fontsize=7.6, color=color)


# Ev router (illuminator)
box(ax2, 4.6, 7.1, 2.8, 0.75, "EV Wi-Fi ROUTER\n(sinyal kaynağı / illuminator)", DIM, fontsize=8.5)

# 3 RX karti + anten
rx_x = [0.6, 4.6, 8.6]
rx_col = [CYAN, NEON, ORANGE]
rx_lbl = ["RX #1\nESP32-WROOM-32U\n+ 3 dBi anten (-45°)",
          "RX #2 (MERKEZ)\nESP32-WROOM-32U\n+ 3 dBi anten (0°)",
          "RX #3\nESP32-WROOM-32U\n+ 3 dBi anten (+45°)"]
for x, col, lbl in zip(rx_x, rx_col, rx_lbl):
    box(ax2, x, 5.15, 2.8, 1.15, lbl, col, fontsize=8)
    arrow(ax2, x + 1.4, 7.1, x + 1.4, 6.32, color=DIM, lstyle="--")
    if x == 4.6:
        ax2.text(x + 1.4, 6.55, "Wi-Fi (CSI yansımaları)", fontsize=6.8, color=DIM, ha="center")

# USB hub
box(ax2, 4.0, 3.15, 4.0, 1.0, "4 PORTLU POWERED USB HUB\n(harici adaptörlü)", FG, fontsize=9.5)
for x, col in zip(rx_x, rx_col):
    arrow(ax2, x + 1.4, 5.15, 6.0, 4.18, color=col, label="USB-C  (data + güç)")

# Güç adaptörü → hub
box(ax2, 0.6, 3.25, 2.2, 0.8, "Şebeke adaptörü\n(5V / 2-3A)", DIM, fontsize=7.8, fc="#11161d")
arrow(ax2, 2.8, 3.65, 4.0, 3.65, color=DIM, label="DC güç")

# Hub → PC
box(ax2, 4.6, 1.1, 2.8, 1.0, "PC / NOTEBOOK\n(radar_ui.py + multi-reader)", FG, fontsize=9)
arrow(ax2, 6.0, 3.15, 6.0, 2.1, color=NEON, label="Tek USB kablo\n→ 3 ayrı COM portu")

# Sag alt: notlar
notes = (
    "NOTLAR\n"
    "──────────────────────────────────────────\n"
    "• Bu aşamada özel TX (12 dBi) modülü kullanılmıyor — ev router'ı\n"
    "  illuminator (sinyal kaynağı) görevi görüyor → \"pasif algılama\"\n"
    "• Her ESP32, USB üzerinden hem veri gönderiyor hem güç alıyor\n"
    "  (TP4056/MT3608 bu kurulumda gerekli değil — sadece taşınabilir\n"
    "  pil modu için ileride devreye girecek)\n"
    "• PC tarafında 3 ayrı seri port (COM/tty) açılır, timestamp bazlı\n"
    "  füzyonla tek 'super-vector' (3×23 = 69 öznitelik) oluşturulur\n"
    "• IPEX→SMA pigtail kablo OLMADAN antenler karta takılamaz — sipariş\n"
    "  listesinde mutlaka bulunmalı"
)
ax2.text(0.6, 0.85, notes, fontsize=7.7, color=FG, family="monospace",
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=DIM))

with PdfPages(OUT) as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)

print("PDF yazildi:", OUT)
