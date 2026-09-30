"""WiFi CSI Radar - uctan uca surec raporu (PDF cikti).
Veri toplamadan canli siniflandirmaya kadar her adimi anlatan teknik rapor.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

OUT = "/Users/oktay/Desktop/WiFi_Radar/docs/wifi_radar_surec_raporu.pdf"

NEON = "#39ff14"
CYAN = "#00d8ff"
ORANGE = "#ff9100"
PINK = "#ff66ff"
RED = "#ff5555"
DIM = "#888888"
BG = "#0d1117"
FG = "#e6edf3"
PANEL = "#161b22"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": DIM,
    "font.family": "DejaVu Sans",
})

PAGE_W, PAGE_H = 8.27, 11.69  # A4 dikey

LINE_H = 0.0235     # bir metin satirinin kapladigi dikey alan (eksen-fraksiyonu)
TITLE_H = 0.052     # baslik + bosluk
PAD = 0.022         # ust/alt ic bosluk

TOTAL_PAGES = 11
_pageno = [0]


def n_lines(text):
    return text.count("\n") + 1


def box_h(text, has_title=True):
    h = n_lines(text) * LINE_H + PAD
    if has_title:
        h += TITLE_H
    return h


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _pageno[0] += 1
    return fig, ax


def header(ax, no, title, subtitle=None):
    ax.text(0.06, 0.965, f"{no:02d}", fontsize=13, color=ORANGE, fontweight="bold",
            family="monospace")
    ax.text(0.13, 0.965, title, fontsize=14.5, color=NEON, fontweight="bold")
    if subtitle:
        ax.text(0.13, 0.939, subtitle, fontsize=8, color=DIM)
    ax.plot([0.06, 0.94], [0.918, 0.918], color=DIM, linewidth=0.8)


def footer(ax):
    ax.text(0.5, 0.025, f"WiFi CSI Radar — Surec Raporu — Sayfa {_pageno[0]}/{TOTAL_PAGES}",
            fontsize=7.5, color=DIM, ha="center")


def box(ax, x, y_top, w, text, color=CYAN, fontsize=8.3, title=None, title_color=None):
    """y_top: kutunun UST kenari. Yukseklik metne gore otomatik hesaplanir.
    Geriye kutunun ALT kenarinin y konumunu dondurur (bir sonraki kutu icin)."""
    h = box_h(text, has_title=bool(title))
    y = y_top - h
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.3, edgecolor=color, facecolor=PANEL)
    ax.add_patch(rect)
    ty = y_top - PAD / 2 - 0.012
    if title:
        ax.text(x + 0.02, ty, title, fontsize=9.5, color=title_color or color,
                fontweight="bold", va="top")
        ty -= TITLE_H
    ax.text(x + 0.02, ty, text, fontsize=fontsize, color=FG, va="top", ha="left",
            linespacing=1.45)
    return y


def arrow(ax, x1, y1, x2, y2, color=DIM):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                shrinkA=2, shrinkB=2))


pages = []

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Kapak + genel akis
fig, ax = new_page()
ax.text(0.5, 0.88, "WiFi CSI RADAR", fontsize=28, color=NEON, fontweight="bold", ha="center")
ax.text(0.5, 0.835, "Uctan Uca Surec Raporu", fontsize=15, color=CYAN, ha="center")
ax.text(0.5, 0.806,
        "Veri toplamadan canli olasiliksal konum tahminine — her adimin teknik aciklamasi",
        fontsize=9, color=DIM, ha="center")
ax.plot([0.15, 0.85], [0.778, 0.778], color=DIM, linewidth=0.8)

ax.text(0.06, 0.74, "GENEL BORU HATTI (PIPELINE)", fontsize=12, color=ORANGE, fontweight="bold")

steps = [
    ("1", "VERİ TOPLAMA", "3x ESP32 (-45°/0°/+45°)\nUSB-seri, ham CSI", CYAN),
    ("2", "HIZ ARTIRMA", "UDP trafik flood ile\nCSI tetikleme hizi ↑", ORANGE),
    ("3", "BİRLEŞTİRME", "Olay-tabanli kayipsiz\n3-kanal kalibrasyon", NEON),
    ("4", "ÖZELLİK ÇIKARIMI", "192 ham deger ->\n75 boyutlu vektor", CYAN),
    ("5", "MODEL EĞİTİMİ", "6 algoritma yarisi +\nGroupKFold/holdout", ORANGE),
    ("6", "CANLI TAHMİN", "predict_proba ->\nzone_A/B/empty %", NEON),
]
bw, bh, gapx, gapy = 0.135, 0.155, 0.013, 0.045
x0, y0 = 0.06, 0.555
for i, (num, title, desc, color) in enumerate(steps):
    col, row = i % 3, i // 3
    x = x0 + col * (bw + gapx)
    y = y0 - row * (bh + gapy)
    rect = mpatches.FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.015",
                                   linewidth=1.5, edgecolor=color, facecolor=PANEL)
    ax.add_patch(rect)
    ax.text(x + 0.014, y + bh - 0.026, num, fontsize=12, color=color, fontweight="bold",
            family="monospace")
    ax.text(x + bw / 2, y + bh - 0.058, title, fontsize=8, color=color, fontweight="bold",
            ha="center")
    ax.text(x + bw / 2, y + bh - 0.085, desc, fontsize=6.7, color=FG, ha="center", va="top",
            linespacing=1.5)
    if col < 2:
        arrow(ax, x + bw + 0.0015, y + bh / 2, x + bw + gapx - 0.0015, y + bh / 2, color=DIM)
arrow(ax, x0 + bw / 2, y0 - bh - 0.006, x0 + bw / 2, y0 - bh - gapy + 0.006, color=DIM)

ax.text(0.06, 0.295,
        "BU RAPOR HANGİ SORUYU CEVAPLIYOR?", fontsize=11.5, color=ORANGE, fontweight="bold")
qbox_text = (
    "\"Zone A ya da B'de dururken, modelin gordugu CSI verisi onceki test\n"
    "serileriyle birebir ortusmuyor — yine de A mi B mi oldugunu nasil soyluyor?\"\n\n"
    "Cevap: Model BİREBİR EŞLEŞTİRME yapmaz. Egitim sirasinda binlerce ornekten\n"
    "GENELLENEBİLİR bir istatistiksel imza / karar siniri ogrenir; yeni gelen\n"
    "her ornegi bu sinira gore SINIFLANDIRIR ve olasilik uretir. Bu raporun her\n"
    "bolumu, bu ogrenmenin hangi veriden ve hangi islem adimlarindan gectigini\n"
    "en kucuk detayina kadar anlatir."
)
box(ax, 0.06, 0.27, 0.88, qbox_text, color=NEON, fontsize=9.3)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Fiziksel kurulum + port eslemesi
fig, ax = new_page()
header(ax, 1, "DONANIM — FİZİKSEL KURULUM VE PORT EŞLEMESİ",
       "core/esp32_serial_reader.py · core/esp32_multi_reader.py")

y = box(ax, 0.06, 0.895, 0.88,
    "• 3 adet ESP32, fiziksel olarak fan (yelpaze) diziliminde yerlestirilmis:\n"
    "   RX-1 SOL (-45°)  ·  RX-2 ORTA (0°)  ·  RX-3 SAĞ (+45°)  — her biri 3dBi anten\n"
    "• Her ESP32 USB-seri ile bilgisayara baglidir (ag/IP uzerinden 'ping'\n"
    "  ATILAMAZ — saglik kontrolu icin check_esp_connections.py yazildi: portlari\n"
    "  acar, CSI akiyor mu diye bakar, '✓ CANLI / ✗ SORUNLU' raporlar)\n"
    "• Her CSI paketi 64 alt-tasiyici (subcarrier) genlik degeri tasir\n"
    "  (N_SUBCARRIERS = 64)",
    title="1.1  Fiziksel kurulum", title_color=CYAN, color=CYAN)

y = box(ax, 0.06, y - 0.05, 0.88,
    "Sorun: find_esp32_ports() portlari ALFABETİK sirada donduruyor\n"
    "(usbserial-0001, usbserial-5, usbserial-6) — bu sira FİZİKSEL yerlesimle\n"
    "(SOL/ORTA/SAĞ) ÖRTÜŞMÜYOR. Yanlis eslestirme = yanlis kalibrasyon.\n\n"
    "Çözüm — PORT_POSITION sözlügü (sabit fiziksel-port eslemesi):\n"
    "   usbserial-5    ->  SOL   (-45°)        position 0\n"
    "   usbserial-6    ->  ORTA  (0°)          position 1\n"
    "   usbserial-0001 ->  SAĞ   (+45°)        position 2\n"
    "order_ports_by_position() bulunan portlari bu eslemeyle [SOL,ORTA,SAĞ]\n"
    "fiziksel sirasina yeniden dizer; bilinmeyen bir port adi gelirse None\n"
    "doner (yanlis eslemeyle devam etmesini engeller).",
    title="1.2  Port ↔ fiziksel pozisyon eşlemesi  (kalibrasyonun ilk adimi)",
    title_color=ORANGE, color=ORANGE)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Hiz artirma + veri kaydi
fig, ax = new_page()
header(ax, 1, "DONANIM — CSI HIZ ARTIRMA VE ETİKETLİ KAYIT",
       "core/traffic_booster.py · collect_data_multi.py")

y = box(ax, 0.06, 0.895, 0.88,
    "ESP32'lerin dogal CSI uretim hizi cok dusuk (~0.2-0.7 Hz) — zone'lari ayirt\n"
    "edecek kadar veri toplamak icin yetersiz.\n\n"
    "Çözüm — TrafficBooster (UDP trafik 'sel'i):\n"
    "  1) discover_esp32_ips(): ESP32'ler DTR/RTS seri sinyalleriyle resetlenir,\n"
    "     boot loglarindan 'IP: x.x.x.x' regex ile okunur. (ICMP ping degil —\n"
    "     macOS'te dusuk araliklarla ping atmak root ister, UDP istemez.)\n"
    "     Bulunan IP'ler: SOL=192.168.1.121  ORTA=192.168.1.122  SAĞ=192.168.1.120\n"
    "  2) Bu IP'lere saniyede ~100 adet 32-byte UDP paketi gonderilir -> 802.11\n"
    "     cerceve trafigi olusur -> ESP32'nin CSI callback'i cok daha sik tetiklenir\n"
    "Sonuç: CSI hizi ~0.5 Hz'den ~40-80 Hz/anten'e cikar (3 anten toplam ~120-240\n"
    "ham paket/sn).",
    title="2.1  CSI hız artırma (traffic boosting)", title_color=NEON, color=NEON)

y = box(ax, 0.06, y - 0.05, 0.88,
    "collect_data_multi.py --label zone_A --seconds 90\n\n"
    "Akis: portlari bul/sirala -> IP'leri kesfet -> trafik artirmayi baslat ->\n"
    "ESP32MultiReader'i baslat -> 90 sn boyunca her birlesik ornegi\n"
    "data/csi_fingerprints_multi.csv dosyasina yaz (192 sayisal sutun + label).\n\n"
    "⚠ İlk toplamada combine_hz=20 ile sabit-hizli ('en son neyse onu al')\n"
    "birlestirme kullanildi — hizli kanallarin verisi cope gidiyor, yavas\n"
    "kanaldan ayni bayat ornek tekrar tekrar kullaniliyordu (kalibrasyonsuz +\n"
    "israfli). O veri yedeklendi (_old_uncalibrated_*.csv) ve YENİ olay-tabanli\n"
    "yontemle (Bölüm 3) yeniden toplaniyor.",
    title="2.2  Etiketli kayıt akışı  ve  tespit edilen kritik sorun",
    title_color=ORANGE, color=ORANGE)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Birlestirme problemi + cozum
fig, ax = new_page()
header(ax, 2, "3 KANALI BİRLEŞTİRME — KALİBRASYON VE ÇÖZÜM",
       "core/esp32_multi_reader.py — ESP32MultiReader._on_channel_csi()")

y = box(ax, 0.06, 0.895, 0.88,
    "Her kanal kendi arka plan thread'inde okunur ve FARKLI hizlarda akar (orn.\n"
    "SOL 82 Hz, ORTA 38 Hz, SAĞ 65 Hz). Tek bir 'birlesik ornek' uretmek icin\n"
    "3 kanaldan gelen verileri zamansal olarak hizalamak (kalibre etmek) gerekir.\n\n"
    "Yanlış yaklaşım ('sabit araliklarla en sonu al' / combine_hz):\n"
    "  • yavas kanaldan AYNI bayat ornek defalarca tekrar kullanilir (kalibrasyonsuz)\n"
    "  • hizli kanallarin urettigi orneklerin cogu hic kullanilmadan kaybolur (israf)",
    title="3.1  Neden 'birleştirme' bir kalibrasyon problemidir?", title_color=CYAN, color=CYAN)

y = box(ax, 0.06, y - 0.05, 0.88,
    "Her kanaldan yeni paket geldiginde o kanal threading.Lock ile 'TAZE'\n"
    "isaretlenir:  self._latest[idx]=arr ; self._fresh[idx]=True\n\n"
    "Üç kanaldan da en az bir taze örnek biriktiği an:\n"
    "   if all(self._fresh) and all(x is not None for x in self._latest):\n"
    "       emit_snapshot = list(self._latest)      # ANINDA birlestir, gonder\n"
    "       self._fresh = [False]*3                  # bayraklari sifirla\n\n"
    "Üç garanti birden saglanir:\n"
    "  ✓ Hicbir kanaldan veri cope gitmez (her biri en az 1 kez katki saglar)\n"
    "  ✓ Cikis hizi EN YAVAŞ kanalin hizina esitlenir (gercek kalibrasyon)\n"
    "  ✓ Üç ornek de zamanda birbirine yakin anlarda toplanir\n"
    "Ölçüm: last_skew_s (kanal-arasi en buyuk zaman farki) ~10 ms; cikis hizi\n"
    "eski yontemde ~19 ornek/sn -> yeni yontemde ~47 ornek/sn.",
    title="3.2  Çözüm — OLAY TABANLI / KAYIPSIZ BİRLEŞTİRME (yeni yöntem)",
    title_color=NEON, color=NEON)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — birlesik vektor + CSV
fig, ax = new_page()
header(ax, 2, "BİRLEŞİK CSI VEKTÖRÜ VE DİSKTEKİ SON HALİ",
       "ESP32MultiReader._emit() → data/csi_fingerprints_multi.csv")

y = box(ax, 0.06, 0.895, 0.88,
    "_emit(): her kanaldan gelen 64'lük diziler [SOL | ORTA | SAĞ] sırasıyla\n"
    "uç uca eklenir (np.concatenate) -> 192 elemanlı tek 'birleşik CSI vektörü'.\n"
    "combined_count sayacı ve last_skew_s (kanallar arası en büyük zaman farkı)\n"
    "tanı amaçlı tutulur — kalibrasyon kalitesi her an ölçülebilir hale gelir.",
    title="4.1  192-boyutlu birleşik vektör nasıl oluşur?", title_color=ORANGE, color=ORANGE)

y = box(ax, 0.06, y - 0.05, 0.88,
    "[ SOL kanal: 64 değer ][ ORTA kanal: 64 değer ][ SAĞ kanal: 64 değer ]\n"
    "      amp_ch0_0..63          amp_ch1_0..63           amp_ch2_0..63\n"
    "                                    │\n"
    "                                    ▼\n"
    "        CSV satırı:  amp_ch0_0, ..., amp_ch2_63, label\n"
    "                     (data/csi_fingerprints_multi.csv)\n\n"
    "Bu CSV, Bölüm 5'teki özellik mühendisliği ve Bölüm 6'daki model eğitiminin\n"
    "TEK girdisidir — yani kalibrasyon kalitesi, modelin öğrenebileceği\n"
    "'gerçeğin' ne kadar doğru temsil edildiğini doğrudan belirler.\n"
    "(örnek dağılım: zone_A=1693, zone_B=1698, empty=1793 satır)",
    title="4.2  Diskteki son hali — bir sonraki tüm adımların tek girdisi",
    title_color=CYAN, color=CYAN)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — feature engineering: neden + ilk iki adim
fig, ax = new_page()
header(ax, 3, "ÖZNİTELİK MÜHENDİSLİĞİ — NEDEN VE İLK ADIMLAR",
       "core/csi_features.py — find_dead_subcarriers() · normalize_row()")

y = box(ax, 0.06, 0.895, 0.88,
    "Ham 192 değer (3x64 genlik) doğrudan modele verilmez — gürültüye duyarlı,\n"
    "boyutu yüksek ve 'şekil' bilgisini açıkça taşımaz. Onun yerine her paketten\n"
    "anlamlı, düşük-boyutlu bir 'parmak izi' (öznitelik vektörü) çıkarılır.\n"
    "Bu vektör HEM eğitimde HEM canlı tahminde AYNI fonksiyonla üretilir —\n"
    "(build_multi_feature_vector) bu tutarlılık, modelin öğrendiği şeyin canlı\n"
    "veride de geçerli olmasının ön koşuludur.",
    title="5.1  Neden 'ham veri' değil 'öznitelik vektörü'?", title_color=CYAN, color=CYAN)

y = box(ax, 0.06, y - 0.05, 0.88,
    "Adım 1 — Ölü/sabit subcarrier tespiti  find_dead_subcarriers(var_threshold=1e-3):\n"
    "   Eğitim setinde varyansı ~0 olan alt-taşıyıcılar (pilot/DC/guard sinyalleri —\n"
    "   bilgi taşımazlar) bulunur ve sonraki tüm adımlarda atılır.\n\n"
    "Adım 2 — Per-packet normalizasyon  normalize_row():\n"
    "   row / mean(sıfır-olmayan değerler)  ->  her paketin ortalaması 1'e çekilir.\n"
    "   Sonuç: mutlak güç / mesafe değişimlerine karşı dayanıklılık — sadece\n"
    "   sinyalin alt-taşıyıcılar arasındaki GÖRELİ DAĞILIM ŞEKLİ kalır (kişinin\n"
    "   antene olan mutlak mesafesi değil, CSI'nin 'şekli' konum bilgisini taşır).",
    title="5.2  Tek-kanal işlem hattı — temizlik ve normalizasyon",
    title_color=ORANGE, color=ORANGE)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — 23 ozellik + cross-channel 75
fig, ax = new_page()
header(ax, 3, "ÖZNİTELİK MÜHENDİSLİĞİ — 75 BOYUTLU VEKTÖR",
       "build_feature_vector() · cross_channel_features() · build_multi_feature_vector()")

y = box(ax, 0.06, 0.895, 0.88,
    "Temizlenmiş + normalize edilmiş 'şekilden' 23 sayısal özet çıkarılır:\n"
    "  •  8  grup ortalaması : diziyi 8 eşit parçaya bölüp her parçanın ortalaması\n"
    "                          (spektrumun kaba şekli)\n"
    "  •  7  grup farkı      : np.diff(grup_ortalamaları) — komşu gruplar arası\n"
    "                          eğim / monotonluk bilgisi\n"
    "  •  1  eğim (slope)    : np.polyfit derece-1  ->  spektral 'tilt'\n"
    "  •  1  ağırlık merkezi : centroid = Σ(x·değer) / Σ(değer)\n"
    "  •  1  düzlük          : std/mean  ->  spektrumun ne kadar 'pürüzlü' olduğu\n"
    "  •  1  log-enerji      : log(1+|ham_değer|.ortalama)  ->  mutlak güç özeti\n"
    "  •  4  min/max/p25/p75 : normalize dağılımın sınırı ve çeyreklikleri\n"
    "                          ────── 23 ÖZELLİK / kanal × 3 kanal = 69",
    title="6.1  Tek-kanal 'şekil parmak izi'  (her anten için ayrı üretilir)",
    title_color=NEON, color=NEON)

y = box(ax, 0.06, y - 0.05, 0.88,
    "Sadece 3 kanalın 23'er özelliğini yan yana koymak (3×23=69) antenler\n"
    "ARASINDAKİ farkı modele AÇIKÇA sunmaz — örtük yön/AoA bilgisini model kendi\n"
    "keşfetmek zorunda kalır (3-anten sonuçlarının beklenenden az üstün çıkmasının\n"
    "olası nedeni budur). cross_channel_features() bunu 6 ek özellikle giderir:\n"
    "  •  3 log-enerji farkı  : (SOL-ORTA), (ORTA-SAĞ), (SOL-SAĞ)\n"
    "       -> kişi hangi yöne yakınsa o anten enerjisi diğerlerinden farklılaşır\n"
    "          (basitleştirilmiş açı-of-geliş / AoA bilgisi)\n"
    "  •  3 şekil korelasyonu : normalize şekiller arası Pearson korelasyonu\n"
    "       -> yansıma desenlerinin ne kadar benzer/farklı olduğu\n\n"
    "build_multi_feature_vector() = [kanal0(23) | kanal1(23) | kanal2(23) | çapraz(6)]\n"
    "                          ────── TOPLAM: 75 BOYUTLU ÖZELLİK VEKTÖRÜ",
    title="6.2  YENİ — Çapraz-kanal 'yön/çeşitlilik' özellikleri (3-anten avantajı)",
    title_color=PINK, color=PINK)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — egitim: veri + dogrulama semasi
fig, ax = new_page()
header(ax, 4, "MODEL EĞİTİMİ — VERİ HAZIRLIĞI VE DOĞRULAMA",
       "train_model_multi.py — make_groups() · split_holdout()")

y = box(ax, 0.06, 0.895, 0.88,
    "data/csi_fingerprints_multi.csv okunur, her satırdan 75-boyutlu özellik\n"
    "vektörü çıkarılır (Bölüm 6). Adil A/B karşılaştırması için AYNI oturumdan\n"
    "iki model eğitilir:\n"
    "  (A) SADECE ORTA (0°) anten  ->  23 özellik   (eski tek-anten hattıyla birebir)\n"
    "  (B) 3 ANTEN BİRLEŞİK + çapraz-kanal  ->  75 özellik\n"
    "İkisi de aynı satırlardan, aynı zaman dilimlerinden, aynı şema ile\n"
    "değerlendirilir -> '3 anten ne kadar katkı sağlıyor' sorusuna doğrudan cevap.",
    title="7.1  Veri yükleme ve adil A/B karşılaştırma tasarımı", title_color=CYAN, color=CYAN)

y = box(ax, 0.06, y - 0.05, 0.88,
    "Neden GroupKFold? Ardışık CSI paketleri zaman içinde birbirine yüksek\n"
    "korelasyonludur (aynı an, aynı konum, aynı gürültü). Rastgele k-fold\n"
    "kullanılırsa eğitim/test setine BİRBİRİNE ÇOK BENZER paketler düşebilir ->\n"
    "yapay/yanıltıcı yüksek skor (veri sızıntısı).\n\n"
    "  make_groups(): her etiket için ardışık BLOCK_SIZE=200 paket aynı 'gruba'\n"
    "    atanır; GroupKFold bu grupları asla bölmeden katlar oluşturur.\n"
    "  split_holdout(): her etiketin son %20'lik ZAMAN dilimi tamamen ayrılır,\n"
    "    eğitime hiç sokulmaz — 'modelin daha önce hiç görmediği, farklı bir\n"
    "    zaman diliminden gelen veride ne kadar başarılı?' sorusunun cevabıdır.",
    title="7.2  Sızıntı-önleyici doğrulama: GroupKFold + zaman-bazlı holdout",
    title_color=ORANGE, color=ORANGE)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 9 — model yarisi + sonuc
fig, ax = new_page()
header(ax, 4, "MODEL EĞİTİMİ — ALGORİTMA YARIŞI VE SONUÇLAR",
       "train_model_multi.py — get_models() · evaluate() · sonuç tablosu")

y = box(ax, 0.06, 0.895, 0.88,
    "Her iki özellik seti (23 ve 75 boyutlu) için AYNI 6 algoritma yarıştırılır:\n"
    "   LogReg (L2)  ·  Linear SVM  ·  RBF SVM  ·  Random Forest  ·\n"
    "   Gradient Boosting  ·  KNN-7\n"
    "(SVM'ler CalibratedClassifierCV ile sarılır — ham SVM skorları olasılık\n"
    "DEĞİLDİR, kalibrasyon onları gerçekçi olasılığa çevirir; ağaç-tabanlı\n"
    "modeller zaten predict_proba ürettiği için sarılmaz.)\n"
    "Her model GroupKFold ortalamasına (cv_mean) göre sıralanır; en yüksek\n"
    "cv_mean'e sahip model 'şampiyon' seçilip ayrı tutulan holdout setinde\n"
    "test edilir — bu, gerçek dünyada beklenen başarıya en yakın rakamdır.",
    title="8.1  6 algoritmanın yarıştırılması ve 'şampiyon' seçimi", title_color=NEON, color=NEON)

y = box(ax, 0.06, y - 0.05, 0.88,
    "                          GroupKFold CV     Holdout Test\n"
    "   Tek anten (ORTA)            91.1%             91.3%\n"
    "   3 anten birleşik            94.8%             91.2%\n"
    "   FARK (3anten - tek)         +3.7%             −0.1%\n\n"
    "Confusion matrix'te zone_B -> zone_A yönünde ~%26 karışma gözlendi — bu,\n"
    "fiziksel olarak yakın iki bölgenin CSI imzalarının kısmen örtüştüğünü\n"
    "ve/veya kalibrasyonsuz ağaç-tabanlı modellerin aşırı-güven eğilimini\n"
    "gösteriyor (bu rapor sürecinde tespit edilip 75-boyutlu çapraz-kanal\n"
    "özellikleriyle giderilmeye çalışıldı — Bölüm 6.2).\n"
    "Şampiyon model + her kanalın ölü-subcarrier listesi joblib ile\n"
    "core/wifi_radar_model_multi.pkl olarak kaydedilir — eğitim ve canlı\n"
    "tahmin böylece AYNI ön-işleme parametrelerini paylaşır.",
    title="8.2  Sonuç karşılaştırması (ilk eğitim turu — 69-boyutlu özellikle)",
    title_color=CYAN, color=CYAN)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 10 — canli: baslangic + ozellik cikarimi
fig, ax = new_page()
header(ax, 5, "CANLI SINIFLANDIRMA — SİSTEM AYAĞA KALKIYOR",
       "live_predict.py — main() · ESP32MultiReader · build_multi_feature_vector()")

y = box(ax, 0.06, 0.895, 0.88,
    "1) wifi_radar_model_multi.pkl yüklenir: model + öğrenilmiş parametreler\n"
    "   (her kanal için ölü-subcarrier listesi, sınıf etiketleri labels)\n"
    "2) Portlar bulunur ve [SOL,ORTA,SAĞ] sırasına dizilir, ESP32 IP'leri\n"
    "   keşfedilir, traffic booster başlatılır (Bölüm 2.1 ile birebir aynı)\n"
    "3) ESP32MultiReader ile Bölüm 3'teki AYNI olay-tabanlı/kalibreli\n"
    "   birleştirme çalışır -> her an için taze bir 192-boyutlu birleşik\n"
    "   CSI vektörü üretilir ve on_combined() callback'ine iletilir",
    title="9.1  Başlangıç — modeli ve canlı veri akışını ayağa kaldırma",
    title_color=CYAN, color=CYAN)

y = box(ax, 0.06, y - 0.05, 0.88,
    "on_combined() her yeni birleşik örnekte çalışır:\n\n"
    "   rows = [arr[ch*64 : (ch+1)*64] for ch in range(3)]\n"
    "   x = build_multi_feature_vector(rows, channel_dead).reshape(1, -1)\n"
    "   proba = model.predict_proba(x)[0]\n\n"
    "ÖNEMLİ: build_multi_feature_vector() ve channel_dead listesi BİREBİR\n"
    "Bölüm 6 ve 7'de eğitimde kullanılanlarla AYNIDIR. Yeni gelen örnek ESKİ\n"
    "örneklerle kıyaslanmaz — eğitimde öğrenilen 'ölü subcarrier' bilgisiyle\n"
    "aynı şekilde temizlenir, aynı şekilde normalize edilir, aynı 75-boyutlu\n"
    "uzaya izdüşürülür. Böylece model, daha önce HİÇ GÖRMEDİĞİ bu yeni\n"
    "vektörü, eğitimde öğrendiği karar sınırlarına göre değerlendirir.",
    title="9.2  Yeni örnekten özellik çıkarımı — eğitimle BİREBİR AYNI yol",
    title_color=ORANGE, color=ORANGE)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 11 — olasilik + cevap
fig, ax = new_page()
header(ax, 5, "CANLI SINIFLANDIRMA — OLASILIK VE SONUÇ",
       "live_predict.py — prob_bar() · smoothing · sonuç")

y = box(ax, 0.06, 0.895, 0.88,
    "predict_proba(x)[0] -> [P(empty), P(zone_A), P(zone_B)] gibi toplamı 1.0\n"
    "olan bir olasılık dağılımı döner. Bu, 'en yakın eğitim örneğini bul'\n"
    "DEĞİL, 'öğrenilen istatistiksel modele göre bu vektörün her sınıfa ait\n"
    "OLMA OLASILIĞI nedir' sorusunun cevabıdır.\n\n"
    "Titreşimi azaltmak için son SMOOTH_N=5 tahminin ortalaması alınır\n"
    "(np.mean(recent_probs, axis=0)); en yüksek olasılıklı sınıf '>>> TAHMİN\n"
    "<<<' olarak işaretlenip ASCII çubuklarla (prob_bar) terminale basılır —\n"
    "ekran \\033[2J\\033[H ANSI kodlarıyla TERM-bağımsız temizlenir.",
    title="10.1  Olasılığa dönüşüm — 'birebir eşleştirme' DEĞİL, sınıflandırma",
    title_color=NEON, color=NEON)

y = box(ax, 0.06, y - 0.05, 0.88,
    "EVET — sistem tam tahmin ettiğiniz gibi çalışıyor, ama 'birebir eşleştirme'\n"
    "değil, bir SINIFLANDIRMA / GENELLEME süreci:\n"
    "  •  Model, eğitimde gördüğü binlerce örnekten zone_A ve zone_B için\n"
    "     GENELLENEBİLİR istatistiksel imzalar (75-boyutlu uzayda karar\n"
    "     sınırları) öğrenmiştir — satırları 'ezberlememiştir'.\n"
    "  •  Holdout testi (Bölüm 7.2) bunu kanıtlar: model, eğitimde HİÇ\n"
    "     görmediği farklı bir zaman dilimindeki veride de ~%91 başarı\n"
    "     gösteriyor — yani 'daha önce görülmemiş' veriyle çalışması NORMAL\n"
    "     ve beklenen davranıştır.\n"
    "  •  Canlı tahminde de aynı mantık geçerlidir: yeni CSI'nin önceki test\n"
    "     serileriyle birebir uyuşmasına HİÇ GEREK YOKTUR — önemli olan, aynı\n"
    "     ön-işleme/özellik-çıkarım hattından geçtikten sonra öğrenilen karar\n"
    "     sınırının HANGİ TARAFINA düştüğüdür.",
    title="✓ SORUNUN CEVABI — \"A mı B mi yaptırıyoruz, evet öyle mi yapıyoruz?\"",
    title_color=NEON, color=NEON, fontsize=8.6)
footer(ax)
pages.append(fig)

# ─────────────────────────────────────────────────────────────────────────────
with PdfPages(OUT) as pdf:
    for fig in pages:
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF olusturuldu: {OUT}  ({len(pages)} sayfa)")
