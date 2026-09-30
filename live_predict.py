"""
Canlı, olasılıksal konum tahmini — terminal çıktısı.
3 ESP32'den gelen CSI'yi okuyup core/wifi_radar_model_multi.pkl ile
her an için zone_A / zone_B / empty olasılıklarını terminale basar.

Kullanım:
    python live_predict.py
    python live_predict.py --no-boost      # CSI hız artırmayı kapat
    python live_predict.py --hz 4          # ekrana basma sıklığı (sn'de kaç tahmin)
"""

import argparse
import os
import sys
import time
import numpy as np
import joblib
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from core.esp32_serial_reader import ESP32SerialReader
from core.esp32_multi_reader import (
    ESP32MultiReader, CHANNEL_LABELS, N_SUBCARRIERS, order_ports_by_position,
)
from core.traffic_booster import TrafficBooster, discover_esp32_ips
from core.csi_features import build_multi_feature_vector
from core.live_indicator import LiveIndicator

MODEL_FILE = os.path.join(os.path.dirname(__file__), "core", "wifi_radar_model_multi.pkl")

BAR_WIDTH = 28
SMOOTH_N  = 5   # son N tahminin ortalaması alınır (titreşimi azaltmak için)


def prob_bar(p: float) -> str:
    filled = int(round(p * BAR_WIDTH))
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-boost", action="store_true",
                        help="CSI hızını artıran arka plan UDP trafiğini KAPAT")
    parser.add_argument("--hz", type=float, default=3.0, help="Saniyede kaç tahmin yazdırılsın")
    args = parser.parse_args()

    if not os.path.exists(MODEL_FILE):
        print(f"HATA: Model bulunamadı: {MODEL_FILE}\nÖnce train_model_multi.py çalıştır.", flush=True)
        sys.exit(1)

    bundle = joblib.load(MODEL_FILE)
    model        = bundle["model"]
    labels       = bundle["labels"]
    channel_dead = bundle["channel_dead"]
    print(f"Model yüklendi: {bundle['name']}  |  Sınıflar: {labels}", flush=True)

    # Portları bul ve [SOL, ORTA, SAĞ] sırasına diz
    found = ESP32SerialReader.find_esp32_ports()
    if len(found) != 3:
        print(f"HATA: 3 ESP32 portu bekleniyor, {len(found)} bulundu: {found}", flush=True)
        sys.exit(1)
    ports = order_ports_by_position(found)
    if ports is None:
        print(f"HATA: Bulunan portlar bilinen fiziksel eşlemeyle uyuşmuyor: {found}", flush=True)
        sys.exit(1)

    print("\nKanal <-> Port eşlemesi:", flush=True)
    for lbl, p in zip(CHANNEL_LABELS, ports):
        print(f"  {lbl:18s} -> {p}", flush=True)

    # CSI hızını artırmak için arka plan trafiği
    booster = None
    if not args.no_boost:
        print("\nESP32'ler resetlenip IP'leri okunuyor (CSI hız artırma için)...", flush=True)
        boost_ips = discover_esp32_ips(ports)
        if boost_ips:
            print("Trafik artırma hedefleri:", ", ".join(boost_ips), flush=True)
            booster = TrafficBooster(boost_ips, packets_per_sec=100.0)
        else:
            print("UYARI: IP'ler bulunamadı — trafik artırma kapalı, hız düşük olabilir.", flush=True)

    recent_probs = deque(maxlen=SMOOTH_N)
    last_print = [0.0]
    print_interval = 1.0 / args.hz
    indicator = LiveIndicator()

    def on_combined(arr: np.ndarray):
        now = time.time()
        if now - last_print[0] < print_interval:
            return
        last_print[0] = now

        rows = [arr[ch * N_SUBCARRIERS:(ch + 1) * N_SUBCARRIERS] for ch in range(3)]
        x = build_multi_feature_vector(rows, channel_dead).reshape(1, -1)

        proba = model.predict_proba(x)[0]
        recent_probs.append(proba)
        smoothed = np.mean(recent_probs, axis=0)

        best_idx = int(np.argmax(smoothed))
        best_label = model.classes_[best_idx]
        raw_idx = int(np.argmax(proba))
        raw_label = model.classes_[raw_idx]

        out = []
        out.append("\033[2J\033[H")  # ekranı temizle, imleci başa al (TERM bağımsız)
        out.append("=" * 64)
        out.append("  CANLI KONUM TAHMİNİ — 3 ANTEN BİRLEŞİK MODEL (olasılıksal)")
        out.append("=" * 64)
        out.append(f"  HAM (anlık tek örnek)  ->  {raw_label}")
        for i, cls in enumerate(model.classes_):
            mark = " <-- ham" if cls == raw_label else ""
            out.append(f"    {cls:10s} {prob_bar(proba[i])}  {proba[i]*100:5.1f}%{mark}")
        out.append("-" * 64)
        out.append(f"  YUMUŞATILMIŞ (son {len(recent_probs)} örnek ort.)  ->  {best_label}")
        for i, cls in enumerate(model.classes_):
            mark = " <-- TAHMİN" if cls == best_label else ""
            out.append(f"    {cls:10s} {prob_bar(smoothed[i])}  {smoothed[i]*100:5.1f}%{mark}")
        out.append("-" * 64)
        out.append(f"  Mevcut konum tahmini:  >>> {best_label} <<<   (Ctrl+C ile çık)")
        out.append("=" * 64)
        print("\n".join(out), flush=True)

        indicator.show(best_label, list(model.classes_), smoothed)

    reader = ESP32MultiReader(ports, on_combined)
    if not reader.start():
        print("\nHATA: Bir veya daha fazla port açılamadı:", flush=True)
        for st in reader.channel_status():
            print(f"  {st}", flush=True)
        sys.exit(1)

    if booster:
        booster.start()

    print("\nCanlı tahmin başlıyor... (Ctrl+C ile durdur)\n", flush=True)
    time.sleep(1.5)

    try:
        while True:
            indicator.pump()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        if booster:
            booster.stop()
        indicator.close()
        print("\n\nDurduruldu.", flush=True)


if __name__ == "__main__":
    main()
