"""
3 antenli (birleşik RX modülü) CSI veri toplama scripti.
Her satır 3 antenden gelen en güncel CSI örneklerinin birleşimidir (3x64 = 192 öznitelik).

Kullanım:
    python collect_data_multi.py --label zone_A --seconds 90
    python collect_data_multi.py --label zone_B --seconds 90
    python collect_data_multi.py --label empty  --seconds 90
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from core.esp32_serial_reader import ESP32SerialReader
from core.esp32_multi_reader import (
    ESP32MultiReader, CHANNEL_LABELS, N_SUBCARRIERS, COMBINED_LEN,
    order_ports_by_position,
)
from core.traffic_booster import TrafficBooster, discover_esp32_ips
from core.recording_indicator import RecordingIndicator

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "csi_fingerprints_multi.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",   required=True, help="Örn: empty, zone_A, zone_B, zone_C")
    parser.add_argument("--seconds", type=int, default=90, help="Kaç saniye kayıt yapılacak")
    parser.add_argument("--ports",   default=None,
                        help="Virgülle ayrılmış 3 port: SOL,ORTA,SAĞ (vermezsen otomatik bulunur/sıralanır)")
    parser.add_argument("--no-boost", action="store_true",
                        help="CSI hızını artıran arka plan UDP trafiğini KAPAT (varsayılan: açık)")
    parser.add_argument("--boost-ips", default=None,
                        help="Virgülle ayrılmış 3 IP: SOL,ORTA,SAĞ (vermezsen ESP32'ler resetlenip otomatik bulunur)")
    parser.add_argument("--boost-rate", type=float, default=100.0,
                        help="Saniyede gönderilecek UDP paket sayısı (varsayılan: 100)")
    args = parser.parse_args()

    # Portları bul ve [SOL, ORTA, SAĞ] sırasına diz
    if args.ports:
        ports = [p.strip() for p in args.ports.split(",")]
        if len(ports) != 3:
            print("HATA: --ports tam olarak 3 port almalı (SOL,ORTA,SAĞ)")
            sys.exit(1)
    else:
        found = ESP32SerialReader.find_esp32_ports()
        if len(found) != 3:
            print(f"HATA: 3 ESP32 portu bekleniyor, {len(found)} bulundu: {found}")
            print("       --ports SOL,ORTA,SAG ile manuel belirtebilirsin.")
            sys.exit(1)
        ports = order_ports_by_position(found)
        if ports is None:
            print(f"HATA: Bulunan portlar bilinen fiziksel eşlemeyle uyuşmuyor: {found}")
            print("       --ports SOL,ORTA,SAG ile manuel belirtebilirsin.")
            sys.exit(1)

    print("Kanal <-> Port eşlemesi:")
    for lbl, p in zip(CHANNEL_LABELS, ports):
        print(f"  {lbl:18s} -> {p}")

    # CSI hızını artırmak için arka plan UDP trafiği (varsayılan: açık)
    booster: Optional[TrafficBooster] = None
    if not args.no_boost:
        if args.boost_ips:
            boost_ips = [ip.strip() for ip in args.boost_ips.split(",")]
            if len(boost_ips) != 3:
                print("HATA: --boost-ips tam olarak 3 IP almalı (SOL,ORTA,SAĞ)")
                sys.exit(1)
        else:
            print("\nCSI hızını artırmak için ESP32'ler resetlenip IP'leri okunuyor...")
            boost_ips = discover_esp32_ips(ports)
            if boost_ips is None:
                print("UYARI: ESP32 IP'leri otomatik bulunamadı — trafik artırma KAPALI.")
                print("       --boost-ips IP1,IP2,IP3 ile manuel verebilir veya --no-boost ile susturabilirsin.")
                boost_ips = None

        if boost_ips:
            print("Trafik artırma hedefleri:")
            for lbl, ip in zip(CHANNEL_LABELS, boost_ips):
                print(f"  {lbl:18s} -> {ip}")
            booster = TrafficBooster(boost_ips, packets_per_sec=args.boost_rate)

    # CSV başlık satırı: amp_<kanal>_<subcarrier>
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    write_header = not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0
    csv_file = open(DATA_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        header = []
        for ch in range(3):
            header += [f"amp_ch{ch}_{i}" for i in range(N_SUBCARRIERS)]
        header.append("label")
        writer.writerow(header)
        csv_file.flush()

    collected = [0]
    start_time = [None]
    done = [False]

    def on_combined(arr: np.ndarray):
        if start_time[0] is None or done[0]:
            return
        elapsed = time.time() - start_time[0]
        if elapsed >= args.seconds:
            done[0] = True
            return
        row = arr.tolist()
        row.append(args.label)
        writer.writerow(row)
        collected[0] += 1
        remaining = args.seconds - elapsed
        print(f"\r  [{args.label}]  {collected[0]} satır (3-anten birleşik)  |  {remaining:.0f}s kaldı   ",
              end="", flush=True)

    reader = ESP32MultiReader(ports, on_combined)
    if not reader.start():
        print("\nHATA: Bir veya daha fazla port açılamadı. Kanal durumu:")
        for st in reader.channel_status():
            print(f"  {st}")
        sys.exit(1)

    if booster:
        print(f"\nCSI hızını artırmak için arka plan UDP trafiği başlatılıyor (~{args.boost_rate:.0f} paket/s/anten)...")
        booster.start()
        time.sleep(1.5)  # hızın oturması için kısa bekleme
        for st in reader.channel_status():
            print(f"  {st['label']:18s} anlık hız: {st['rate_hz']:.1f} paket/s")

    print(f"\nKayıt başlıyor: label='{args.label}'  süre={args.seconds}s  birleşik vektör uzunluğu={COMBINED_LEN}")
    print(f"{'='*50}")
    print(f"  Şimdi '{args.label}' için hazırlan.")
    print(f"  Kayıt otomatik duracak.")
    print(f"{'='*50}\n")

    indicator = RecordingIndicator()

    for i in range(5, 0, -1):
        print(f"\r  Kayıt {i} saniye sonra başlıyor...  ", end="", flush=True)
        indicator.show_waiting(i)
        for _ in range(10):
            indicator.pump()
            time.sleep(0.1)
    print(f"\r  KAYIT BASLADI — {args.label}                 \n")

    start_time[0] = time.time()

    while not done[0]:
        indicator.pump()
        indicator.show_recording(args.label, args.seconds - (time.time() - start_time[0]))
        time.sleep(0.2)
        if (time.time() - start_time[0]) > args.seconds + 2:
            break

    # Birleştirme thread'inin son turunu bitirip on_combined'dan çıkması için kısa bekleme
    time.sleep(0.3)

    reader.stop()
    if booster:
        booster.stop()
    csv_file.flush()
    csv_file.close()

    indicator.show_done()
    for _ in range(30):
        indicator.pump()
        time.sleep(0.1)
    indicator.close()

    print(f"\n\nTamamlandı! {collected[0]} satır kaydedildi → {DATA_FILE}")
    print(f"Toplam CSV satırı: {_count_rows()}")


def _count_rows():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE) as f:
        return sum(1 for _ in f) - 1


if __name__ == "__main__":
    main()
