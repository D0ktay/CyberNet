"""
CSI veri toplama scripti.
Kullanım:
    python collect_data.py --label zone_A --seconds 90
    python collect_data.py --label empty  --seconds 90
"""

import argparse
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.esp32_serial_reader import ESP32SerialReader

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "csi_fingerprints.csv")
N_SUBCARRIERS = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",   required=True, help="Örn: empty, zone_A, zone_B, zone_C")
    parser.add_argument("--seconds", type=int, default=90, help="Kaç saniye kayıt yapılacak")
    parser.add_argument("--port",    default=None, help="Serial port (otomatik bulunur)")
    args = parser.parse_args()

    # Port bul
    port = args.port
    if port is None:
        ports = ESP32SerialReader.find_esp32_ports()
        if not ports:
            print("HATA: ESP32 portu bulunamadı. --port ile manuel gir.")
            sys.exit(1)
        port = ports[0]
        print(f"Port bulundu: {port}")

    # CSV başlık satırı
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    write_header = not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0
    csv_file = open(DATA_FILE, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        header = [f"amp_{i}" for i in range(N_SUBCARRIERS)] + ["label"]
        writer.writerow(header)
        csv_file.flush()

    # Sayaçlar
    collected = [0]
    start_time = [None]

    def on_csi(arr: np.ndarray):
        if start_time[0] is None:
            return
        # 64'e pad veya kırp
        if len(arr) >= N_SUBCARRIERS:
            row = arr[:N_SUBCARRIERS].tolist()
        else:
            row = arr.tolist() + [0.0] * (N_SUBCARRIERS - len(arr))
        row.append(args.label)
        writer.writerow(row)
        collected[0] += 1
        elapsed = time.time() - start_time[0]
        remaining = args.seconds - elapsed
        print(f"\r  [{args.label}]  {collected[0]} paket  |  {remaining:.0f}s kaldı   ", end="", flush=True)
        if elapsed >= args.seconds:
            reader.stop()

    reader = ESP32SerialReader(port, on_csi)
    if not reader.start():
        print(f"HATA: Port açılamadı — {reader.last_error}")
        sys.exit(1)

    print(f"\nKayıt başlıyor: label='{args.label}'  süre={args.seconds}s  port={port}")
    print(f"{'='*50}")
    print(f"  Şimdi '{args.label}' için hazırlan.")
    print(f"  Kayıt otomatik duracak.")
    print(f"{'='*50}\n")

    for i in range(5, 0, -1):
        print(f"\r  Kayıt {i} saniye sonra başlıyor...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r  KAYIT BASLADI — {args.label}                 \n")

    start_time[0] = time.time()

    # Süre dolana veya reader durana kadar bekle
    while reader.connected or start_time[0] is None:
        time.sleep(0.2)
        if start_time[0] and (time.time() - start_time[0]) > args.seconds + 2:
            break

    reader.stop()
    csv_file.flush()
    csv_file.close()

    print(f"\n\nTamamlandı! {collected[0]} paket kaydedildi → {DATA_FILE}")
    print(f"Toplam CSV satırı: {_count_rows()}")


def _count_rows():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE) as f:
        return sum(1 for _ in f) - 1  # başlık satırı çıkar


if __name__ == "__main__":
    main()
