"""
3 ESP32'nin bağlantı durumunu kontrol eder ("ping" eşdeğeri — USB-seri cihazlar
ağ cihazı olmadığı için ICMP ping çalışmaz; bunun yerine portu açıp gerçekten
CSI verisi akıp akmadığına bakarız).

Kullanım:
    python check_esp_connections.py
    python check_esp_connections.py --seconds 5
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from core.esp32_serial_reader import ESP32SerialReader
from core.esp32_multi_reader import ESP32MultiReader, CHANNEL_LABELS, order_ports_by_position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0, help="Kontrol süresi (sn)")
    args = parser.parse_args()

    found = ESP32SerialReader.find_esp32_ports()
    print(f"Bulunan portlar ({len(found)}):")
    for p in found:
        print(f"  - {p}")

    if len(found) != 3:
        print(f"\nUYARI: 3 port bekleniyordu, {len(found)} bulundu. "
              f"USB bağlantılarını kontrol et.")
        sys.exit(1)

    ports = order_ports_by_position(found)
    if ports is None:
        print("\nUYARI: Bulunan port adları bilinen fiziksel eşlemeyle uyuşmuyor "
              "(usbserial-5/-6/-0001 bekleniyordu).")
        sys.exit(1)

    print("\nKanal <-> Port eşlemesi:")
    for lbl, p in zip(CHANNEL_LABELS, ports):
        print(f"  {lbl:18s} -> {p}")

    counts = [0, 0, 0]

    def on_combined(arr):
        pass  # sadece akışın çalıştığını doğrulamak için tetikleniyor

    reader = ESP32MultiReader(ports, on_combined)
    print(f"\n3 port açılıyor ve {args.seconds:.0f} saniye dinleniyor...\n")
    if not reader.start():
        print("HATA: Bir veya daha fazla port açılamadı:")
        for st in reader.channel_status():
            mark = "OK " if st["connected"] else "X  "
            print(f"  [{mark}] {st['label']:18s} {st['port']:28s} hata: {st['last_error']}")
        sys.exit(1)

    time.sleep(args.seconds)

    print("SONUÇ — kanal başına durum:")
    print("-" * 60)
    all_ok = True
    for st in reader.channel_status():
        alive = st["connected"] and st["rate_hz"] > 0
        mark = "✓ CANLI" if alive else "✗ SORUNLU"
        if not alive:
            all_ok = False
        print(f"  [{mark:10s}] {st['label']:18s}  port={st['port']:24s}  "
              f"hız={st['rate_hz']:.2f} paket/s  bağlı={st['connected']}")

    reader.stop()

    print("-" * 60)
    if all_ok:
        print("\n✓ 3 ESP32 de CANLI ve veri akıtıyor — veri toplamaya hazırsın.")
    else:
        print("\n✗ Bir veya daha fazla kanal veri akıtmıyor — kabloyu/portu kontrol et.")
        sys.exit(1)


if __name__ == "__main__":
    main()
