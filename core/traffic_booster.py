"""
ESP32'lerin IP adreslerine düşük seviyeli sürekli UDP trafiği göndererek
CSI yakalama hızını artırır.

Neden işe yarar: ESP32'nin CSI callback'i, kendisine yönelik 802.11 çerçeve
trafiği geldiğinde tetiklenir. Ev Wi-Fi'sinde boşta trafik azken hız ~0.5-2 Hz
civarında kalıyor. Bilgisayardan ESP32'nin IP'sine sürekli paket göndermek,
router üzerinden ESP32'ye yönlendirilen çerçeve sayısını artırır → CSI hızı
~50-80 Hz'e çıkar (ölçüldü).

Not: UDP paketler herhangi bir porta gönderiliyor, ESP32'nin onu dinlemesi
gerekmiyor — amaç sadece 802.11 çerçeve trafiği üretmek (ICMP ping ile aynı
mantık, ama macOS'ta düşük aralıklı ping root gerektirdiği için UDP kullanıyoruz).
"""

import socket
import threading
import time
from typing import List, Optional


class TrafficBooster:
    """Verilen IP listesine arka planda sürekli UDP paketi gönderir."""

    def __init__(self, ips: List[str], udp_port: int = 9999, packets_per_sec: float = 100.0):
        self._ips = ips
        self._udp_port = udp_port
        self._interval = 1.0 / packets_per_sec
        self._running = False
        self._threads: List[threading.Thread] = []

    def start(self):
        if self._running:
            return
        self._running = True
        payload = b"x" * 32
        for ip in self._ips:
            t = threading.Thread(
                target=self._flood, args=(ip, payload),
                daemon=True, name=f"TrafficBoost-{ip}")
            t.start()
            self._threads.append(t)

    def _flood(self, ip: str, payload: bytes):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while self._running:
                try:
                    sock.sendto(payload, (ip, self._udp_port))
                except OSError:
                    pass
                time.sleep(self._interval)
        finally:
            sock.close()

    def stop(self):
        self._running = False
        self._threads.clear()


def discover_esp32_ips(ports: List[str], baud: int = 921600, timeout_s: float = 12.0) -> Optional[List[str]]:
    """Verilen seri portlardaki ESP32'leri resetleyip boot loglarından IP'lerini okur.
    Sıra, verilen `ports` listesiyle birebir eşleşir. Bulunamazsa None döner."""
    import re
    import serial

    ip_re = re.compile(r"IP:\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)")
    ips: List[Optional[str]] = []

    for port in ports:
        found_ip = None
        try:
            ser = serial.Serial(port, baud, timeout=1)
            ser.dtr = False
            ser.rts = True
            time.sleep(0.1)
            ser.rts = False
            time.sleep(0.1)

            t0 = time.time()
            while time.time() - t0 < timeout_s:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if line:
                    m = ip_re.search(line)
                    if m:
                        found_ip = m.group(1)
                        break
            ser.close()
        except Exception:
            found_ip = None

        ips.append(found_ip)

    if any(ip is None for ip in ips):
        return None
    return ips
