"""
3 ESP32'den gelen CSI akışlarını eşzamanlı okur ve tek birleşik vektöre dönüştürür.
Her kanal kendi arka plan thread'inde okunur (ESP32SerialReader).

Birleştirme stratejisi — OLAY TABANLI / KAYIPSIZ KALİBRASYON:
  Sabit aralıkta "en son neyse onu al" yöntemi YANLIŞ çünkü:
    (a) Kanallar farklı hızlarda akıyorsa (ör. 38 Hz vs 82 Hz), yavaş kanaldan
        aynı eski örnek defalarca tekrar kullanılır (bayat veri / yanlış kalibrasyon)
    (b) Hızlı kanallardan üretilen örneklerin çoğu hiç kullanılmadan kaybolur (israf)
  Bunun yerine: her kanaldan yeni bir CSI paketi geldiğinde 'taze' olarak işaretlenir;
  3 kanaldan da en az bir taze örnek birikince ANINDA birleştirilip dışarı verilir
  ve bayraklar sıfırlanır. Böylece:
    - Hiçbir kanaldan veri çöpe gitmez (her biri en az bir kez birleşime katkı sağlar)
    - Çıktı hızı doğal olarak EN YAVAŞ kanalın hızına eşitlenir (gerçek kalibrasyon)
    - Üç örnek de zaman olarak birbirine yakın anlarda toplanmış olur

Sıra: [sol(-45°), orta(0°), sağ(+45°)]
"""

import time
import threading
import numpy as np
from typing import Optional, Callable, List

from core.esp32_serial_reader import ESP32SerialReader

N_SUBCARRIERS = 64
N_CHANNELS = 3
COMBINED_LEN = N_SUBCARRIERS * N_CHANNELS

CHANNEL_LABELS = ["RX-1 SOL (-45°)", "RX-2 ORTA (0°)", "RX-3 SAĞ (+45°)"]

# Fiziksel yerleşime göre sabit port eşlemesi (kullanıcı tarafından doğrulandı):
#   usbserial-5    → SOL   (-45°)
#   usbserial-6    → ORTA  (0°)
#   usbserial-0001 → SAĞ   (+45°)
# NOT: Bu isimler alfabetik sıralamayla fiziksel sırayla ÖRTÜŞMÜYOR —
# bu yüzden otomatik bulunan portları bu sabit eşlemeyle yeniden sıralamak gerekir.
PORT_POSITION = {
    "usbserial-5": 0,
    "usbserial-6": 1,
    "usbserial-0001": 2,
}


def order_ports_by_position(found_ports: List[str]) -> Optional[List[str]]:
    """find_esp32_ports() çıktısını [SOL, ORTA, SAĞ] fiziksel sırasına göre dizer.
    Bilinmeyen bir port adıyla karşılaşırsa None döner (manuel eşleme gerekir)."""
    if len(found_ports) != N_CHANNELS:
        return None
    ordered: List[Optional[str]] = [None] * N_CHANNELS
    for port in found_ports:
        matched = False
        for suffix, pos in PORT_POSITION.items():
            if port.endswith(suffix):
                ordered[pos] = port
                matched = True
                break
        if not matched:
            return None
    if any(p is None for p in ordered):
        return None
    return ordered


class ESP32MultiReader:
    """3 ESP32'yi paralel okuyup, her kanaldan en az bir taze örnek geldiği anda
    kayıpsız ve kalibre biçimde birleştirip tek vektör üretir."""

    def __init__(self, ports: List[str], on_combined: Callable[[np.ndarray], None]):
        if len(ports) != N_CHANNELS:
            raise ValueError(f"{N_CHANNELS} port gerekli (sol/orta/sağ), {len(ports)} verildi")

        self._ports = ports
        self._on_combined = on_combined

        self._latest: List[Optional[np.ndarray]] = [None] * N_CHANNELS
        self._latest_ts: List[float] = [0.0] * N_CHANNELS
        self._fresh: List[bool] = [False] * N_CHANNELS
        self._lock = threading.Lock()

        # Tanı amaçlı: son birleşimde kanallar arası en büyük zaman farkı (sn)
        self.last_skew_s: float = 0.0
        self.combined_count: int = 0

        self._readers: List[ESP32SerialReader] = [
            ESP32SerialReader(port, self._make_channel_cb(i))
            for i, port in enumerate(ports)
        ]

        self._running = False

    def _make_channel_cb(self, idx: int) -> Callable[[np.ndarray], None]:
        def _cb(arr: np.ndarray):
            self._on_channel_csi(idx, arr)
        return _cb

    def _on_channel_csi(self, idx: int, arr: np.ndarray):
        emit_snapshot = None
        emit_ts = None
        with self._lock:
            self._latest[idx] = arr
            self._latest_ts[idx] = time.time()
            self._fresh[idx] = True

            if all(self._fresh) and all(x is not None for x in self._latest):
                emit_snapshot = list(self._latest)
                emit_ts = list(self._latest_ts)
                self._fresh = [False] * N_CHANNELS

        if emit_snapshot is not None:
            self._emit(emit_snapshot, emit_ts)

    def _emit(self, snapshot: List[np.ndarray], ts: List[float]):
        parts = []
        for arr in snapshot:
            if len(arr) >= N_SUBCARRIERS:
                part = arr[:N_SUBCARRIERS]
            else:
                part = np.pad(arr, (0, N_SUBCARRIERS - len(arr)))
            parts.append(part)

        combined = np.concatenate(parts).astype(np.float32)
        self.last_skew_s = max(ts) - min(ts)
        self.combined_count += 1
        self._on_combined(combined)

    # ── Başlatma / Durdurma ───────────────────────────────────────────────────
    def start(self) -> bool:
        all_ok = True
        for r in self._readers:
            if not r.start():
                all_ok = False
        self._running = True
        return all_ok

    def stop(self):
        self._running = False
        for r in self._readers:
            r.stop()

    # ── Durum ────────────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return all(r.connected for r in self._readers)

    def channel_status(self) -> List[dict]:
        return [
            {
                "label": CHANNEL_LABELS[i],
                "port": self._ports[i],
                "connected": r.connected,
                "rate_hz": r.csi_rate_hz,
                "last_error": r.last_error,
            }
            for i, r in enumerate(self._readers)
        ]
