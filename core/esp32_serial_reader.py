"""
ESP32'den gelen CSI satırlarını arka plan thread'inde okur.
Format: "CSI:<amp0>,<amp1>,...,<ampN>\n"
"""

import serial
import threading
import time
import numpy as np
from typing import Optional, Callable, List


class ESP32SerialReader:

    BAUD = 921600

    def __init__(self, port: str, on_csi: Callable[[np.ndarray], None]):
        self._port     = port
        self._on_csi   = on_csi
        self._ser: Optional[serial.Serial] = None
        self._running  = False
        self._thread: Optional[threading.Thread] = None

        self.packet_count  = 0
        self.csi_rate_hz   = 0.0   # son 1 sn'deki paket/sn
        self.last_error: Optional[str] = None
        self.connected     = False

        self._rate_count   = 0
        self._rate_ts      = time.time()

    # ── Başlatma / Durdurma ───────────────────────────────────────────────────
    def start(self) -> bool:
        try:
            self._ser = serial.Serial(
                self._port, self.BAUD,
                timeout=0.5,
                write_timeout=1.0
            )
            self._running = True
            self.connected = True
            self._thread = threading.Thread(
                target=self._read_loop, daemon=True, name="ESP32Reader")
            self._thread.start()
            return True
        except serial.SerialException as e:
            self.last_error = str(e)
            self.connected  = False
            return False

    def stop(self):
        self._running = False
        self.connected = False
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ── Okuma Döngüsü (arka plan thread) ─────────────────────────────────────
    def _read_loop(self):
        while self._running and self._ser and self._ser.is_open:
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()

                if line.startswith("CSI:"):
                    self._parse_csi(line[4:])
                # ">> STATUS ..." gibi durum satırlarını sessizce geç

            except serial.SerialException as e:
                self.last_error = str(e)
                self.connected  = False
                break
            except Exception:
                pass

        self.connected = False

    def _parse_csi(self, payload: str):
        try:
            vals = [float(x) for x in payload.split(",") if x.strip()]
        except ValueError:
            return
        if len(vals) < 16:
            return

        arr = np.array(vals, dtype=np.float32)

        # Hz sayacı güncelle
        self.packet_count += 1
        self._rate_count  += 1
        now = time.time()
        elapsed = now - self._rate_ts
        if elapsed >= 1.0:
            self.csi_rate_hz = self._rate_count / elapsed
            self._rate_count = 0
            self._rate_ts    = now

        self._on_csi(arr)

    # ── Port Tarama Yardımcısı (UI'dan çağrılabilir) ─────────────────────────
    @staticmethod
    def find_esp32_ports() -> List[str]:
        """Mac'te olası ESP32 portlarını döndürür."""
        import glob
        candidates = []
        patterns = [
            "/dev/cu.usbserial*",
            "/dev/cu.SLAB_USBtoUART*",
            "/dev/cu.wchusbserial*",
            "/dev/cu.usbmodem*",
        ]
        for pat in patterns:
            candidates.extend(glob.glob(pat))
        return sorted(candidates)
