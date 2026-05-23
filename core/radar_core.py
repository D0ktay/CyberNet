import serial
import time
import threading
import numpy as np
from scipy.signal import butter, lfilter
import os
import csv

# ==========================================
# ⚙️ MACBOOK PRO INTEL & PORT AYARLARI
# ==========================================
SERIAL_PORT = '/dev/cu.usbserial-1410'  # Donanım gelince 'ls /dev/cu.*' ile güncelleyeceğiz
BAUD_RATE = 921600                      # Saniyede ~92 KB veri akışı için yüksek hız
DATA_FILE = 'data/csi_fingerprints.csv'  # Radyo parmak izlerinin kaydolacağı yer

class WifiCsiRadar:
    def __init__(self):
        self.is_running = False
        self.serial_conn = None
        self.current_label = "Durağan/Boş" # O an toplanan bölgenin adı (Bölge_A vb.)
        self.is_collecting = False          # Veri toplama modu açık mı?
        
        # Butterworth Filtre Parametreleri (0.5 Hz - 10 Hz arası insan hareketidir)
        self.b, self.a = butter(2, [0.5, 10.0], btype='bandpass', fs=100.0) 

    def calculate_amplitude(self, i_vals, q_vals):
        """Ham I ve Q koordinatlarından saf Genlik (Amplitude) hesaplar"""
        # Formül: sqrt(I^2 + Q^2)
        return np.sqrt(np.square(i_vals) + np.square(q_vals))

    def apply_filter(self, data_buffer):
        """SciPy kullanarak statik duvar yansımalarını (gürültüyü) temizler"""
        if len(data_buffer) < 15: # Filtrenin kararlı çalışması için minimum paket sayısı
            return data_buffer
        return lfilter(self.b, self.a, data_buffer, axis=0)

    def parse_csi_line(self, line):
        """ESP32'den gelen ham string satırını anlamlı sayılara böler"""
        try:
            # Örnek gelen satır: CSI_DATA,-52,64,[12,-4,15,2,...]
            if not line.startswith("CSI_DATA"):
                return None
                
            parts = line.split(",[")
            header_parts = parts[0].split(",")
            rssi = int(header_parts[1])
            
            # İçerideki I/Q sayılarını ayıkla
            raw_iq = parts[1].replace("]", "").split(",")
            iq_numbers = np.array([int(x) for x in raw_iq])
            
            # Çift indeksler I, tek indeksler Q'dur
            i_vals = iq_numbers[0::2]
            q_vals = iq_numbers[1::2]
            
            # Genlik matrisini hesapla
            amplitude = self.calculate_amplitude(i_vals, q_vals)
            return amplitude
            
        except Exception as e:
            # Hatalı veya eksik gelen satırları pas geç
            return None

    def save_to_csv(self, filtered_amplitude):
        """Odayı 1'er metrekarelik alanlara böldüğümüzde parmak izini kaydeder"""
        if not self.is_collecting:
            return
            
        file_exists = os.path.isfile(DATA_FILE)
        with open(DATA_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Eğer dosya yeni açılıyorsa başlık satırını (Header) ekle
            if not file_exists:
                header = [f"subcarrier_{i}" for i in range(len(filtered_amplitude))] + ["Bölge_Etiketi"]
                writer.writerow(header)
            
            # Temizlenmiş radyo dalgasını ve hangi bölgeye ait olduğunu yaz
            row = list(filtered_amplitude) + [self.current_label]
            writer.writerow(row)

    def read_serial_loop(self):
        """Mac'i kilitlememesi için arka planda (Thread) asenkron çalışan döngü"""
        data_window = []
        while self.is_running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    raw_line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    amplitude = self.parse_csi_line(raw_line)
                    
                    if amplitude is not None:
                        data_window.append(amplitude)
                        if len(data_window) > 100: # Son 100 paketi hafızada tut
                            data_window.pop(0)
                        
                        # Butterworth filtresini uygula
                        filtered_matrix = self.apply_filter(np.array(data_window))
                        filtered_data = filtered_matrix[-1]
                        
                        # Eğer kayıt modundaysak CSV'ye bas
                        if self.is_collecting:
                            self.save_to_csv(filtered_data)
                            print(f"📸 {self.current_label} için Parmak İzi Kaydedildi...")
                            
            except Exception as e:
                print(f"⚠️ Veri okuma hatası: {e}")
                time.sleep(0.1)

    def start(self):
        self.is_running = True
        print(f"📡 Seri Port {SERIAL_PORT} bağlanıyor...")
        try:
            self.serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            print("✅ Radar başarıyla tetiklendi! Cihazsız algılama modülü aktif.")
            
            # Arka plan iş parçacığını (Thread) başlatıyoruz
            self.thread = threading.Thread(target=self.read_serial_loop)
            self.thread.daemon = True
            self.thread.start()
        except serial.SerialException:
            print(f"❌ Donanım Hatası: {SERIAL_PORT} bulunamadı. Donanımlar gelince burası canlanacak.")

    def stop(self):
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        print("🔌 Radar güvenli bir şekilde kapatıldı.")

if __name__ == "__main__":
    # Kodun tek başına çalışabilirliğini test etmek için
    radar = WifiCsiRadar()
    radar.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        radar.stop()