#pragma once

// ─── Wi-Fi Ayarları ────────────────────────────────────────────────────────────
// Ev modeminizin SSID ve şifresini buraya yazın.
// Kanal numarasını Mac'te Option+WiFi simgesi → "Kanal" satırından öğrenin.
#define WIFI_SSID     "FiberHGW_ZTG26N"
#define WIFI_PASS     "NfdsxR3PbFA9"
#define WIFI_CHANNEL  4       // Modem kanalı: 1, 6 veya 11 olası değerler

// ─── Seri Port ─────────────────────────────────────────────────────────────────
#define SERIAL_BAUD   921600  // Python tarafıyla eşleşmeli

// ─── CSI Parametreleri ─────────────────────────────────────────────────────────
// ESP32 LLTF: 52 kullanılabilir subcarrier → 52 I/Q çifti = 104 byte
// HTLTF açıksa: 56 subcarrier → 112 byte
// Burada amplitüd olarak parse edilen değer sayısı:
#define CSI_MAX_SUBCARRIERS  64   // parse edilecek maksimum subcarrier
#define CSI_PRINT_RAW        0    // 1 = ham I/Q yaz, 0 = amplitüd yaz

// ─── Pasif/Aktif Mod ──────────────────────────────────────────────────────────
// PASSIVE_MODE = 1: Sadece beacon/probe yakalar, modeme bağlanmaz
// PASSIVE_MODE = 0: Modeme bağlanır (daha kararlı CSI, power delivery gerekir)
#define PASSIVE_MODE  0
