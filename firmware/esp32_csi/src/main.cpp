/*
 * Wi-Fi CSI Radar — ESP32 Firmware v1.0
 * PlatformIO / Arduino Framework
 *
 * Çıktı formatı (Python tarafı bu formatı parse eder):
 *   CSI:<amp0>,<amp1>,...,<ampN>\n
 *   Örn: CSI:64.2,31.5,88.0,12.3,...
 *
 * PASSIVE_MODE=0 ise modeme bağlanır, daha temiz CSI alır.
 * PASSIVE_MODE=1 ise beacon/probe paketlerinden pasif okur.
 */

#include <Arduino.h>
#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "nvs_flash.h"
#include "csi_config.h"

// ─── Global sayaçlar ──────────────────────────────────────────────────────────
static volatile uint32_t g_packet_count = 0;
static volatile uint32_t g_csi_count    = 0;

// ─── CSI Callback (ISR benzeri — hızlı tutulmalı) ────────────────────────────
static void IRAM_ATTR wifi_csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len < 2) return;

    // Amplitüd hesabı: I/Q çiftlerinden sqrt(I²+Q²)
    // info->buf: int8_t dizisi, [I0, Q0, I1, Q1, ...]
    int pairs = info->len / 2;
    if (pairs > CSI_MAX_SUBCARRIERS) pairs = CSI_MAX_SUBCARRIERS;

    // Serial.print ISR'dan çağrılmamalı — flag + buffer yaklaşımı
    // Basit PoC için direkt print kullanıyoruz (yeterince yavaş bir ISR değil)
    Serial.print("CSI:");
    for (int i = 0; i < pairs; i++) {
        int8_t I_val = info->buf[i * 2];
        int8_t Q_val = info->buf[i * 2 + 1];
        float  amp   = sqrtf((float)(I_val * I_val) + (float)(Q_val * Q_val));
        Serial.print(amp, 1);
        if (i < pairs - 1) Serial.print(',');
    }
    Serial.println();

    g_csi_count++;
    g_packet_count++;
}

// ─── Pasif mod: promiscuous callback ─────────────────────────────────────────
#if PASSIVE_MODE
static void promiscuous_cb(void *buf, wifi_promiscuous_pkt_type_t type) {
    // Promiscuous modda CSI callback zaten tetiklenir,
    // bu fonksiyon sadece MGMT frame sayımı için.
    (void)buf; (void)type;
}
#endif

// ─── WiFi + CSI başlatma ─────────────────────────────────────────────────────
static void init_wifi_csi() {
    // NVS + netif + event loop
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();

#if PASSIVE_MODE
    // Pasif mod: STA başlat ama bağlanma
    WiFi.mode(WIFI_STA);
    WiFi.disconnect(true);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_promiscuous_rx_cb(promiscuous_cb);
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    Serial.println(">> MODE: PASSIVE (beacon/probe capture)");
#else
    // Aktif mod: modeme bağlan
    Serial.printf(">> CONNECTING to SSID: %s ...\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    uint8_t retries = 0;
    while (WiFi.status() != WL_CONNECTED && retries < 20) {
        delay(500);
        Serial.print('.');
        retries++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n>> WIFI CONNECTED  IP: %s  CH: %d  RSSI: %d dBm\n",
            WiFi.localIP().toString().c_str(),
            WiFi.channel(),
            WiFi.RSSI());
    } else {
        Serial.println("\n!! WIFI FAILED — falling back to passive mode");
        esp_wifi_set_promiscuous(true);
        esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    }
#endif

    // CSI yapılandırması
    wifi_csi_config_t csi_cfg = {};
    csi_cfg.lltf_en           = true;   // Long Training Field (52 subcarrier)
    csi_cfg.htltf_en          = true;   // HT-LTF (56 subcarrier, 802.11n)
    csi_cfg.stbc_htltf2_en    = true;
    csi_cfg.ltf_merge_en      = true;   // LLTF + HTLTF birleştir
    csi_cfg.channel_filter_en = false;  // Ham kanal verisi (filtre kapalı = daha fazla bilgi)
    csi_cfg.manu_scale        = false;
    csi_cfg.shift             = 0;

    esp_wifi_set_csi_config(&csi_cfg);
    esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL);
    esp_wifi_set_csi(true);
}

// ─── Durum LED (GPIO2 = built-in LED on most ESP32 boards) ───────────────────
#define STATUS_LED 2

static void blink_led(int times, int ms) {
    for (int i = 0; i < times; i++) {
        digitalWrite(STATUS_LED, HIGH);
        delay(ms);
        digitalWrite(STATUS_LED, LOW);
        delay(ms);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(200);

    pinMode(STATUS_LED, OUTPUT);
    blink_led(3, 100);  // Boot sinyali

    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  Wi-Fi CSI Radar  Firmware v1.0  ║");
    Serial.println("╚══════════════════════════════════╝");
    Serial.printf(">> Build: %s %s\n", __DATE__, __TIME__);
    Serial.printf(">> Chip: %s  Rev: %d  Cores: %d  Freq: %d MHz\n",
        ESP.getChipModel(),
        ESP.getChipRevision(),
        ESP.getChipCores(),
        ESP.getCpuFreqMHz());
    Serial.printf(">> Free heap: %u bytes\n", ESP.getFreeHeap());

    init_wifi_csi();

    blink_led(2, 50);  // Hazır sinyali
    Serial.println(">> CSI_READY");
    Serial.println(">> Streaming CSI amplitudes... (format: CSI:v0,v1,...,vN)");
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    static uint32_t last_report_ms = 0;
    static uint32_t last_csi_snap  = 0;

    uint32_t now = millis();

    if (now - last_report_ms > 5000) {
        uint32_t rate = (g_csi_count - last_csi_snap) / 5;
        Serial.printf(">> STATUS  uptime:%lus  packets:%lu  csi_rate:%lu/s  heap:%u\n",
            now / 1000,
            g_packet_count,
            rate,
            ESP.getFreeHeap());
        last_csi_snap  = g_csi_count;
        last_report_ms = now;
    }

    delay(1);
}
