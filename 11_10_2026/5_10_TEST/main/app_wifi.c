/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "app_wifi.h"

#include <stdio.h>
#include <string.h>
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "captive_dns.h"

#define APP_WIFI_AP_SSID            CONFIG_VEDC_AP_SSID
#define APP_WIFI_AP_PASSWORD        CONFIG_VEDC_AP_PASSWORD

static const char *TAG = "app_wifi";

esp_err_t app_wifi_start(void)
{
    esp_err_t ret = esp_netif_init();
    if (ret != ESP_OK) return ret;

    ret = esp_event_loop_create_default();
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "Create event loop failed: %s", esp_err_to_name(ret));
        return ret;
    }

    esp_netif_t *ap_netif = esp_netif_create_default_wifi_ap();
    if (!ap_netif) {
        ESP_LOGE(TAG, "Create default Wi-Fi AP netif failed");
        return ESP_FAIL;
    }

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) return ret;

    wifi_config_t wifi_config = {0};
    strlcpy((char *)wifi_config.ap.ssid, APP_WIFI_AP_SSID, sizeof(wifi_config.ap.ssid));
    strlcpy((char *)wifi_config.ap.password, APP_WIFI_AP_PASSWORD, sizeof(wifi_config.ap.password));
    wifi_config.ap.ssid_len = strlen(APP_WIFI_AP_SSID);
    wifi_config.ap.channel = 1;
    wifi_config.ap.max_connection = 4;
    wifi_config.ap.authmode = APP_WIFI_AP_PASSWORD[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;

    ret = esp_wifi_set_mode(WIFI_MODE_AP);
    if (ret != ESP_OK) return ret;
    ret = esp_wifi_set_config(WIFI_IF_AP, &wifi_config);
    if (ret != ESP_OK) return ret;
    ret = esp_wifi_start();
    if (ret != ESP_OK) return ret;
    /* LATENCY: tắt Wi-Fi power-save (mỗi lần ngủ/thức thêm tới ~100 ms trễ),
     * bật đủ giao thức b/g/n và dùng HT20 ổn định cho stream MJPEG. */
    (void)esp_wifi_set_ps(WIFI_PS_NONE);
    (void)esp_wifi_set_protocol(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    (void)esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20);
    ESP_LOGI(TAG, "Wi-Fi AP started: SSID=%s IP=192.168.4.1", APP_WIFI_AP_SSID);
    const captive_dns_config_t portal = {
        .ap_netif = ap_netif,
        .configure_dhcp_dns = true,
    };
    ret = captive_dns_start(&portal);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Captive portal DNS start failed: %s", esp_err_to_name(ret));
    }
    return ESP_OK;
}
