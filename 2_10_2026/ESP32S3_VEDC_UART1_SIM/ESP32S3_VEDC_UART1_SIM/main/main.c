/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "app_usb_host.h"
#include "app_uac_manager.h"
#include "app_uvc_manager.h"
#include "app_wifi.h"
#include "app_web.h"
#include "app_ai_patch.h"
#include "app_ai_infer.h"
#include "app_ai_coreset.h"
#include "app_ai_session.h"
#include "vedc_controller.h"

static const char *TAG = "main_app";

void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_INFO);

    /* Khởi tạo Flash NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "NVS init failed: %s", esp_err_to_name(ret));
        return;
    }

    /*
     * QUAN TRỌNG: Khởi động UART1 (giao tiếp STM32) NGAY từ đầu, trước khi
     * USB Host / UVC / UAC / Wi-Fi / Web Server khởi tạo. Các module này có
     * thể mất 1-3 giây (dò USB device, mở AP Wi-Fi, khởi động HTTP server...).
     * Nếu STM32 gửi lệnh (VD: PB0 -> 0x10) ngay khi vừa cấp nguồn, mà UART1
     * chưa được uart_driver_install()/uart_set_pin(), thì byte đó sẽ RỚT MẤT
     * hoàn toàn (không có driver nào lắng nghe GPIO18) -> ESP32 "không nhận,
     * không phản hồi" dù dây và CRC hoàn toàn đúng. Khởi tạo UART1 sớm giúp
     * ESP32 sẵn sàng nhận lệnh ngay từ những mili-giây đầu tiên sau boot.
     */
    ESP_ERROR_CHECK(vedc_controller_start());

    /* Khởi tạo USB Host, Camera UVC & Audio UAC */
    ESP_ERROR_CHECK(app_usb_host_start());
    ESP_ERROR_CHECK(app_uvc_manager_start());
    ESP_ERROR_CHECK(app_uac_manager_start());

    /* Khởi tạo Wi-Fi & Web Server Console */
    ESP_ERROR_CHECK(app_wifi_start());
    ESP_ERROR_CHECK(app_web_start());

    /* Khởi tạo bộ cắt Patch 40x40 trên PSRAM */
    esp_err_t ai_patch_ret = app_ai_patch_init();
    if (ai_patch_ret != ESP_OK) {
        ESP_LOGW(TAG, "AI patch module init failed: %s", esp_err_to_name(ai_patch_ret));
    }

    /* Khởi tạo TFLite-Micro Inference Engine */
    esp_err_t ai_infer_ret = app_ai_infer_init();
    if (ai_infer_ret != ESP_OK) {
        ESP_LOGW(TAG, "AI infer module init failed: %s", esp_err_to_name(ai_infer_ret));
    }

    /* Khởi tạo Coreset Memory Bank Engine */
    esp_err_t ai_coreset_ret = app_ai_coreset_init();
    if (ai_coreset_ret != ESP_OK) {
        ESP_LOGW(TAG, "AI coreset init failed: %s", esp_err_to_name(ai_coreset_ret));
    }

    ESP_LOGI(TAG, "==========================================================");
    ESP_LOGI(TAG, " HỆ THỐNG VEDC ESP32-S3 AI INSPECTION ĐÃ SẴN SÀNG");
    ESP_LOGI(TAG, " Web Console UI: http://192.168.4.1");
    ESP_LOGI(TAG, " Giao tiếp UART1 STM32 (GPIO18 RX / GPIO17 TX) - Enabled");
    ESP_LOGI(TAG, "==========================================================");
}