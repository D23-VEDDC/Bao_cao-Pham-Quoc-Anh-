/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VEDC_CONTROLLER_H
#define VEDC_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Khởi tạo phần cứng UART1 và các FreeRTOS Task điều khiển VEDC.
 *
 * Hàm thực hiện:
 * - Cấu hình UART1 (Baud 115200 8N1: RX=GPIO18, TX=GPIO17) giao tiếp với STM32F103C8T6.
 * - Khởi tạo Queue nhận khung truyền giao thức (`rx_frame_t`).
 * - Khởi tạo bộ nhớ lưu trữ mẫu huấn luyện.
 * - Tạo Task thu nhận dữ liệu UART (`stm32_uart_rx_task`).
 * - Tạo Task xử lý lệnh lệnh thực thi (`command_worker_task`) cho hai giai đoạn LEARN và CHECK.
 *
 * @return 
 *  - ESP_OK: Khởi tạo thành công.
 *  - ESP_ERR_NO_MEM: Không đủ bộ nhớ để tạo Queue hoặc Task.
 *  - Mã lỗi khác từ `uart_driver_install` / `uart_param_config` nếu cấu hình UART thất bại.
 */
esp_err_t vedc_controller_start(void);

#ifdef __cplusplus
}
#endif

#endif // VEDC_CONTROLLER_H