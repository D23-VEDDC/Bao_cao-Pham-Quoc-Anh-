/*
 * VEDC PROTOCOL MODULE - UART HARDWARE & FRAME PARSER
 */

#ifndef APP_PROTOCOL_H
#define APP_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

/* UART Hardware Configuration */
#define STM32_UART_NUM        UART_NUM_1
#define STM32_RX_GPIO         18
#define STM32_TX_GPIO         17
#define STM32_BAUD            115200
#define UART_RX_BUF_SIZE      4096
#define UART_QUEUE_LEN        32
#define UART_READ_TIMEOUT_MS  50

/* Protocol Framing Delimiters */
#define FRAME_STX           0xAA
#define FRAME_ETX           0x55
#define FRAME_MAX_DATA      64

/* STM32 -> ESP32 Commands */
#define CMD_LEARN_START     0x10
#define CMD_CAPTURE_SAMPLE  0x11
#define CMD_POWER_SAMPLE    0x12
#define CMD_TRAIN_START     0x13
#define CMD_PING            0x14
#define CMD_CHECK_START     0x20
#define CMD_VISUAL_CHECK    0x21
#define CMD_POWER_CHECK     0x22
#define CMD_CHECK_NEXT      0x23

/* ESP32 -> STM32 Responses */
#define CMD_CAPTURE_ACK     0x91
#define CMD_POWER_ACK       0x92
#define CMD_TRAIN_DONE      0x93
#define CMD_ERROR           0x94
#define CMD_ACK             0xFF
#define CMD_VISUAL_RESULT   0xA1
#define CMD_POWER_RESULT    0xA2

/* System Error Codes */
#define ERR_CAMERA_FAIL     0x01
#define ERR_FEATURE_FAIL    0x02
#define ERR_MEMORY_FULL     0x03
#define ERR_TRAIN_FAIL      0x04
#define ERR_UNKNOWN         0xFF

/* Inspection Result Status */
#define RESULT_PASS         0x00
#define RESULT_FAIL         0x01

/* RX Frame Structure */
typedef struct {
    uint8_t cmd;
    uint8_t len;
    uint8_t data[FRAME_MAX_DATA];
} rx_frame_t;

/**
 * @brief Khởi tạo UART1, tạo Queue nhận dữ liệu và kích hoạt Task Parser / Task Event.
 * 
 * @param out_frame_queue Con trỏ nhận Handle của FreeRTOS Queue chứa các rx_frame_t hoàn chỉnh.
 * @return esp_err_t ESP_OK nếu thành công.
 */
esp_err_t app_protocol_init(QueueHandle_t *out_frame_queue);

/**
 * @brief Đóng gói và gửi một Frame chuẩn [STX | LEN | CMD | DATA | CRC8 | ETX] qua UART1.
 * 
 * @param cmd Mã lệnh gửi cho STM32.
 * @param data Con trỏ buffer dữ liệu (có thể NULL nếu len = 0).
 * @param len Độ dài dữ liệu (byte).
 * @return esp_err_t ESP_OK nếu gửi thành công.
 */
esp_err_t app_protocol_send_frame(uint8_t cmd, const uint8_t *data, uint8_t len);

/**
 * @brief Bắn thông báo lỗi CMD_ERROR (0x94) về cho STM32.
 * 
 * @param err_code Mã lỗi (ERR_CAMERA_FAIL, ERR_TRAIN_FAIL,...).
 * @param reason Dòng mô tả nguyên nhân lỗi để log internal.
 */
void app_protocol_send_error(uint8_t err_code, const char *reason);

/* Các hàm Helper xử lý Endianness, CRC8 & Giới hạn dữ liệu */
uint8_t protocol_crc8(const uint8_t *data, uint8_t len);
void pack_float_le(uint8_t *dst, float value);
float unpack_float_le(const uint8_t *src);
void pack_u16_le(uint8_t *dst, uint16_t value);
void pack_u32_le(uint8_t *dst, uint32_t value);
uint16_t clamp_u16(uint32_t value);

#ifdef __cplusplus
}
#endif

#endif // APP_PROTOCOL_H