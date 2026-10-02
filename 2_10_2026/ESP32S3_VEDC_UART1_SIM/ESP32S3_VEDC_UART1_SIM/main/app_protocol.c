/*
 * VEDC PROTOCOL MODULE IMPLEMENTATION
 */

#include "app_protocol.h"
#include <string.h>
#include "driver/gpio.h"
#include "esp_log.h"

#define TAG "APP_PROTOCOL"

typedef enum {
    RX_WAIT_STX = 0,
    RX_WAIT_LEN,
    RX_WAIT_CMD,
    RX_WAIT_DATA,
    RX_WAIT_CRC,
    RX_WAIT_ETX
} rx_state_t;

static rx_state_t s_rx_state = RX_WAIT_STX;
static rx_frame_t s_rx_frame;
static uint8_t s_rx_index = 0;

static QueueHandle_t s_frame_queue = NULL;
static QueueHandle_t s_uart_event_queue = NULL;

/* Utility Helpers */
static uint8_t crc8_update(uint8_t crc, uint8_t data) {
    crc ^= data;
    for (uint8_t i = 0; i < 8; i++) {
        crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x07U) : (uint8_t)(crc << 1);
    }
    return crc;
}

uint8_t protocol_crc8(const uint8_t *data, uint8_t len) {
    uint8_t crc = 0;
    for (uint8_t i = 0; i < len; i++) {
        crc = crc8_update(crc, data[i]);
    }
    return crc;
}

void pack_float_le(uint8_t *dst, float value) {
    memcpy(dst, &value, sizeof(value));
}

float unpack_float_le(const uint8_t *src) {
    float value;
    memcpy(&value, src, sizeof(value));
    return value;
}

void pack_u16_le(uint8_t *dst, uint16_t value) {
    dst[0] = (uint8_t)(value & 0xFF);
    dst[1] = (uint8_t)((value >> 8) & 0xFF);
}

void pack_u32_le(uint8_t *dst, uint32_t value) {
    memcpy(dst, &value, sizeof(value));
}

uint16_t clamp_u16(uint32_t value) {
    return (value > 0xFFFFU) ? 0xFFFFU : (uint16_t)value;
}

/* TX Routines */
esp_err_t app_protocol_send_frame(uint8_t cmd, const uint8_t *data, uint8_t len) {
    if (len > FRAME_MAX_DATA) return ESP_ERR_INVALID_ARG;

    uint8_t frame[FRAME_MAX_DATA + 5];
    uint8_t crc_buf[FRAME_MAX_DATA + 2];
    size_t p = 0;

    frame[p++] = FRAME_STX;
    frame[p++] = len;
    frame[p++] = cmd;

    crc_buf[0] = len;
    crc_buf[1] = cmd;

    if (len > 0 && data != NULL) {
        memcpy(&frame[p], data, len);
        memcpy(&crc_buf[2], data, len);
        p += len;
    }

    frame[p++] = protocol_crc8(crc_buf, (uint8_t)(len + 2));
    frame[p++] = FRAME_ETX;

    int written = uart_write_bytes(STM32_UART_NUM, frame, p);
    esp_err_t tx_ret = (written == (int)p) ? ESP_OK : ESP_FAIL;

    if (tx_ret == ESP_OK) {
        ESP_LOGI(TAG, "UART1 TX -> STM32: CMD=0x%02X LEN=%u (%u bytes) OK", cmd, len, (unsigned)p);
    } else {
        ESP_LOGE(TAG, "UART1 TX THẤT BẠI: CMD=0x%02X (ghi %d/%u bytes)", cmd, written, (unsigned)p);
    }
    return tx_ret;
}

void app_protocol_send_error(uint8_t err_code, const char *reason) {
    uint8_t data[1] = { err_code };
    ESP_LOGW(TAG, "ERROR 0x%02X: %s", err_code, reason ? reason : "unknown");
    app_protocol_send_frame(CMD_ERROR, data, 1);
}

/* RX Parser State Machine */
static void parser_feed_byte(uint8_t b) {
    ESP_LOGI(TAG, "UART1 RAW RX: 0x%02X", b);

    switch (s_rx_state) {
    case RX_WAIT_STX:
        if (b == FRAME_STX) { s_rx_state = RX_WAIT_LEN; s_rx_index = 0; }
        break;

    case RX_WAIT_LEN:
        if (b > FRAME_MAX_DATA) { s_rx_state = RX_WAIT_STX; break; }
        s_rx_frame.len = b;
        s_rx_state = RX_WAIT_CMD;
        break;

    case RX_WAIT_CMD:
        s_rx_frame.cmd = b;
        s_rx_index = 0;
        s_rx_state = (s_rx_frame.len == 0) ? RX_WAIT_CRC : RX_WAIT_DATA;
        break;

    case RX_WAIT_DATA:
        s_rx_frame.data[s_rx_index++] = b;
        if (s_rx_index >= s_rx_frame.len) s_rx_state = RX_WAIT_CRC;
        break;

    case RX_WAIT_CRC: {
        uint8_t crc_buf[FRAME_MAX_DATA + 2];
        crc_buf[0] = s_rx_frame.len;
        crc_buf[1] = s_rx_frame.cmd;
        if (s_rx_frame.len > 0) memcpy(&crc_buf[2], s_rx_frame.data, s_rx_frame.len);

        if (protocol_crc8(crc_buf, s_rx_frame.len + 2) != b) {
            ESP_LOGW(TAG, "UART1 CRC mismatch: got=0x%02X expected=0x%02X (len=%u cmd=0x%02X)",
                     b, protocol_crc8(crc_buf, s_rx_frame.len + 2), s_rx_frame.len, s_rx_frame.cmd);
            s_rx_state = RX_WAIT_STX;
            break;
        }
        s_rx_state = RX_WAIT_ETX;
        break;
    }

    case RX_WAIT_ETX:
        if (b == FRAME_ETX) {
            if (s_frame_queue == NULL) {
                ESP_LOGE(TAG, "Queue chưa được khởi tạo!");
            } else {
                if (xQueueSend(s_frame_queue, &s_rx_frame, 0) != pdTRUE) {
                    ESP_LOGW(TAG, "UART1 Queue đầy! Bỏ qua CMD=0x%02X", s_rx_frame.cmd);
                } else {
                    ESP_LOGI(TAG, "Khung truyền hợp lệ: CMD=0x%02X LEN=%u", s_rx_frame.cmd, s_rx_frame.len);
                }
            }
        }
        s_rx_state = RX_WAIT_STX;
        break;
    }
}

/* FreeRTOS Tasks */
static void stm32_uart_rx_task(void *arg) {
    uint8_t buf[128];
    while (1) {
        int n = uart_read_bytes(STM32_UART_NUM, buf, sizeof(buf), pdMS_TO_TICKS(UART_READ_TIMEOUT_MS));
        if (n > 0) {
            for (int i = 0; i < n; ++i) parser_feed_byte(buf[i]);
        }
    }
}

static void stm32_uart_event_task(void *arg) {
    uart_event_t event;
    while (1) {
        if (xQueueReceive(s_uart_event_queue, &event, portMAX_DELAY) != pdTRUE) continue;
        switch (event.type) {
        case UART_FRAME_ERR:
            ESP_LOGE(TAG, "UART1 FRAME ERROR -> Lỗi Baudrate hoặc nhiễu đường truyền!");
            uart_flush_input(STM32_UART_NUM);
            s_rx_state = RX_WAIT_STX;
            break;
        case UART_PARITY_ERR:
            ESP_LOGE(TAG, "UART1 PARITY ERROR -> Sai cấu hình 8N1!");
            uart_flush_input(STM32_UART_NUM);
            s_rx_state = RX_WAIT_STX;
            break;
        case UART_BREAK:
            ESP_LOGW(TAG, "UART1 BREAK -> Chân RX (GPIO%d) trôi nổi hoặc hở mạch GND!", STM32_RX_GPIO);
            break;
        case UART_FIFO_OVF:
        case UART_BUFFER_FULL:
            ESP_LOGW(TAG, "UART1 Buffer tràn -> Đang xả bộ đệm");
            uart_flush_input(STM32_UART_NUM);
            xQueueReset(s_uart_event_queue);
            s_rx_state = RX_WAIT_STX;
            break;
        default:
            break;
        }
    }
}

esp_err_t app_protocol_init(QueueHandle_t *out_frame_queue) {
    if (!out_frame_queue) return ESP_ERR_INVALID_ARG;

    const uart_config_t uart1_cfg = {
        .baud_rate = STM32_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t err = uart_driver_install(STM32_UART_NUM, UART_RX_BUF_SIZE, UART_RX_BUF_SIZE, 20, &s_uart_event_queue, 0);
    if (err != ESP_OK) return err;

    if ((err = uart_param_config(STM32_UART_NUM, &uart1_cfg)) != ESP_OK ||
        (err = uart_set_pin(STM32_UART_NUM, STM32_TX_GPIO, STM32_RX_GPIO, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE)) != ESP_OK) {
        uart_driver_delete(STM32_UART_NUM);
        return err;
    }

    ESP_ERROR_CHECK(gpio_set_pull_mode(STM32_RX_GPIO, GPIO_PULLUP_ONLY));
    uart_flush_input(STM32_UART_NUM);
    s_rx_state = RX_WAIT_STX;

    s_frame_queue = xQueueCreate(UART_QUEUE_LEN, sizeof(rx_frame_t));
    if (!s_frame_queue) return ESP_ERR_NO_MEM;

    *out_frame_queue = s_frame_queue;

    xTaskCreate(stm32_uart_rx_task, "stm32_uart_rx", 4096, NULL, 10, NULL);
    xTaskCreate(stm32_uart_event_task, "stm32_uart_evt", 3072, NULL, 10, NULL);

    ESP_LOGI(TAG, "Khởi tạo thành công Module Protocol UART1 (RX: GPIO%d, TX: GPIO%d)", STM32_RX_GPIO, STM32_TX_GPIO);
    return ESP_OK;
}