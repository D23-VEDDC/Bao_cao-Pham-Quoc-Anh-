/*
 * VEDC CONTROLLER - STATE MACHINE & AI WORKER
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "app_protocol.h"
#include "app_ai_patch.h"
#include "app_ai_infer.h"
#include "app_ai_coreset.h"
#include "vedc_controller.h"

#define TAG "VEDC_CTRL"

#define LEARN_SAMPLES          5
#define DEFAULT_THRESHOLD      1.15f
#define CURRENT_SIGMA_FACTOR   3.0f   /* Quy tắc 3-Sigma */
#define CURRENT_TOLERANCE_PCT  0.05f  /* Dung sai 5% */

/* Thống kê dòng điện */
static float g_learn_imax[LEARN_SAMPLES];
static float g_learn_iavg[LEARN_SAMPLES];

static float g_imax_mean = 300.0f;
static float g_imax_std  = 1.0f;
static float g_imax_max_dev = 0.0f;

static float g_iavg_mean = 250.0f;
static float g_iavg_std  = 1.0f;
static float g_iavg_max_dev = 0.0f;

/* State Machines */
typedef enum {
    LEARN_STATE_IDLE = 0,
    LEARN_STATE_WAIT_VISION,
    LEARN_STATE_WAIT_POWER,
    LEARN_STATE_READY_TRAIN,
    LEARN_STATE_TRAINED
} learn_state_t;

typedef enum {
    CHECK_STATE_IDLE = 0,
    CHECK_STATE_WAIT_VISUAL,
    CHECK_STATE_WAIT_POWER,
    CHECK_STATE_WAIT_NEXT
} check_state_t;

static learn_state_t g_learn_state = LEARN_STATE_IDLE;
static check_state_t g_check_state = CHECK_STATE_IDLE;

static uint8_t g_expected_vision = 1;
static uint8_t g_expected_power  = 1;
static uint8_t g_vision_count = 0;
static uint8_t g_power_count = 0;
static bool g_model_trained = false;

/* Memory lưu Feature Vector 5 mẫu ảnh */
static float *g_train_features[LEARN_SAMPLES] = {NULL};
static uint32_t g_train_feature_count[LEARN_SAMPLES] = {0};

static uint8_t g_check_board = 1;
static int64_t g_check_board_start_us = 0;

static QueueHandle_t g_frame_queue = NULL;

/* Struct kết quả kiểm tra */
typedef struct {
    uint8_t  result;
    uint16_t patch_total;
    uint16_t patch_fail_count;
    uint32_t elapsed_ms;
    float    max_dmin; /* dmin lớn nhất trong quá trình kiểm tra */
} visual_check_result_t;

typedef struct {
    uint8_t  result;
    float    imax_measured;
    float    imax_threshold;
    float    iavg_measured;
    float    iavg_threshold;
    uint32_t elapsed_ms;
} power_check_result_t;

/* Reset dữ liệu học */
static void clear_learn_storage(void) {
    memset(g_learn_imax, 0, sizeof(g_learn_imax));
    memset(g_learn_iavg, 0, sizeof(g_learn_iavg));
    for (uint32_t i = 0; i < LEARN_SAMPLES; ++i) {
        if (g_train_features[i]) {
            heap_caps_free(g_train_features[i]);
            g_train_features[i] = NULL;
        }
        g_train_feature_count[i] = 0;
    }
    app_ai_coreset_reset();

    g_vision_count = 0;
    g_power_count = 0;
    g_expected_vision = 1;
    g_expected_power = 1;
    g_imax_mean = 300.0f;
    g_iavg_mean = 250.0f;
    g_imax_std = 1.0f;
    g_iavg_std = 1.0f;
    g_imax_max_dev = 0.0f;
    g_iavg_max_dev = 0.0f;
    g_model_trained = false;
}

/* Chụp ảnh và trích xuất Feature Block */
static esp_err_t capture_real_feature_block(float **out, uint32_t *count, uint32_t *out_elapsed_ms) {
    if (!out || !count) return ESP_ERR_INVALID_ARG;
    *out = NULL;
    *count = 0;
    if (out_elapsed_ms) *out_elapsed_ms = 0;

    int64_t t_start_us = esp_timer_get_time();

    esp_err_t err = app_ai_patch_capture();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Lỗi chụp ảnh từ camera: %s", esp_err_to_name(err));
        return err;
    }

    err = app_ai_infer_run_all();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Lỗi suy luận TFLite: %s", esp_err_to_name(err));
        return err;
    }

    uint32_t elapsed_ms = (uint32_t)((esp_timer_get_time() - t_start_us) / 1000);
    if (out_elapsed_ms) *out_elapsed_ms = elapsed_ms;

    uint32_t n = app_ai_infer_get_count();
    if (n == 0) return ESP_ERR_NOT_FOUND;

    size_t bytes = (size_t)n * APP_AI_INFER_FEATURE_DIM * sizeof(float);
    float *copy = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!copy) copy = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
    if (!copy) return ESP_ERR_NO_MEM;

    for (uint32_t i = 0; i < n; ++i) {
        err = app_ai_infer_get_vector(i, copy + (size_t)i * APP_AI_INFER_FEATURE_DIM, APP_AI_INFER_FEATURE_DIM);
        if (err != ESP_OK) {
            heap_caps_free(copy);
            return err;
        }
    }
    *out = copy;
    *count = n;
    return ESP_OK;
}

/* Thực thi kiểm tra ngoại quan (CMD 0x21) */
static void perform_real_visual_check(visual_check_result_t *out) {
    memset(out, 0, sizeof(*out));
    out->result = 0xFF;

    int64_t t_start_us = esp_timer_get_time();

    float *features = NULL;
    uint32_t count = 0;
    uint32_t capture_ms = 0;
    esp_err_t err = capture_real_feature_block(&features, &count, &capture_ms);
    if (err != ESP_OK) {
        app_protocol_send_error((err == ESP_ERR_INVALID_STATE || err == ESP_ERR_TIMEOUT) ? ERR_CAMERA_FAIL : ERR_FEATURE_FAIL,
                                "Chụp ảnh/Trích xuất thất bại");
        return;
    }

    const uint32_t bank_count = app_ai_coreset_get_bank_count();
    const float threshold = app_ai_coreset_get_threshold();
    if (!app_ai_coreset_is_ready() || bank_count == 0 || threshold <= 0.0f) {
        heap_caps_free(features);
        app_protocol_send_error(ERR_TRAIN_FAIL, "Memory Bank chưa được huấn luyện");
        return;
    }

    float max_score = 0.0f;
    uint32_t fail_count = 0;
    for (uint32_t p = 0; p < count; ++p) {
        const float *v = features + (size_t)p * APP_AI_INFER_FEATURE_DIM;
        float score = app_ai_coreset_compute_score(v); /* Khoảng cách dmin của patch v tới MMB */
        if (score > threshold) fail_count++;
        if (score > max_score) max_score = score;
    }
    heap_caps_free(features);

    uint32_t elapsed_ms = (uint32_t)((esp_timer_get_time() - t_start_us) / 1000);

    out->result = (fail_count == 0) ? RESULT_PASS : RESULT_FAIL;
    out->patch_total = (uint16_t)clamp_u16(count);
    out->patch_fail_count = (uint16_t)clamp_u16(fail_count);
    out->elapsed_ms = elapsed_ms;
    out->max_dmin = max_score;

    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, " [ESP32] VISUAL CHECK BOARD %u RESULT: %s (%s)", 
             g_check_board,
             (out->result == RESULT_PASS) ? "PASS" : "FAIL",
             (out->result == RESULT_PASS) ? "OK" : "NG");
    ESP_LOGI(TAG, "  + Số patch bị lỗi : %u / %u patches", out->patch_fail_count, out->patch_total);
    ESP_LOGI(TAG, "  + dmin lớn nhất   : %.6f (Ngưỡng = %.6f)", out->max_dmin, threshold);
    ESP_LOGI(TAG, "  + Thời gian       : %" PRIu32 " ms", elapsed_ms);
    ESP_LOGI(TAG, "==================================================");
}

/* Tính toán Mean & Std cho dòng điện */
static void calc_learn_statistics(void) {
    float sum_imax = 0.0f, sum_iavg = 0.0f;
    for (uint8_t i = 0; i < LEARN_SAMPLES; i++) {
        sum_imax += g_learn_imax[i];
        sum_iavg += g_learn_iavg[i];
    }
    g_imax_mean = sum_imax / (float)LEARN_SAMPLES;
    g_iavg_mean = sum_iavg / (float)LEARN_SAMPLES;

    float var_imax = 0.0f, var_iavg = 0.0f;
    for (uint8_t i = 0; i < LEARN_SAMPLES; i++) {
        var_imax += (g_learn_imax[i] - g_imax_mean) * (g_learn_imax[i] - g_imax_mean);
        var_iavg += (g_learn_iavg[i] - g_iavg_mean) * (g_learn_iavg[i] - g_iavg_mean);
    }
    g_imax_std = sqrtf(var_imax / (float)(LEARN_SAMPLES - 1));
    g_iavg_std = sqrtf(var_iavg / (float)(LEARN_SAMPLES - 1));

    float tol_imax = g_imax_mean * CURRENT_TOLERANCE_PCT;
    float tol_iavg = g_iavg_mean * CURRENT_TOLERANCE_PCT;

    g_imax_max_dev = (CURRENT_SIGMA_FACTOR * g_imax_std) + tol_imax;
    g_iavg_max_dev = (CURRENT_SIGMA_FACTOR * g_iavg_std) + tol_iavg;
}

/* Thực thi kiểm tra dòng điện */
static void perform_real_power_check(float imax, float iavg_or_imin, power_check_result_t *out) {
    int64_t t_start_us = esp_timer_get_time();

    memset(out, 0, sizeof(*out));
    out->imax_measured = imax;
    out->iavg_measured = iavg_or_imin;
    out->imax_threshold = g_imax_max_dev;
    out->iavg_threshold = g_iavg_max_dev;

    float dev_imax = fabsf(imax - g_imax_mean);
    float dev_iavg = fabsf(iavg_or_imin - g_iavg_mean);

    bool pass_imax = (dev_imax <= g_imax_max_dev);
    bool pass_iavg = (dev_iavg <= g_iavg_max_dev);

    out->result = (pass_imax && pass_iavg) ? RESULT_PASS : RESULT_FAIL;
    out->elapsed_ms = (uint32_t)((esp_timer_get_time() - t_start_us) / 1000);
}

/* State Machine xử lý Frame từ STM32 */
static void handle_frame(const rx_frame_t *f) {
    ESP_LOGI(TAG, "RX <- STM32: CMD=0x%02X LEN=%u", f->cmd, f->len);

    switch (f->cmd) {
    case CMD_LEARN_START:
        clear_learn_storage();
        g_learn_state = LEARN_STATE_WAIT_VISION;
        g_check_state = CHECK_STATE_IDLE;
        g_check_board = 1;
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;

    case CMD_CAPTURE_SAMPLE: {
        if (f->len != 1) {
            app_protocol_send_error(ERR_UNKNOWN, "CAPTURE_SAMPLE LEN != 1");
            break;
        }

        uint8_t idx = f->data[0];
        if (g_learn_state != LEARN_STATE_WAIT_VISION || idx != g_expected_vision) {
            app_protocol_send_error(ERR_UNKNOWN, "CAPTURE sai thứ tự");
            break;
        }

        float *features = NULL;
        uint32_t feature_count = 0;
        uint32_t elapsed_ms = 0;
        esp_err_t err = capture_real_feature_block(&features, &feature_count, &elapsed_ms);
        if (err != ESP_OK) {
            app_protocol_send_error(ERR_CAMERA_FAIL, "Chụp ảnh thất bại");
            break;
        }

        g_train_features[idx - 1U] = features;
        g_train_feature_count[idx - 1U] = feature_count;
        g_vision_count = idx;

        uint8_t ack[7];
        ack[0] = idx;
        ack[1] = RESULT_PASS;
        ack[2] = g_vision_count;
        pack_u16_le(&ack[3], clamp_u16(feature_count));
        pack_u16_le(&ack[5], clamp_u16(elapsed_ms));
        app_protocol_send_frame(CMD_CAPTURE_ACK, ack, sizeof(ack));

        if (g_vision_count == LEARN_SAMPLES) {
            g_expected_power = 1;
            g_learn_state = LEARN_STATE_WAIT_POWER;
        } else {
            g_expected_vision++;
        }
        break;
    }

    case CMD_POWER_SAMPLE: {
        if (f->len != 9) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_SAMPLE LEN != 9");
            break;
        }

        uint8_t idx = f->data[0];
        float imax = unpack_float_le(&f->data[1]);
        float iavg = unpack_float_le(&f->data[5]);

        if (g_learn_state != LEARN_STATE_WAIT_POWER || idx != g_expected_power) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_SAMPLE sai thứ tự");
            break;
        }

        g_learn_imax[idx - 1] = imax;
        g_learn_iavg[idx - 1] = iavg;
        g_power_count = idx;
        g_expected_power++;

        uint8_t ack[3] = { idx, RESULT_PASS, g_power_count };
        if (g_power_count == LEARN_SAMPLES) {
            g_learn_state = LEARN_STATE_READY_TRAIN;
            ESP_LOGI(TAG, "Thu đủ 5 ảnh + 5 dòng -> Sẵn sàng TRAIN (CMD 0x13)!");
        }

        app_protocol_send_frame(CMD_POWER_ACK, ack, sizeof(ack));
        break;
    }

    case CMD_TRAIN_START: { /* Lệnh 0x13 từ STM32 */
        if (g_learn_state != LEARN_STATE_READY_TRAIN) {
            app_protocol_send_error(ERR_TRAIN_FAIL, "Chưa đủ 5 ảnh + 5 mẫu dòng");
            break;
        }

        int64_t t_train_start_us = esp_timer_get_time();

        app_ai_coreset_block_t blocks[LEARN_SAMPLES] = {0};
        for (uint32_t i = 0; i < LEARN_SAMPLES; ++i) {
            blocks[i].vectors = g_train_features[i];
            blocks[i].count = g_train_feature_count[i];
            blocks[i].source_image = i;
        }

        // 1. TÍNH TOÁN CORESET MEMORY BANK
        if (app_ai_coreset_build(blocks, LEARN_SAMPLES) != ESP_OK) {
            app_protocol_send_error(ERR_TRAIN_FAIL, "Build Coreset thất bại");
            break;
        }

        // 2. TÍNH TOÁN THỐNG KÊ DÒNG ĐIỆN (MEAN & STD)
        calc_learn_statistics();

        g_model_trained = true;
        g_learn_state = LEARN_STATE_TRAINED;

        uint32_t bank_count = app_ai_coreset_get_bank_count();
        float image_threshold = app_ai_coreset_get_threshold();
        uint32_t train_elapsed_ms = (uint32_t)((esp_timer_get_time() - t_train_start_us) / 1000);

        // 3. IN LOG CONFIRM TẠI ESP32 XÁC NHẬN ĐÃ TÍNH XONG 100%
        ESP_LOGI(TAG, "==================================================");
        ESP_LOGI(TAG, " [ESP32] TÍNH TOÁN HOÀN TẤT - PHÁT FRAME 0x93 SANG STM32");
        ESP_LOGI(TAG, "  + Số patch 5 ảnh mẫu : P1=%u, P2=%u, P3=%u, P4=%u, P5=%u",
                (unsigned)g_train_feature_count[0], (unsigned)g_train_feature_count[1],
                (unsigned)g_train_feature_count[2], (unsigned)g_train_feature_count[3],
                (unsigned)g_train_feature_count[4]);
        ESP_LOGI(TAG, "  + Số phần tử MMB    : %" PRIu32 " vectors", bank_count);
        ESP_LOGI(TAG, "  + Ngưỡng lọc ảnh     : %.6f", image_threshold);
        ESP_LOGI(TAG, "  + Ngưỡng Imax dev    : +-%.3f mA", g_imax_max_dev);
        ESP_LOGI(TAG, "  + Ngưỡng Iavg dev    : +-%.3f mA", g_iavg_max_dev);
        ESP_LOGI(TAG, "  + Thời gian tính toán: %" PRIu32 " ms", train_elapsed_ms);
        ESP_LOGI(TAG, "==================================================");

        // 4. ĐÓNG GÓI VÀ GỬI 26 BYTES CHUẨN SANG STM32
        uint8_t data[26];
        pack_u16_le(&data[0],  clamp_u16(g_train_feature_count[0]));
        pack_u16_le(&data[2],  clamp_u16(g_train_feature_count[1]));
        pack_u16_le(&data[4],  clamp_u16(g_train_feature_count[2]));
        pack_u16_le(&data[6],  clamp_u16(g_train_feature_count[3]));
        pack_u16_le(&data[8],  clamp_u16(g_train_feature_count[4]));
        pack_float_le(&data[10], image_threshold);
        pack_u16_le(&data[14], clamp_u16(bank_count));
        pack_float_le(&data[16], g_imax_max_dev);
        pack_float_le(&data[20], g_iavg_max_dev);
        pack_u16_le(&data[24], clamp_u16(train_elapsed_ms));

        app_protocol_send_frame(CMD_TRAIN_DONE, data, sizeof(data));
        break;
    }

    case CMD_PING:
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;

    case CMD_CHECK_START:
        if (!g_model_trained) {
            app_protocol_send_error(ERR_TRAIN_FAIL, "Chưa TRAIN mô hình");
            break;
        }
        g_check_state = CHECK_STATE_WAIT_VISUAL;
        g_check_board = 1;
        g_check_board_start_us = esp_timer_get_time();
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;

    case CMD_VISUAL_CHECK: { /* Lệnh 0x21 từ STM32 */
        if (f->len != 1 || g_check_state != CHECK_STATE_WAIT_VISUAL || f->data[0] != g_check_board) {
            app_protocol_send_error(ERR_UNKNOWN, "VISUAL_CHECK không hợp lệ");
            break;
        }

        visual_check_result_t vres;
        perform_real_visual_check(&vres);
        if (vres.result == 0xFF) break;

        /* Đóng gói 11 bytes dữ liệu trả về cho STM32 qua CMD_VISUAL_RESULT (0xA1) */
        uint8_t data[11];
        data[0] = vres.result;
        pack_u16_le(&data[1], vres.patch_fail_count);
        pack_u16_le(&data[3], vres.patch_total);
        pack_u16_le(&data[5], clamp_u16(vres.elapsed_ms));
        pack_float_le(&data[7], vres.max_dmin); /* Bổ sung max dmin */

        app_protocol_send_frame(CMD_VISUAL_RESULT, data, sizeof(data));

        g_check_state = (vres.result == RESULT_PASS) ? CHECK_STATE_WAIT_POWER : CHECK_STATE_WAIT_NEXT;
        break;
    }

    case CMD_POWER_CHECK: {
        if (f->len != 9 || g_check_state != CHECK_STATE_WAIT_POWER || f->data[0] != g_check_board) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_CHECK không hợp lệ");
            break;
        }

        float imax = unpack_float_le(&f->data[1]);
        float iavg_or_imin = unpack_float_le(&f->data[5]);

        power_check_result_t pres;
        perform_real_power_check(imax, iavg_or_imin, &pres);

        uint8_t data[19];
        data[0] = pres.result;
        pack_float_le(&data[1], pres.imax_measured);
        pack_float_le(&data[5], pres.imax_threshold);
        pack_float_le(&data[9], pres.iavg_measured);
        pack_float_le(&data[13], pres.iavg_threshold);
        pack_u16_le(&data[17], clamp_u16(pres.elapsed_ms));
        app_protocol_send_frame(CMD_POWER_RESULT, data, sizeof(data));

        g_check_state = CHECK_STATE_WAIT_NEXT;
        break;
    }

    case CMD_CHECK_NEXT:
        if (f->len != 1 || g_check_state != CHECK_STATE_WAIT_NEXT || f->data[0] != g_check_board) {
            app_protocol_send_error(ERR_UNKNOWN, "CHECK_NEXT không hợp lệ");
            break;
        }

        g_check_board++;
        g_check_state = CHECK_STATE_WAIT_VISUAL;
        g_check_board_start_us = esp_timer_get_time();
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;

    default:
        app_protocol_send_error(ERR_UNKNOWN, "CMD không xác định");
        break;
    }
}

static void command_worker_task(void *arg) {
    rx_frame_t frame;
    while (1) {
        if (xQueueReceive(g_frame_queue, &frame, portMAX_DELAY) == pdTRUE) {
            handle_frame(&frame);
        }
    }
}

esp_err_t vedc_controller_start(void) {
    /* Khởi tạo phần cứng UART1 và nhận Queue từ Protocol Module */
    esp_err_t err = app_protocol_init(&g_frame_queue);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Lỗi khởi tạo Protocol UART1: %s", esp_err_to_name(err));
        return err;
    }

    clear_learn_storage();

    /* Tạo Worker Task xử lý logic */
    xTaskCreatePinnedToCore(command_worker_task, "command_worker", 12288, NULL, 9, NULL, 0);

    ESP_LOGI(TAG, "VEDC Controller sẵn sàng!");
    return ESP_OK;
}