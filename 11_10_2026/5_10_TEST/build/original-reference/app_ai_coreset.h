/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "app_ai_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define APP_AI_CORESET_MAX_BANK 64U
#define APP_AI_CORESET_MIN_BANK 32U
#define APP_AI_CORESET_RATIO 0.25f
#define APP_AI_CORESET_EPSILON_ALPHA 0.20f
#define APP_AI_CORESET_EPSILON_MIN 0.01f
#define APP_AI_CORESET_THRESHOLD_MARGIN 1.10f
#define APP_AI_CORESET_PREVIEW_DIMS 8U

typedef struct {
    const float *vectors;     /* count * APP_AI_INFER_FEATURE_DIM floats */
    uint32_t count;
    uint32_t source_image;    /* zero-based image index */
} app_ai_coreset_block_t;

esp_err_t app_ai_coreset_init(void);
void app_ai_coreset_reset(void);

esp_err_t app_ai_coreset_build(const app_ai_coreset_block_t *blocks,
                               uint32_t block_count);

bool app_ai_coreset_is_ready(void);
uint32_t app_ai_coreset_get_bank_count(void);
float app_ai_coreset_get_threshold(void);

/* Tăng tốc tính khoảng cách nhỏ nhất (Anomaly Score) của 1 vector 32-D tới Memory Bank dùng ESP-DSP SIMD */
float app_ai_coreset_compute_score(const float *vector);

esp_err_t app_ai_coreset_get_status_json(char *buf, size_t len);
esp_err_t app_ai_coreset_get_bank_json(char *buf, size_t len);

esp_err_t app_ai_coreset_get_vector(uint32_t index,
                                    float *out,
                                    size_t out_len,
                                    uint32_t *source_image,
                                    uint32_t *source_patch,
                                    float *selection_distance);

#ifdef __cplusplus
}
#endif