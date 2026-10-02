/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// The user can keep pressing P4 until P3. The practical limit is PSRAM, with
// this safety cap preventing the session from consuming the entire module.
#define APP_AI_SESSION_MAX_IMAGES 64
#define APP_AI_SESSION_MAX_RGB_BYTES (5 * 1024 * 1024)

// Start/append one physical P4 capture. The first P4 after IDLE/FINISHED
// automatically starts a fresh session; following P4 presses append images.
esp_err_t app_ai_session_capture_async(void);

// End the capture phase (P3) and, in a background task, process every stored
// crop in capture order: 40x40 patch grid -> TFLite model -> 32-d vectors ->
// Greedy Farthest-Point Coreset -> compact Memory Bank.
esp_err_t app_ai_session_finish_async(void);

bool app_ai_session_is_collecting(void);
bool app_ai_session_is_busy(void);
uint32_t app_ai_session_get_image_count(void);
uint32_t app_ai_session_get_processed_count(void);

// JSON summary for /api/ai_session/status.
esp_err_t app_ai_session_get_status_json(char *buf, size_t len);

// JSON containing every feature vector for one 0-based image index.
esp_err_t app_ai_session_get_image_features_json(uint32_t image_index, char *buf, size_t len);

// Returns the stored crop RGB888 buffer for an image. Pointer remains valid
// until the next session reset; callers must not free it.
const uint8_t *app_ai_session_get_image_crop(uint32_t image_index,
                                             uint32_t *w, uint32_t *h,
                                             size_t *len);

#ifdef __cplusplus
}
#endif
