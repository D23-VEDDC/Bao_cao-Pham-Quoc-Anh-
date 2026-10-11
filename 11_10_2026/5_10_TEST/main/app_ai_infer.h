/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"
#include "app_ai_patch.h"

#ifdef __cplusplus
extern "C" {
#endif

// Length of the embedding vector the teammate's model produces per patch
// (feature_extractor_32d_full_int8.tflite: input [1,40,40,3] int8, output
// [1,32] int8). If a future model version changes this, update it here -
// everything downstream (status JSON, HTTP endpoint, web UI table) reads
// this constant instead of a hard-coded 32.
#define APP_AI_INFER_FEATURE_DIM 32

// Loads the model embedded into the firmware image (see
// main/model/feature_extractor_32d_full_int8.tflite + the EMBED_FILES entry
// in main/CMakeLists.txt), builds the TFLite-Micro interpreter and
// allocates its tensor arena (PSRAM). Call once from app_main(), any time
// after app_ai_patch_init() (this module only reads patches that module
// already produced - it doesn't touch the camera/JPEG pipeline itself).
// Not fatal if it fails (bad/incompatible model, out of memory): logs a
// warning and app_ai_infer_is_ready() stays false, so the rest of the
// console (camera, crop, patch-cut) keeps working with feature extraction
// simply unavailable.
esp_err_t app_ai_infer_init(void);

// True once app_ai_infer_init() has loaded the model, allocated tensors and
// confirmed the input/output tensors match what this pipeline expects
// (1x40x40x3 int8 in, 1xAPP_AI_INFER_FEATURE_DIM int8 out).
bool app_ai_infer_is_ready(void);

// Runs the model once per patch produced by the last app_ai_patch_capture()
// call (app_ai_patch_get_count() / app_ai_patch_get_data()), strictly in
// order (index 0, 1, 2, ...) - i.e. exactly "feed the cropped patches into
// the model one by one" - and stores each patch's dequantized
// APP_AI_INFER_FEATURE_DIM-float feature vector internally. Call this right
// after a successful app_ai_patch_capture(); it does not capture or re-crop
// anything itself.
// Blocking (runs all patches before returning); safe to call from a normal
// task context. Returns ESP_ERR_INVALID_STATE if the model isn't ready or
// no capture has produced any patches yet.
esp_err_t app_ai_infer_run_all(void);

// Number of feature vectors available from the last app_ai_infer_run_all()
// call.
uint32_t app_ai_infer_get_count(void);

// Copies the APP_AI_INFER_FEATURE_DIM-float feature vector for patch
// `index` into out (caller-owned buffer of at least out_len floats, needs
// out_len >= APP_AI_INFER_FEATURE_DIM). Returns ESP_ERR_INVALID_ARG /
// ESP_ERR_NOT_FOUND if index/out_len are invalid or no vector is available.
esp_err_t app_ai_infer_get_vector(uint32_t index, float *out, size_t out_len);

// Builds a JSON status object describing the last app_ai_infer_run_all()
// run into buf: readiness, feature dimension, per-run timing, and - when
// include_vectors is true - a "vectors" array with every patch's feature
// vector (used by the web UI's feature-extraction panel).
esp_err_t app_ai_infer_get_status_json(char *buf, size_t len, bool include_vectors);

#ifdef __cplusplus
}
#endif
