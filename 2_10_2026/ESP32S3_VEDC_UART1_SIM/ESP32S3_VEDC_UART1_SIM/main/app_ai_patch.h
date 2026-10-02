/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model input size (bạn của bạn gửi model nhận ảnh 40x40).
#define APP_AI_PATCH_SIZE       40
// Output pixel format is RGB888 (3 bytes/pixel), matches most CNN inputs.
#define APP_AI_PATCH_CHANNELS   3
// AI crop is pixel-aligned to the model patch size. The grid is derived
// straight from whatever resolution the camera actually delivered (see
// compute_fixed_crop() in app_ai_patch.c): width/height are floored to the
// nearest multiple of APP_AI_PATCH_SIZE, so the patch count scales with
// resolution instead of being pinned to one fixed number - e.g. a 640x480
// capture yields 16x12 = 192 patches, an 800x600 capture yields 20x15 =
// 300 patches.
//
// APP_AI_PATCH_MAX_COLS/ROWS below are ONLY a safety ceiling so the patch
// pool (and everything sized off it: feature-vector storage, JSON buffers)
// stays a fixed, predictable allocation instead of growing unbounded. If a
// captured frame is larger than this ceiling in either dimension, that
// dimension is center-cropped down to it - the grid for that capture is
// then smaller than "full resolution / 40", but still internally
// consistent (crop image, patches and vectors always agree).
//
// Deliberately kept at 20x15 = 800x600 (not the camera's full HD ceiling):
// this project also streams USB mic/speaker audio and a live MJPEG preview
// alongside the AI pipeline, and PSRAM is shared with all of that. Going
// higher (e.g. 1280x720 = 576 patches) was tried and reliably starved the
// transient full-frame JPEG-decode buffer of PSRAM ("allocate NNN byte
// decode buffer failed") on a real N16R8 board, because it roughly tripled
// this pool's permanent footprint. 20x15 = 300 patches (~1.44 MB pool, only
// ~520 KB more than the original 192-patch/16x12 pool) is a safe increase
// that still exactly covers the 800x600 case. Raise this further only after
// re-checking free PSRAM headroom (heap_caps_get_free_size(MALLOC_CAP_SPIRAM))
// on real hardware with the camera/mic/speaker all active at once.
#define APP_AI_PATCH_MAX_COLS   20
#define APP_AI_PATCH_MAX_ROWS   15
#define APP_AI_PATCH_MAX_COUNT  (APP_AI_PATCH_MAX_COLS * APP_AI_PATCH_MAX_ROWS)
#define APP_AI_PATCH_MAX_CROP_W (APP_AI_PATCH_MAX_COLS * APP_AI_PATCH_SIZE)
#define APP_AI_PATCH_MAX_CROP_H (APP_AI_PATCH_MAX_ROWS * APP_AI_PATCH_SIZE)

// Allocates the (PSRAM) patch pool. Call once from app_main() before the
// button/web handlers can trigger a capture.
esp_err_t app_ai_patch_init(void);

// Legacy synchronization API retained for compatibility with the web UI.
// The firmware no longer accepts an arbitrary crop rectangle: the actual crop
// is derived from the captured frame resolution and aligned to 40x40 pixels.
// The supplied values are therefore only a compatibility hint and are not
// allowed to change the fixed AI crop geometry.
esp_err_t app_ai_patch_set_crop_percent(float left, float top, float width, float height);

// Grabs one frame directly from the UVC driver (independent of whatever is
// being streamed to the browser), decodes the JPEG on-device, cuts the
// configured crop rectangle into a grid of tiles and nearest-neighbour
// resizes each tile to exactly APP_AI_PATCH_SIZE x APP_AI_PATCH_SIZE so the
// result is ready to feed into the 40x40 model later. Does NOT run any
// model inference - patches are only produced and stored.
// Blocking; safe to call from a normal task context (not an ISR).
esp_err_t app_ai_patch_capture(void);

// Same as app_ai_patch_capture(), but runs in its own short-lived task so
// the caller (e.g. the physical-button debounce loop) is not blocked while
// the JPEG decode/patch-cut runs.
esp_err_t app_ai_patch_capture_async(void);

// Re-splits an already captured, fixed/aligned crop RGB888 image into the
// same 40x40 grid used by app_ai_patch_capture(). This is used by the
// multi-photo session:
// P4 stores each crop, then P3 processes those stored crops in order.
esp_err_t app_ai_patch_process_crop_rgb(const uint8_t *crop_rgb, uint32_t crop_w, uint32_t crop_h);

// Gets the current ESP32-side crop percentages so a multi-photo session can
// remember the crop that belonged to each P4 capture.
esp_err_t app_ai_patch_get_crop_percent(float *left, float *top, float *width, float *height);

// Number of valid patches produced by the last successful capture or crop-processing pass.
uint32_t app_ai_patch_get_count(void);

// Grid layout of the last successful capture or crop-processing pass.
void app_ai_patch_get_grid(uint32_t *cols, uint32_t *rows);

// Raw pointer to patch `index` (row-major, index = row * cols + col).
// Each patch is APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS
// bytes of packed RGB888, ready to hand to a model later. Returns NULL if
// index is out of range or no capture has succeeded yet.
const uint8_t *app_ai_patch_get_data(uint32_t index, size_t *len);

// Pointer to the exact crop-rectangle RGB888 image (post on-device crop,
// *before* the 40x40 grid split) produced by the last successful capture.
// This is the SAME decoded frame the patches above were cut from, so
// serving it back to the web UI as the "captured photo" guarantees the
// photo the user sees and the patches cut from it always match - there is
// no separate/independent frame grab for the preview photo. *w/*h are
// filled in with the crop's pixel size. Returns NULL (and leaves *w/*h/
// *len untouched) if no successful capture has produced one yet.
const uint8_t *app_ai_patch_get_crop_image(uint32_t *w, uint32_t *h, size_t *len);

// Builds a small JSON status object describing the last capture (or lack
// thereof) into buf. Used by the web UI to show capture results.
esp_err_t app_ai_patch_get_status_json(char *buf, size_t len);
void app_ai_patch_debug_dump(uint32_t index);

#ifdef __cplusplus
}
#endif
