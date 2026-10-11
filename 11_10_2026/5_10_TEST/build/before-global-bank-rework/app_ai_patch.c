/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * On-device (ESP32-S3) pipeline that turns whatever the camera currently
 * sees into a grid of APP_AI_PATCH_SIZE x APP_AI_PATCH_SIZE (40x40) RGB888
 * patches, ready to be fed one-by-one into the teammate's AI model later.
 *
 * This module owns the camera->JPEG->crop->40x40 patch pipeline. Model inference
 * remains in app_ai_infer.cpp so the multi-photo session can capture first and
 * run the model later, after P3 ends the capture phase.
 *
 * Pipeline for one capture:
 *   1. Grab one MJPEG frame straight from the UVC driver (app_uvc_get_frame),
 *      independent of the live browser preview.
 *   2. Decode the JPEG on-device with the espressif/esp_jpeg component
 *      (TJpgDec) into a full-frame RGB888 buffer in PSRAM.
 *   3. Derive a FIXED pixel-aligned crop from the camera resolution: width
 *      and height are multiples of 40, centered if the source resolution is
 *      not divisible by 40, and capped at the APP_AI_PATCH_MAX_COLS x
 *      APP_AI_PATCH_MAX_ROWS safety ceiling (see app_ai_patch.h). Below
 *      that ceiling the crop is the whole frame, so the patch count scales
 *      with resolution: 640x480 -> 16x12 = 192 patches, 800x600 -> 20x15 =
 *      300 patches, and so on - it is no longer pinned to one fixed number.
 *   4. Split that SAME crop into exact 40x40 source cells. No arbitrary
 *      percentage crop and no variable resizing is used anymore.
 *
 * Steps 3 and 4 both read from the one frame decoded in step 2, so the
 * "captured photo" and the 40x40 patches always come from the exact same
 * on-device capture - unlike an earlier version of this pipeline, where the
 * browser took its own canvas snapshot of the *live* preview stream while
 * this code independently grabbed a different UVC frame, so the "photo"
 * and the patches could silently drift apart if anything in the scene
 * moved between the two.
 */

#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "app_ai_patch.h"
#include "app_uvc_manager.h"
#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "jpeg_decoder.h"
#include <stdlib.h>

// Sanity cap on the decoded (full-frame) RGB888 buffer so a mis-negotiated
// huge resolution can't blow the PSRAM budget: ~4 MB covers well beyond any
// resolution the Logitech C270 actually offers over this USB path.
#define APP_AI_PATCH_MAX_DECODE_BYTES (4 * 1024 * 1024)

// How long to wait for a fresh UVC frame before giving up.
#define APP_AI_PATCH_FRAME_WAIT_MS (1500)

// app_uvc_get_frame() dequeues from ONE shared FreeRTOS queue that the
// another camera consumer may be continuously draining. When both tasks are
// blocked waiting on that queue, FreeRTOS always wakes the higher-priority
// one first - so as long as a browser tab has the camera page open, it wins
// every single frame and this capture starves forever.
// Fix: briefly raise this task's own priority above the streaming task's
// (5) only while waiting for the frame, then restore it right away. This
// must stay higher than the camera consumer's priority.
#define APP_AI_PATCH_FRAME_GRAB_PRIORITY (6)

static const char *TAG = "app_ai_patch";

typedef struct {
    float left;
    float top;
    float width;
    float height;
} app_ai_patch_crop_t;

static struct {
    SemaphoreHandle_t lock;
    uint8_t *pool;                 // APP_AI_PATCH_MAX_COUNT patches, back to back
    app_ai_patch_crop_t crop;      // percent, same convention as the web UI
    uint32_t cols;
    uint32_t rows;
    uint32_t count;                // cols * rows of the last successful capture
    uint32_t source_w;
    uint32_t source_h;
    int crop_x_px;
    int crop_y_px;
    uint8_t *registration_ref;
    uint8_t *registration_mask;
    uint32_t registration_w;
    uint32_t registration_h;
    uint32_t registration_textured;
    int64_t last_capture_us;
    bool last_capture_ok;
    char last_error[48];
    // Exact copy of the crop rectangle itself (pre-grid-split), kept around
    // so the web UI can be served the *actual captured photo* instead of a
    // separately/live-grabbed browser canvas snapshot. Grown with realloc()
    // as needed since crop pixel size varies with the crop box and camera
    // resolution; capacity is tracked separately so a smaller later capture
    // doesn't force a reallocation.
    uint8_t *crop_image;
    size_t crop_image_cap;
    uint32_t crop_image_w;
    uint32_t crop_image_h;
    uint8_t *last_jpeg;
    size_t last_jpeg_len;
    uint32_t last_jpeg_capture_id;
    bool last_jpeg_pending;
} s_ai = {
    .crop = { .left = 0.0f, .top = 0.0f, .width = 100.0f, .height = 100.0f },
};

#define REG_STEP 8
#define REG_MAX_SHIFT 20

static uint8_t rgb_luma(const uint8_t *p)
{
    return (uint8_t)(((uint32_t)p[0] * 77 + (uint32_t)p[1] * 150 +
                      (uint32_t)p[2] * 29) >> 8);
}

void app_ai_patch_reset_registration_reference(void)
{
    if (!s_ai.lock) return;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    heap_caps_free(s_ai.registration_ref);
    heap_caps_free(s_ai.registration_mask);
    s_ai.registration_ref = NULL;
    s_ai.registration_mask = NULL;
    s_ai.registration_w = s_ai.registration_h = 0;
    s_ai.registration_textured = 0;
    xSemaphoreGive(s_ai.lock);
}

esp_err_t app_ai_patch_set_registration_reference(void)
{
    if (!s_ai.lock) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    const uint32_t w = s_ai.cols * APP_AI_PATCH_SIZE / REG_STEP;
    const uint32_t h = s_ai.rows * APP_AI_PATCH_SIZE / REG_STEP;
    if (!s_ai.last_capture_ok || !w || !h) {
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_INVALID_STATE;
    }
    const size_t n = (size_t)w * h;
    uint8_t *ref = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *mask = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ref) ref = heap_caps_malloc(n, MALLOC_CAP_8BIT);
    if (!mask) mask = heap_caps_malloc(n, MALLOC_CAP_8BIT);
    if (!ref || !mask) {
        heap_caps_free(ref);
        heap_caps_free(mask);
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_NO_MEM;
    }
    uint32_t textured = 0;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const uint32_t px = x * REG_STEP + REG_STEP / 2;
            const uint32_t py = y * REG_STEP + REG_STEP / 2;
            const uint32_t patch = (py / APP_AI_PATCH_SIZE) * s_ai.cols + px / APP_AI_PATCH_SIZE;
            const uint32_t lx = px % APP_AI_PATCH_SIZE;
            const uint32_t ly = py % APP_AI_PATCH_SIZE;
            const uint8_t *p = s_ai.pool + (size_t)patch * APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * 3 +
                               ((size_t)ly * APP_AI_PATCH_SIZE + lx) * 3;
            ref[y * w + x] = rgb_luma(p);
            // Ignore flat areas and the outer edge. Foreign objects then have
            // little influence on translation while board traces anchor it.
            const bool inside = px >= REG_MAX_SHIFT + 2 && py >= REG_MAX_SHIFT + 2 &&
                px + REG_MAX_SHIFT + 2 < s_ai.cols * APP_AI_PATCH_SIZE &&
                py + REG_MAX_SHIFT + 2 < s_ai.rows * APP_AI_PATCH_SIZE;
            const uint8_t *qx = p + 6;
            const uint8_t *qy = p + APP_AI_PATCH_SIZE * 2 * 3;
            mask[y * w + x] = inside &&
                (abs((int)rgb_luma(p) - (int)rgb_luma(qx)) >= 7 ||
                 abs((int)rgb_luma(p) - (int)rgb_luma(qy)) >= 7);
            textured += mask[y * w + x] != 0;
        }
    }
    heap_caps_free(s_ai.registration_ref);
    heap_caps_free(s_ai.registration_mask);
    s_ai.registration_ref = ref;
    s_ai.registration_mask = mask;
    s_ai.registration_w = w;
    s_ai.registration_h = h;
    s_ai.registration_textured = textured;
    xSemaphoreGive(s_ai.lock);
    ESP_LOGI(TAG, "Registration reference: %ux%u, textured=%u", (unsigned)w, (unsigned)h, (unsigned)textured);
    if (textured < 40) {
        ESP_LOGW(TAG, "Reference lacks trace detail for reliable translation; keeping zero offset");
    }
    return ESP_OK;
}

static uint32_t registration_cost(const uint8_t *rgb, int frame_w, int frame_h,
                                  int center_x, int center_y, int dx, int dy,
                                  uint32_t *used)
{
    uint32_t sum = 0;
    uint32_t n = 0;
    for (uint32_t y = 0; y < s_ai.registration_h; ++y) {
        const int py = center_y + (int)y * REG_STEP + REG_STEP / 2 + dy;
        if (py < 0 || py >= frame_h) continue;
        for (uint32_t x = 0; x < s_ai.registration_w; ++x) {
            const size_t i = (size_t)y * s_ai.registration_w + x;
            if (!s_ai.registration_mask[i]) continue;
            const int px = center_x + (int)x * REG_STEP + REG_STEP / 2 + dx;
            if (px < 0 || px >= frame_w) continue;
            const uint8_t *p = rgb + ((size_t)py * frame_w + px) * 3;
            int d = abs((int)rgb_luma(p) - (int)s_ai.registration_ref[i]);
            sum += (uint32_t)(d > 40 ? 40 : d);
            ++n;
        }
    }
    if (used) *used = n;
    return n ? (uint32_t)((uint64_t)sum * 256 / n) : UINT32_MAX;
}

static void find_registration(const uint8_t *rgb, int frame_w, int frame_h,
                              int center_x, int center_y, int *dx, int *dy)
{
    *dx = *dy = 0;
    if (!s_ai.registration_ref || !s_ai.registration_mask || s_ai.registration_textured < 40) return;
    uint32_t used = 0;
    const uint32_t initial = registration_cost(rgb, frame_w, frame_h, center_x, center_y, 0, 0, &used);
    if (used < 40) return;
    uint32_t best = initial;
    int bx = 0, by = 0;
    for (int y = -REG_MAX_SHIFT; y <= REG_MAX_SHIFT; y += 4) {
        for (int x = -REG_MAX_SHIFT; x <= REG_MAX_SHIFT; x += 4) {
            uint32_t cost = registration_cost(rgb, frame_w, frame_h, center_x, center_y, x, y, NULL);
            if (cost < best) { best = cost; bx = x; by = y; }
        }
    }
    for (int y = by - 3; y <= by + 3; ++y) {
        if (y < -REG_MAX_SHIFT || y > REG_MAX_SHIFT) continue;
        for (int x = bx - 3; x <= bx + 3; ++x) {
            if (x < -REG_MAX_SHIFT || x > REG_MAX_SHIFT) continue;
            uint32_t cost = registration_cost(rgb, frame_w, frame_h, center_x, center_y, x, y, NULL);
            if (cost < best) { best = cost; *dx = x; *dy = y; }
        }
    }
    // A tiny apparent improvement is usually JPEG noise, not board motion.
    if (best + 128 >= initial) *dx = *dy = 0;
    ESP_LOGI(TAG, "Registration offset=(%d,%d) cost=%u->%u", *dx, *dy,
             (unsigned)initial, (unsigned)best);
}

void app_ai_patch_get_crop_origin(int *x, int *y)
{
    if (!s_ai.lock) return;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    if (x) *x = s_ai.crop_x_px;
    if (y) *y = s_ai.crop_y_px;
    xSemaphoreGive(s_ai.lock);
}

bool app_ai_patch_is_patch_visible(uint32_t index)
{
    if (!s_ai.lock) return false;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    const int x = s_ai.crop_x_px + (int)(index % (s_ai.cols ? s_ai.cols : 1)) * APP_AI_PATCH_SIZE;
    const int y = s_ai.crop_y_px + (int)(index / (s_ai.cols ? s_ai.cols : 1)) * APP_AI_PATCH_SIZE;
    const bool ok = index < s_ai.count && x >= 0 && y >= 0 &&
        x + APP_AI_PATCH_SIZE <= (int)s_ai.source_w &&
        y + APP_AI_PATCH_SIZE <= (int)s_ai.source_h;
    xSemaphoreGive(s_ai.lock);
    return ok;
}

esp_err_t app_ai_patch_extract_appearance(uint8_t *out, size_t out_len)
{
    if (!s_ai.lock || !out) return ESP_ERR_INVALID_ARG;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    if (!s_ai.last_capture_ok || out_len < (size_t)s_ai.count * APP_AI_PATCH_APPEARANCE_DIM) {
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_INVALID_SIZE;
    }
    for (uint32_t p = 0; p < s_ai.count; ++p) {
        const uint8_t *rgb = s_ai.pool + (size_t)p * APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * 3;
        uint8_t *dst = out + (size_t)p * APP_AI_PATCH_APPEARANCE_DIM;
        for (uint32_t cy = 0; cy < 4; ++cy) {
            for (uint32_t cx = 0; cx < 4; ++cx) {
                uint32_t sum[3] = {0};
                for (uint32_t y = 0; y < 10; ++y) {
                    for (uint32_t x = 0; x < 10; ++x) {
                        const uint8_t *v = rgb + (((size_t)cy * 10 + y) * APP_AI_PATCH_SIZE + cx * 10 + x) * 3;
                        sum[0] += v[0]; sum[1] += v[1]; sum[2] += v[2];
                    }
                }
                for (uint32_t c = 0; c < 3; ++c) dst[(cy * 4 + cx) * 3 + c] = (uint8_t)((sum[c] + 50) / 100);
            }
        }
    }
    xSemaphoreGive(s_ai.lock);
    return ESP_OK;
}

// ---------------------------------------------------------------------
// Fix: many UVC webcams (this Logitech C270 included) send MJPEG frames
// that only contain a DQT (quantization table) marker and NO DHT (Huffman
// table) marker. That is legal MJPEG - the Motion-JPEG convention is that
// the receiver already knows the "standard" JPEG Huffman tables from
// ITU-T.81 Annex K.3 and is expected to supply them itself. Browsers'
// built-in JPEG decoders do this silently, which is why the live preview
// in this project (raw bytes forwarded straight to the browser over the
// WebSocket) always looks fine. The on-device decoder (espressif/esp_jpeg)
// does NOT do this substitution, so every single on-device decode call
// fails with "Error in preparing JPEG image! 6" as soon as it hits a
// table-less frame - which, for this camera, is every frame.
//
// The fix is to inspect each captured frame and, if it has no DHT marker,
// splice the standard tables in (right after the SOI marker, before
// anything else) so esp_jpeg_decode() gets a fully self-contained JPEG.
// ---------------------------------------------------------------------

static const uint8_t s_dht_bits_dc_luma[16]     = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
static const uint8_t s_dht_vals_dc_luma[12]     = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
static const uint8_t s_dht_bits_dc_chroma[16]   = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
static const uint8_t s_dht_vals_dc_chroma[12]   = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
static const uint8_t s_dht_bits_ac_luma[16]     = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
static const uint8_t s_dht_vals_ac_luma[162]    = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};
static const uint8_t s_dht_bits_ac_chroma[16]   = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
static const uint8_t s_dht_vals_ac_chroma[162]  = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

// FFC4 + 2 length bytes + 4 tables, each (1 id-byte + 16 count-bytes + N value-bytes)
#define APP_AI_PATCH_STD_DHT_SIZE \
    (2 + 2 + (1 + 16 + sizeof(s_dht_vals_dc_luma)) + (1 + 16 + sizeof(s_dht_vals_dc_chroma)) + \
     (1 + 16 + sizeof(s_dht_vals_ac_luma)) + (1 + 16 + sizeof(s_dht_vals_ac_chroma)))

static size_t append_dht_table(uint8_t *out, uint8_t class_and_id, const uint8_t *bits, const uint8_t *vals, size_t n_vals)
{
    size_t pos = 0;
    out[pos++] = class_and_id;
    memcpy(out + pos, bits, 16);
    pos += 16;
    memcpy(out + pos, vals, n_vals);
    pos += n_vals;
    return pos;
}

// Builds the standard-Huffman-tables DHT marker segment once and caches it;
// the bytes are constant so there is no need to rebuild them per frame.
static const uint8_t *get_standard_dht_segment(size_t *out_len)
{
    static uint8_t s_segment[APP_AI_PATCH_STD_DHT_SIZE];
    static bool s_ready = false;
    if (!s_ready) {
        size_t pos = 0;
        s_segment[pos++] = 0xFF;
        s_segment[pos++] = 0xC4;
        size_t len_pos = pos;
        pos += 2; // filled in below, once payload length is known
        pos += append_dht_table(s_segment + pos, 0x00, s_dht_bits_dc_luma, s_dht_vals_dc_luma, sizeof(s_dht_vals_dc_luma));
        pos += append_dht_table(s_segment + pos, 0x01, s_dht_bits_dc_chroma, s_dht_vals_dc_chroma, sizeof(s_dht_vals_dc_chroma));
        pos += append_dht_table(s_segment + pos, 0x10, s_dht_bits_ac_luma, s_dht_vals_ac_luma, sizeof(s_dht_vals_ac_luma));
        pos += append_dht_table(s_segment + pos, 0x11, s_dht_bits_ac_chroma, s_dht_vals_ac_chroma, sizeof(s_dht_vals_ac_chroma));
        uint16_t seg_len = (uint16_t)(pos - len_pos); // length field covers itself + everything after it
        s_segment[len_pos] = (uint8_t)(seg_len >> 8);
        s_segment[len_pos + 1] = (uint8_t)(seg_len & 0xFF);
        s_ready = true;
    }
    *out_len = sizeof(s_segment);
    return s_segment;
}

// Scans the marker segments between SOI and SOS looking for an existing
// DHT (0xFFC4) marker. Stops as soon as it reaches the entropy-coded scan
// data (SOS) or runs out of well-formed markers.
static bool jpeg_has_dht(const uint8_t *data, size_t len)
{
    if (len < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false;
    }
    size_t pos = 2;
    while (pos + 1 < len) {
        if (data[pos] != 0xFF) {
            pos++;
            continue;
        }
        uint8_t marker = data[pos + 1];
        if (marker == 0xFF) { // fill byte
            pos++;
            continue;
        }
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) {
            // TEM / RSTn / SOI: no length field
            pos += 2;
            continue;
        }
        if (marker == 0xD9) { // EOI
            return false;
        }
        if (marker == 0xDA) { // SOS - entropy data starts, DHT must appear before this
            return false;
        }
        if (marker == 0xC4) {
            return true;
        }
        if (pos + 4 > len) {
            return false;
        }
        uint16_t seg_len = ((uint16_t)data[pos + 2] << 8) | data[pos + 3];
        if (seg_len < 2) {
            return false; // malformed segment length, give up
        }
        pos += 2 + seg_len;
    }
    return false;
}

// If `data` is missing its DHT marker, returns a freshly-allocated copy
// (SOI + standard DHT segment + the rest of the original bytes) that the
// caller must free. Returns NULL (and leaves *out_len untouched) if the
// frame already has its own tables or the allocation failed - in the
// latter case the caller should just try decoding the original data as-is.
static uint8_t *jpeg_ensure_dht(const uint8_t *data, size_t len, size_t *out_len)
{
    static bool s_logged = false;
    if (jpeg_has_dht(data, len)) {
        return NULL;
    }
    if (!s_logged) {
        ESP_LOGI(TAG, "Camera MJPEG frames have no embedded Huffman table (common for UVC webcams); "
                 "injecting standard JPEG tables before on-device decode");
        s_logged = true;
    }
    if (len < 2) {
        return NULL;
    }
    size_t dht_len = 0;
    const uint8_t *dht = get_standard_dht_segment(&dht_len);
    size_t total = 2 + dht_len + (len - 2);
    uint8_t *buf = heap_caps_malloc(total, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        buf = heap_caps_malloc(total, MALLOC_CAP_8BIT);
    }
    if (!buf) {
        ESP_LOGW(TAG, "Allocate %u byte DHT-fix buffer failed, decoding frame as-is", (unsigned)total);
        return NULL;
    }
    memcpy(buf, data, 2);                 // SOI
    memcpy(buf + 2, dht, dht_len);         // injected standard Huffman tables
    memcpy(buf + 2 + dht_len, data + 2, len - 2); // rest of the original frame
    *out_len = total;
    return buf;
}

static size_t patch_stride_bytes(void)
{
    return (size_t)APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS;
}

esp_err_t app_ai_patch_init(void)
{
    if (s_ai.lock) {
        return ESP_OK; // already initialized
    }
    s_ai.lock = xSemaphoreCreateMutex();
    if (!s_ai.lock) {
        ESP_LOGE(TAG, "Create AI patch mutex failed");
        return ESP_ERR_NO_MEM;
    }

    size_t pool_bytes = (size_t)APP_AI_PATCH_MAX_COUNT * patch_stride_bytes();
    s_ai.pool = heap_caps_malloc(pool_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_ai.pool) {
        ESP_LOGW(TAG, "PSRAM allocation for patch pool failed (%u bytes), trying internal RAM", (unsigned)pool_bytes);
        s_ai.pool = heap_caps_malloc(pool_bytes, MALLOC_CAP_8BIT);
    }
    if (!s_ai.pool) {
        ESP_LOGE(TAG, "Allocate AI patch pool failed (%u bytes)", (unsigned)pool_bytes);
        vSemaphoreDelete(s_ai.lock);
        s_ai.lock = NULL;
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "AI patch pool ready: %ux%u max grid, %d x %d px/patch, %u bytes",
             APP_AI_PATCH_MAX_COLS, APP_AI_PATCH_MAX_ROWS, APP_AI_PATCH_SIZE, APP_AI_PATCH_SIZE,
             (unsigned)pool_bytes);
    return ESP_OK;
}

esp_err_t app_ai_patch_set_crop_percent(float left, float top, float width, float height)
{
    (void)left;
    (void)top;
    (void)width;
    (void)height;
    // Compatibility API: crop geometry is now derived from the actual camera
    // frame resolution and aligned to APP_AI_PATCH_SIZE. We deliberately do
    // not accept arbitrary percentages anymore, because that was the source
    // of non-40-aligned crop sizes and variable patch counts.
    if (!s_ai.lock) {
        return ESP_ERR_INVALID_STATE;
    }
    return ESP_OK;
}

// Derive the only crop geometry used by the AI pipeline. The crop scales
// with whatever resolution the camera actually delivered, bounded only by
// the APP_AI_PATCH_MAX_COLS x APP_AI_PATCH_MAX_ROWS safety ceiling (see
// app_ai_patch.h). If the camera frame is not divisible by 40, the largest
// divisible rectangle is centered in the frame so the discarded border is
// symmetric. Examples below the ceiling: 640x480 -> whole frame, 16x12 =>
// 192 patches; 800x600 -> whole frame, 20x15 => 300 patches.
static bool compute_fixed_crop(int frame_w, int frame_h,
                               int *crop_x, int *crop_y,
                               int *crop_w, int *crop_h,
                               app_ai_patch_crop_t *crop_percent)
{
    if (frame_w < APP_AI_PATCH_SIZE || frame_h < APP_AI_PATCH_SIZE ||
        !crop_x || !crop_y || !crop_w || !crop_h) {
        return false;
    }

    int w = (frame_w / APP_AI_PATCH_SIZE) * APP_AI_PATCH_SIZE;
    int h = (frame_h / APP_AI_PATCH_SIZE) * APP_AI_PATCH_SIZE;

    // Do not allow the crop/pool to exceed the known N16R8-safe 640x480 AI
    // working area. Higher camera modes remain usable for the web preview,
    // but the AI crop stays a deterministic 40-aligned 640x480 center window.
    if (w > APP_AI_PATCH_MAX_CROP_W) w = APP_AI_PATCH_MAX_CROP_W;
    if (h > APP_AI_PATCH_MAX_CROP_H) h = APP_AI_PATCH_MAX_CROP_H;

    *crop_x = (frame_w - w) / 2;
    *crop_y = (frame_h - h) / 2;
    *crop_w = w;
    *crop_h = h;

    if (crop_percent) {
        crop_percent->left = 100.0f * (float)(*crop_x) / (float)frame_w;
        crop_percent->top = 100.0f * (float)(*crop_y) / (float)frame_h;
        crop_percent->width = 100.0f * (float)(*crop_w) / (float)frame_w;
        crop_percent->height = 100.0f * (float)(*crop_h) / (float)frame_h;
    }
    return true;
}

// Nearest-neighbour resize of a `src_w`x`src_h` RGB888 window (top-left at
// src_x0,src_y0 inside a full-frame buffer `src_stride` pixels wide) into a
// fixed APP_AI_PATCH_SIZE x APP_AI_PATCH_SIZE RGB888 patch. Works for both
// down- and up-scaling, so it doesn't matter whether a grid cell is bigger
// or smaller than the model's 40x40 input.
static void resize_cell_to_patch(const uint8_t *src, int src_stride,
                                  int src_x0, int src_y0, int src_w, int src_h,
                                  uint8_t *dst)
{
    if (src_w < 1) {
        src_w = 1;
    }
    if (src_h < 1) {
        src_h = 1;
    }

    // Fast path: pipeline hiện tại luôn cắt ô nguồn đúng bằng APP_AI_PATCH_SIZE
    // x APP_AI_PATCH_SIZE (crop được ép chia hết cho 40 trước khi vào đây),
    // nên gần như mọi lệnh gọi rơi vào trường hợp KHÔNG cần resize - tỉ lệ
    // out/src luôn là 1:1. Bản tổng quát bên dưới vẫn tính "(out*src_h)/SIZE"
    // cho từng pixel dù kết quả luôn bằng out_y/out_x, tốn phép nhân+chia lặp
    // lại 40*40 lần/patch x tới 300 patch/ảnh một cách vô ích. Ở đây thay
    // bằng memcpy nguyên hàng (120 byte/hàng x 40 hàng), rẻ hơn nhiều so với
    // vòng lặp per-pixel.
    if (src_w == APP_AI_PATCH_SIZE && src_h == APP_AI_PATCH_SIZE) {
        const size_t row_bytes = (size_t)APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS;
        for (int y = 0; y < APP_AI_PATCH_SIZE; y++) {
            const uint8_t *src_row = src +
                (size_t)(src_y0 + y) * (size_t)src_stride * APP_AI_PATCH_CHANNELS +
                (size_t)src_x0 * APP_AI_PATCH_CHANNELS;
            uint8_t *dst_row = dst + (size_t)y * row_bytes;
            memcpy(dst_row, src_row, row_bytes);
        }
        return;
    }

    for (int out_y = 0; out_y < APP_AI_PATCH_SIZE; out_y++) {
        int sy = src_y0 + (out_y * src_h) / APP_AI_PATCH_SIZE;
        const uint8_t *src_row = src + (size_t)sy * src_stride * APP_AI_PATCH_CHANNELS;
        uint8_t *dst_row = dst + (size_t)out_y * APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS;
        for (int out_x = 0; out_x < APP_AI_PATCH_SIZE; out_x++) {
            int sx = src_x0 + (out_x * src_w) / APP_AI_PATCH_SIZE;
            const uint8_t *sp = src_row + (size_t)sx * APP_AI_PATCH_CHANNELS;
            uint8_t *dp = dst_row + (size_t)out_x * APP_AI_PATCH_CHANNELS;
            dp[0] = sp[0];
            dp[1] = sp[1];
            dp[2] = sp[2];
        }
    }
}


esp_err_t app_ai_patch_process_crop_rgb(const uint8_t *crop_rgb, uint32_t crop_w, uint32_t crop_h)
{
    if (!s_ai.lock || !s_ai.pool || !crop_rgb) {
        ESP_LOGW(TAG, "AI crop processing skipped: module not initialized or invalid buffer");
        return ESP_ERR_INVALID_STATE;
    }
    if (crop_w == 0 || crop_h == 0) {
        return ESP_ERR_INVALID_SIZE;
    }

    // P3 processes only fixed/aligned crops. Every cell is therefore already
    // exactly 40x40 in source pixels: no proportional re-scaling and no
    // variable patch geometry. This is the invariant expected by the model.
    if ((crop_w % APP_AI_PATCH_SIZE) != 0 || (crop_h % APP_AI_PATCH_SIZE) != 0 ||
        crop_w > APP_AI_PATCH_MAX_CROP_W || crop_h > APP_AI_PATCH_MAX_CROP_H) {
        ESP_LOGE(TAG, "AI crop is not fixed/40-aligned: %ux%u",
                 (unsigned)crop_w, (unsigned)crop_h);
        return ESP_ERR_INVALID_SIZE;
    }

    uint32_t cols = crop_w / APP_AI_PATCH_SIZE;
    uint32_t rows = crop_h / APP_AI_PATCH_SIZE;
    if (cols == 0 || rows == 0 || cols > APP_AI_PATCH_MAX_COLS || rows > APP_AI_PATCH_MAX_ROWS) {
        return ESP_ERR_INVALID_SIZE;
    }

    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    for (uint32_t r = 0; r < rows; r++) {
        int cell_y0 = (int)(r * APP_AI_PATCH_SIZE);
        for (uint32_t c = 0; c < cols; c++) {
            int cell_x0 = (int)(c * APP_AI_PATCH_SIZE);
            uint32_t idx = r * cols + c;
            uint8_t *dst = s_ai.pool + (size_t)idx * patch_stride_bytes();
            // Source cell is exactly 40x40, so this is effectively a direct
            // copy through the common helper (no geometric distortion).
            resize_cell_to_patch(crop_rgb, (int)crop_w, cell_x0, cell_y0,
                                  APP_AI_PATCH_SIZE, APP_AI_PATCH_SIZE, dst);
        }
    }
    s_ai.cols = cols;
    s_ai.rows = rows;
    s_ai.count = cols * rows;
    s_ai.source_w = crop_w;
    s_ai.source_h = crop_h;
    s_ai.last_capture_ok = true;
    s_ai.last_capture_us = esp_timer_get_time();
    s_ai.last_error[0] = '\0';
    xSemaphoreGive(s_ai.lock);

    ESP_LOGI(TAG, "AI patch process crop ok: crop=%ux%u grid=%ux%u (%u patches of %dx%d)",
             (unsigned)crop_w, (unsigned)crop_h, (unsigned)cols, (unsigned)rows,
             (unsigned)(cols * rows), APP_AI_PATCH_SIZE, APP_AI_PATCH_SIZE);
    return ESP_OK;
}

esp_err_t app_ai_patch_get_crop_percent(float *left, float *top, float *width, float *height)
{
    if (!left || !top || !width || !height || !s_ai.lock) {
        return ESP_ERR_INVALID_ARG;
    }
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    *left = s_ai.crop.left;
    *top = s_ai.crop.top;
    *width = s_ai.crop.width;
    *height = s_ai.crop.height;
    xSemaphoreGive(s_ai.lock);
    return ESP_OK;
}

esp_err_t app_ai_patch_capture(void)
{
    if (!s_ai.lock || !s_ai.pool) {
        ESP_LOGW(TAG, "AI patch capture skipped: not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    /* The previous JPEG must not be served with its old capture ID while a
     * new capture is replacing it or before its inference has completed. */
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    s_ai.last_jpeg_capture_id = 0;
    s_ai.last_jpeg_pending = false;
    xSemaphoreGive(s_ai.lock);
    if (!app_uvc_is_streaming() || app_uvc_get_format() != APP_UVC_FORMAT_MJPEG) {
        ESP_LOGW(TAG, "AI patch capture skipped: camera not streaming MJPEG");
        xSemaphoreTake(s_ai.lock, portMAX_DELAY);
        s_ai.last_capture_ok = false;
        snprintf(s_ai.last_error, sizeof(s_ai.last_error), "camera_not_streaming");
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_INVALID_STATE;
    }

    UBaseType_t orig_prio = uxTaskPriorityGet(NULL);
    bool boosted = orig_prio < APP_AI_PATCH_FRAME_GRAB_PRIORITY;
    if (boosted) {
        vTaskPrioritySet(NULL, APP_AI_PATCH_FRAME_GRAB_PRIORITY);
    }
    uvc_host_frame_t *frame = app_uvc_get_frame(pdMS_TO_TICKS(APP_AI_PATCH_FRAME_WAIT_MS));
    if (boosted) {
        vTaskPrioritySet(NULL, orig_prio);
    }
    if (!frame) {
        ESP_LOGW(TAG, "AI patch capture failed: no camera frame available");
        xSemaphoreTake(s_ai.lock, portMAX_DELAY);
        s_ai.last_capture_ok = false;
        snprintf(s_ai.last_error, sizeof(s_ai.last_error), "frame_timeout");
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_TIMEOUT;
    }

    esp_err_t ret = ESP_OK;
    const char *err_code = NULL;
    uint8_t *decoded = NULL;
    uint8_t *dht_fixed = NULL; // only set if the raw frame needed patched Huffman tables

    size_t decode_bytes = (size_t)frame->vs_format.h_res * frame->vs_format.v_res * APP_AI_PATCH_CHANNELS;
    if (decode_bytes == 0 || decode_bytes > APP_AI_PATCH_MAX_DECODE_BYTES) {
        ESP_LOGE(TAG, "AI patch capture failed: unsupported frame size %ux%u",
                 frame->vs_format.h_res, frame->vs_format.v_res);
        ret = ESP_ERR_INVALID_SIZE;
        err_code = "resolution_out_of_range";
        goto done;
    }

    decoded = heap_caps_malloc(decode_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!decoded) {
        decoded = heap_caps_malloc(decode_bytes, MALLOC_CAP_8BIT);
    }
    if (!decoded) {
        ESP_LOGE(TAG, "AI patch capture failed: allocate %u byte decode buffer failed", (unsigned)decode_bytes);
        ret = ESP_ERR_NO_MEM;
        err_code = "no_mem_decode";
        goto done;
    }

    // This camera's MJPEG frames commonly omit the Huffman tables (DHT) -
    // browsers substitute the JPEG standard tables silently, but the
    // on-device decoder needs them spliced in explicitly or it fails with
    // "Error in preparing JPEG image".
    size_t decode_indata_len = frame->data_len;
    uint8_t *decode_indata = frame->data;
    dht_fixed = jpeg_ensure_dht(frame->data, frame->data_len, &decode_indata_len);
    if (dht_fixed) {
        decode_indata = dht_fixed;
    } else {
        decode_indata_len = frame->data_len;
    }

    esp_jpeg_image_cfg_t jpeg_cfg = {
        .indata = decode_indata,
        .indata_size = decode_indata_len,
        .outbuf = decoded,
        .outbuf_size = decode_bytes,
        .out_format = JPEG_IMAGE_FORMAT_RGB888,
        .out_scale = JPEG_IMAGE_SCALE_0,
        .flags = {
            .swap_color_bytes = 0,
        },
    };
    esp_jpeg_image_output_t out_info = {0};
    ret = esp_jpeg_decode(&jpeg_cfg, &out_info);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "AI patch capture failed: JPEG decode error %s", esp_err_to_name(ret));
        err_code = "jpeg_decode_failed";
        goto done;
    }

    // Keep the camera's original JPEG so the remote server receives a real
    // image without an RGB-to-JPEG encode pass on the ESP32.
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    /* Reuse the previous JPEG buffer where possible. Allocating a second
     * full frame before freeing the first can fail under PSRAM pressure. */
    uint8_t *jpeg_copy = heap_caps_realloc(s_ai.last_jpeg, frame->data_len,
                                          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!jpeg_copy) {
        jpeg_copy = heap_caps_realloc(s_ai.last_jpeg, frame->data_len, MALLOC_CAP_8BIT);
    }
    if (jpeg_copy) {
        memcpy(jpeg_copy, frame->data, frame->data_len);
        s_ai.last_jpeg = jpeg_copy;
        s_ai.last_jpeg_len = frame->data_len;
        s_ai.last_jpeg_pending = true;
    } else {
        ESP_LOGW(TAG, "Could not retain JPEG frame for remote upload");
    }
    xSemaphoreGive(s_ai.lock);

    // Frame data is no longer needed once decoded - return it to the UVC
    // driver as early as possible so the camera pipeline keeps flowing.
    esp_err_t ret2 = app_uvc_return_frame(frame);
    frame = NULL;
    if (ret2 != ESP_OK) {
        ESP_LOGW(TAG, "Return UVC frame after AI decode failed: %s", esp_err_to_name(ret2));
    }

    int width = out_info.width;
    int height = out_info.height;
    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
    app_ai_patch_crop_t fixed_crop = {0};
    if (!compute_fixed_crop(width, height, &crop_x, &crop_y, &crop_w, &crop_h, &fixed_crop)) {
        ESP_LOGE(TAG, "AI patch capture failed: cannot form fixed 40-aligned crop from %dx%d", width, height);
        ret = ESP_ERR_INVALID_SIZE;
        err_code = "fixed_crop_invalid";
        goto done;
    }

    if (s_ai.registration_ref) {
        if (s_ai.registration_w != (uint32_t)(crop_w / REG_STEP) ||
            s_ai.registration_h != (uint32_t)(crop_h / REG_STEP)) {
            ret = ESP_ERR_INVALID_SIZE;
            err_code = "registration_grid_changed";
            goto done;
        }
        int dx = 0, dy = 0;
        find_registration(decoded, width, height, crop_x, crop_y, &dx, &dy);
        crop_x += dx;
        crop_y += dy;
    }
    fixed_crop.left = 100.0f * crop_x / width;
    fixed_crop.top = 100.0f * crop_y / height;

    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    s_ai.crop = fixed_crop;
    xSemaphoreGive(s_ai.lock);

    // Keep an exact copy of the crop rectangle itself (before it gets diced
    // into the 40x40 grid below) so it can be served back to the web UI as
    // the "captured photo". Because this copy and the patches below both
    // come from this SAME decoded frame, the photo the user sees and the
    // patches cut from it can never drift apart the way a browser-side
    // canvas snapshot (drawn from the separate, continuously-updating live
    // stream) could versus this on-device capture.
    {
        size_t row_bytes = (size_t)crop_w * APP_AI_PATCH_CHANNELS;
        size_t need = row_bytes * (size_t)crop_h;
        xSemaphoreTake(s_ai.lock, portMAX_DELAY);
        if (need > s_ai.crop_image_cap) {
            uint8_t *bigger = heap_caps_realloc(s_ai.crop_image, need, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!bigger) {
                bigger = heap_caps_realloc(s_ai.crop_image, need, MALLOC_CAP_8BIT);
            }
            if (bigger) {
                s_ai.crop_image = bigger;
                s_ai.crop_image_cap = need;
            }
        }
        if (s_ai.crop_image && need <= s_ai.crop_image_cap) {
            for (int y = 0; y < crop_h; y++) {
                uint8_t *dst_row = s_ai.crop_image + (size_t)y * row_bytes;
                if (crop_x >= 0 && crop_y + y >= 0 && crop_x + crop_w <= width && crop_y + y < height) {
                    const uint8_t *src_row = decoded +
                        ((size_t)(crop_y + y) * width + crop_x) * APP_AI_PATCH_CHANNELS;
                    memcpy(dst_row, src_row, row_bytes);
                } else {
                    for (int x = 0; x < crop_w; ++x) {
                        int sx = crop_x + x, sy = crop_y + y;
                        if (sx < 0) sx = 0; else if (sx >= width) sx = width - 1;
                        if (sy < 0) sy = 0; else if (sy >= height) sy = height - 1;
                        memcpy(dst_row + (size_t)x * 3,
                               decoded + ((size_t)sy * width + sx) * 3, 3);
                    }
                }
            }
            s_ai.crop_image_w = (uint32_t)crop_w;
            s_ai.crop_image_h = (uint32_t)crop_h;
        } else {
            ESP_LOGW(TAG, "Allocate %u byte crop-image buffer failed; captured-photo preview unavailable this capture (patches are unaffected)",
                     (unsigned)need);
            s_ai.crop_image_w = 0;
            s_ai.crop_image_h = 0;
        }
        xSemaphoreGive(s_ai.lock);
    }

    // Fixed 40x40 source cells: for 640x480 this is exactly 16x12 = 192.
    uint32_t cols = (uint32_t)crop_w / APP_AI_PATCH_SIZE;
    uint32_t rows = (uint32_t)crop_h / APP_AI_PATCH_SIZE;
    if (cols == 0 || rows == 0 || cols > APP_AI_PATCH_MAX_COLS || rows > APP_AI_PATCH_MAX_ROWS) {
        ESP_LOGE(TAG, "AI patch capture failed: fixed crop grid out of bounds: crop=%dx%d grid=%ux%u",
                 crop_w, crop_h, (unsigned)cols, (unsigned)rows);
        ret = ESP_ERR_INVALID_SIZE;
        err_code = "fixed_grid_out_of_range";
        goto done;
    }

    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    for (uint32_t r = 0; r < rows; r++) {
        int cell_y0 = crop_y + (int)(r * APP_AI_PATCH_SIZE);
        for (uint32_t c = 0; c < cols; c++) {
            int cell_x0 = crop_x + (int)(c * APP_AI_PATCH_SIZE);
            uint32_t idx = r * cols + c;
            uint8_t *dst = s_ai.pool + (size_t)idx * patch_stride_bytes();
            if (cell_x0 >= 0 && cell_y0 >= 0 &&
                cell_x0 + APP_AI_PATCH_SIZE <= width &&
                cell_y0 + APP_AI_PATCH_SIZE <= height) {
                resize_cell_to_patch(decoded, width, cell_x0, cell_y0,
                                      APP_AI_PATCH_SIZE, APP_AI_PATCH_SIZE, dst);
            } else {
                for (int py = 0; py < APP_AI_PATCH_SIZE; ++py) {
                    for (int px = 0; px < APP_AI_PATCH_SIZE; ++px) {
                        int sx = cell_x0 + px, sy = cell_y0 + py;
                        if (sx < 0) sx = 0; else if (sx >= width) sx = width - 1;
                        if (sy < 0) sy = 0; else if (sy >= height) sy = height - 1;
                        memcpy(dst + ((size_t)py * APP_AI_PATCH_SIZE + px) * 3,
                               decoded + ((size_t)sy * width + sx) * 3, 3);
                    }
                }
            }
        }
    }
    s_ai.cols = cols;
    s_ai.rows = rows;
    s_ai.count = cols * rows;
    s_ai.source_w = (uint32_t)width;
    s_ai.source_h = (uint32_t)height;
    s_ai.crop_x_px = crop_x;
    s_ai.crop_y_px = crop_y;
    s_ai.last_capture_ok = true;
    s_ai.last_capture_us = esp_timer_get_time();
    s_ai.last_error[0] = '\0';
    xSemaphoreGive(s_ai.lock);

    ESP_LOGI(TAG, "AI patch capture ok: source=%dx%d crop=%d,%d %dx%d grid=%ux%u (%u patches of %dx%d)",
             width, height, crop_x, crop_y, crop_w, crop_h, (unsigned)cols, (unsigned)rows,
             (unsigned)(cols * rows), APP_AI_PATCH_SIZE, APP_AI_PATCH_SIZE);

done:
    if (frame) {
        esp_err_t ret3 = app_uvc_return_frame(frame);
        if (ret3 != ESP_OK) {
            ESP_LOGW(TAG, "Return UVC frame after AI capture error failed: %s", esp_err_to_name(ret3));
        }
    }
    if (decoded) {
        heap_caps_free(decoded);
    }
    if (dht_fixed) {
        heap_caps_free(dht_fixed);
    }
    if (ret != ESP_OK && err_code) {
        xSemaphoreTake(s_ai.lock, portMAX_DELAY);
        s_ai.last_capture_ok = false;
        snprintf(s_ai.last_error, sizeof(s_ai.last_error), "%s", err_code);
        xSemaphoreGive(s_ai.lock);
    }
    return ret;
}

static void ai_patch_capture_task(void *arg)
{
    (void)arg;
    esp_err_t ret = app_ai_patch_capture();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Async AI patch capture failed: %s", esp_err_to_name(ret));
    }
    vTaskDelete(NULL);
}

esp_err_t app_ai_patch_capture_async(void)
{
    if (!s_ai.lock || !s_ai.pool) {
        return ESP_ERR_INVALID_STATE;
    }
    // Stack needs to cover the JPEG decode (TJpgDec) working buffer plus the
    // patch-cutting loop; 6 KB matches the margin used elsewhere in this
    // project for similarly-sized image work.
    // Pinned to core 1: the WiFi/lwIP driver tasks always run on core 0 in
    // ESP-IDF, so keeping this CPU-heavy JPEG-decode + crop work on core 1
    // avoids contending with them for cycles/cache and gives more
    // consistent capture timing.
    if (xTaskCreatePinnedToCore(ai_patch_capture_task, "ai_patch_cap", 6144, NULL, 3, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "Create async AI patch capture task failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

uint32_t app_ai_patch_get_count(void)
{
    if (!s_ai.lock) {
        return 0;
    }
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    uint32_t count = s_ai.last_capture_ok ? s_ai.count : 0;
    xSemaphoreGive(s_ai.lock);
    return count;
}

void app_ai_patch_get_grid(uint32_t *cols, uint32_t *rows)
{
    uint32_t c = 0, r = 0;
    if (s_ai.lock) {
        xSemaphoreTake(s_ai.lock, portMAX_DELAY);
        if (s_ai.last_capture_ok) {
            c = s_ai.cols;
            r = s_ai.rows;
        }
        xSemaphoreGive(s_ai.lock);
    }
    if (cols) {
        *cols = c;
    }
    if (rows) {
        *rows = r;
    }
}

const uint8_t *app_ai_patch_get_data(uint32_t index, size_t *len)
{
    if (!s_ai.lock || !s_ai.pool) {
        return NULL;
    }
    const uint8_t *ptr = NULL;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    if (s_ai.last_capture_ok && index < s_ai.count) {
        ptr = s_ai.pool + (size_t)index * patch_stride_bytes();
    }
    xSemaphoreGive(s_ai.lock);
    if (ptr && len) {
        *len = patch_stride_bytes();
    }
    return ptr;
}

const uint8_t *app_ai_patch_get_crop_image(uint32_t *w, uint32_t *h, size_t *len)
{
    if (!s_ai.lock) {
        return NULL;
    }
    const uint8_t *ptr = NULL;
    uint32_t cw = 0, ch = 0;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    if (s_ai.last_capture_ok && s_ai.crop_image && s_ai.crop_image_w > 0 && s_ai.crop_image_h > 0) {
        ptr = s_ai.crop_image;
        cw = s_ai.crop_image_w;
        ch = s_ai.crop_image_h;
    }
    xSemaphoreGive(s_ai.lock);
    if (!ptr) {
        return NULL;
    }
    if (w) {
        *w = cw;
    }
    if (h) {
        *h = ch;
    }
    if (len) {
        *len = (size_t)cw * (size_t)ch * APP_AI_PATCH_CHANNELS;
    }
    return ptr;
}

const uint8_t *app_ai_patch_get_last_jpeg(size_t *len)
{
    if (!len || !s_ai.lock) {
        return NULL;
    }
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    const uint8_t *data = s_ai.last_jpeg;
    *len = s_ai.last_jpeg_len;
    xSemaphoreGive(s_ai.lock);
    return data;
}

void app_ai_patch_mark_jpeg_capture(uint32_t capture_id)
{
    if (!s_ai.lock || capture_id == 0) return;
    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    if (s_ai.last_jpeg_pending) {
        s_ai.last_jpeg_capture_id = capture_id;
        s_ai.last_jpeg_pending = false;
    }
    xSemaphoreGive(s_ai.lock);
}

static esp_err_t copy_last_jpeg(uint32_t capture_id, uint8_t **jpeg, size_t *len)
{
    if (!jpeg || !len) {
        return ESP_ERR_INVALID_ARG;
    }
    *jpeg = NULL;
    *len = 0;
    if (!s_ai.lock) {
        return ESP_ERR_INVALID_STATE;
    }

    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    if (!s_ai.last_jpeg || s_ai.last_jpeg_len == 0 ||
        (capture_id != 0 && capture_id != s_ai.last_jpeg_capture_id)) {
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_NOT_FOUND;
    }

    uint8_t *copy = heap_caps_malloc(s_ai.last_jpeg_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!copy) {
        copy = heap_caps_malloc(s_ai.last_jpeg_len, MALLOC_CAP_8BIT);
    }
    if (!copy) {
        xSemaphoreGive(s_ai.lock);
        return ESP_ERR_NO_MEM;
    }

    memcpy(copy, s_ai.last_jpeg, s_ai.last_jpeg_len);
    *jpeg = copy;
    *len = s_ai.last_jpeg_len;
    xSemaphoreGive(s_ai.lock);
    return ESP_OK;
}

esp_err_t app_ai_patch_copy_last_jpeg(uint8_t **jpeg, size_t *len)
{
    return copy_last_jpeg(0, jpeg, len);
}

esp_err_t app_ai_patch_copy_jpeg_for_capture(uint32_t capture_id, uint8_t **jpeg, size_t *len)
{
    if (capture_id == 0) return ESP_ERR_INVALID_ARG;
    return copy_last_jpeg(capture_id, jpeg, len);
}

esp_err_t app_ai_patch_get_status_json(char *buf, size_t len)
{
    if (!buf || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!s_ai.lock) {
        // Module never initialized successfully (e.g. PSRAM alloc failed at
        // boot) - report a well-formed "unavailable" status instead of
        // touching an uncreated mutex.
        int written = snprintf(buf, len,
                                "{\"ok\":false,\"error\":\"not_initialized\",\"patch_size\":%d,"
                                "\"max_cols\":%d,\"max_rows\":%d,"
                                "\"cols\":0,\"rows\":0,\"count\":0}",
                                APP_AI_PATCH_SIZE, APP_AI_PATCH_MAX_COLS, APP_AI_PATCH_MAX_ROWS);
        return (written > 0 && (size_t)written < len) ? ESP_OK : ESP_ERR_NO_MEM;
    }
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        return ESP_ERR_NO_MEM;
    }

    xSemaphoreTake(s_ai.lock, portMAX_DELAY);
    bool ok = s_ai.last_capture_ok;
    uint32_t cols = s_ai.cols, rows = s_ai.rows, count = s_ai.count;
    uint32_t source_w = s_ai.source_w, source_h = s_ai.source_h;
    int64_t last_us = s_ai.last_capture_us;
    app_ai_patch_crop_t crop = s_ai.crop;
    char err[sizeof(s_ai.last_error)];
    memcpy(err, s_ai.last_error, sizeof(err));
    xSemaphoreGive(s_ai.lock);

    cJSON_AddBoolToObject(root, "ok", ok);
    cJSON_AddNumberToObject(root, "patch_size", APP_AI_PATCH_SIZE);
    // Exposed so the web UI can compute (and draw) the exact same grid the
    // device will cut *before* a capture happens, instead of guessing/
    // hard-coding these constants client-side.
    cJSON_AddNumberToObject(root, "max_cols", APP_AI_PATCH_MAX_COLS);
    cJSON_AddNumberToObject(root, "max_rows", APP_AI_PATCH_MAX_ROWS);
    cJSON_AddNumberToObject(root, "cols", cols);
    cJSON_AddNumberToObject(root, "rows", rows);
    cJSON_AddNumberToObject(root, "count", count);
    cJSON_AddNumberToObject(root, "source_width", source_w);
    cJSON_AddNumberToObject(root, "source_height", source_h);
    cJSON_AddNumberToObject(root, "last_capture_ms", (double)(last_us / 1000));
    if (!ok && err[0]) {
        cJSON_AddStringToObject(root, "error", err);
    }
    cJSON *crop_json = cJSON_AddObjectToObject(root, "crop");
    if (crop_json) {
        cJSON_AddNumberToObject(crop_json, "left", crop.left);
        cJSON_AddNumberToObject(crop_json, "top", crop.top);
        cJSON_AddNumberToObject(crop_json, "width", crop.width);
        cJSON_AddNumberToObject(crop_json, "height", crop.height);
        if (ok && source_w && source_h) {
            uint32_t cw = (source_w / APP_AI_PATCH_SIZE) * APP_AI_PATCH_SIZE;
            uint32_t ch = (source_h / APP_AI_PATCH_SIZE) * APP_AI_PATCH_SIZE;
            if (cw > APP_AI_PATCH_MAX_CROP_W) cw = APP_AI_PATCH_MAX_CROP_W;
            if (ch > APP_AI_PATCH_MAX_CROP_H) ch = APP_AI_PATCH_MAX_CROP_H;
            cJSON_AddNumberToObject(crop_json, "pixel_width", cw);
            cJSON_AddNumberToObject(crop_json, "pixel_height", ch);
            cJSON_AddNumberToObject(crop_json, "patch_cols", cw / APP_AI_PATCH_SIZE);
            cJSON_AddNumberToObject(crop_json, "patch_rows", ch / APP_AI_PATCH_SIZE);
            cJSON_AddNumberToObject(crop_json, "patch_count", (cw / APP_AI_PATCH_SIZE) * (ch / APP_AI_PATCH_SIZE));
        }
    }

    bool printed = cJSON_PrintPreallocated(root, buf, (int)len, false);
    cJSON_Delete(root);
    return printed ? ESP_OK : ESP_ERR_NO_MEM;
}

void app_ai_patch_debug_dump(uint32_t index)
{
    const uint8_t *ptr;
    size_t len;
    size_t i;

    ptr = app_ai_patch_get_data(index, &len);
    if (!ptr || len == 0) {
        ESP_LOGE("PATCH_DUMP", "[PATCH DUMP] patch=%lu unavailable (chua capture hoac sai index)", (unsigned long)index);
        return;
    }

    ESP_LOGI("PATCH_DUMP", "[PATCH DUMP] patch=%lu len=%lu",
           (unsigned long)index,
           (unsigned long)len);

    for (i = 0; i < len; i++) {
        printf("%02X", ptr[i]);

        if (i + 1 < len) {
            printf(",");
        }

        if ((i + 1) % 32 == 0) {
            printf("\n");
        }
    }

    printf("\n[PATCH DUMP END]\n");
    fflush(stdout); // FORCE FLUSH DỮ LIỆU RA SERIAL PORT NGAY LẬP TỨC
}
