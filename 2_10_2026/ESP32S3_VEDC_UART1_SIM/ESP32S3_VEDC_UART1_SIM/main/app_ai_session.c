/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "app_ai_session.h"
#include "app_ai_patch.h"
#include "app_ai_infer.h"
#include "app_ai_coreset.h"
#include "app_web.h"
#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char *TAG = "app_ai_session";

#define SESSION_WORK_PRIORITY 3
#define SESSION_STACK_BYTES 16384

typedef enum {
    SESSION_IDLE = 0,
    SESSION_COLLECTING,
    SESSION_PROCESSING,
    SESSION_BUILDING_CORESET,
    SESSION_FINISHED,
} session_state_t;

typedef struct {
    uint8_t *crop_rgb;
    size_t crop_bytes;
    uint32_t w;
    uint32_t h;
    uint32_t patch_count;
    uint32_t cols;
    uint32_t rows;
    float *features;
    float crop_left;
    float crop_top;
    float crop_width;
    float crop_height;
} session_image_t;

static struct {
    SemaphoreHandle_t lock;
    session_state_t state;
    session_image_t images[APP_AI_SESSION_MAX_IMAGES];
    uint32_t image_count;
    uint32_t processed_count;
    uint32_t total_patches;
    size_t total_rgb_bytes;
    int32_t processing_image;
    bool capture_busy;
    char error[64];
} s_session;

static const char *state_name(session_state_t state)
{
    switch (state) {
    case SESSION_COLLECTING: return "collecting";
    case SESSION_PROCESSING: return "processing";
    case SESSION_BUILDING_CORESET: return "building_coreset";
    case SESSION_FINISHED: return "finished";
    default: return "idle";
    }
}

static void free_images_locked(void)
{
    app_ai_coreset_reset();
    for (uint32_t i = 0; i < s_session.image_count; i++) {
        if (s_session.images[i].crop_rgb) {
            heap_caps_free(s_session.images[i].crop_rgb);
        }
        if (s_session.images[i].features) {
            heap_caps_free(s_session.images[i].features);
        }
        memset(&s_session.images[i], 0, sizeof(s_session.images[i]));
    }
    s_session.image_count = 0;
    s_session.processed_count = 0;
    s_session.total_patches = 0;
    s_session.total_rgb_bytes = 0;
    s_session.processing_image = -1;
    s_session.error[0] = '\0';
}

static esp_err_t ensure_initialized(void)
{
    if (s_session.lock) {
        return ESP_OK;
    }
    s_session.lock = xSemaphoreCreateMutex();
    if (!s_session.lock) {
        return ESP_ERR_NO_MEM;
    }
    s_session.state = SESSION_IDLE;
    s_session.processing_image = -1;
    return ESP_OK;
}

static esp_err_t capture_one(void)
{
    esp_err_t ret = ensure_initialized();
    if (ret != ESP_OK) {
        return ret;
    }

    // Do not start a camera capture after P3 has already switched the session
    // into processing mode. This closes the small race between the polling
    // button task and the worker task.
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    if (s_session.state != SESSION_COLLECTING) {
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreGive(s_session.lock);

    // Capture the exact same on-device crop as the existing single-photo path.
    ret = app_ai_patch_capture();
    if (ret != ESP_OK) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ret;
    }

    uint32_t w = 0, h = 0;
    size_t len = 0;
    const uint8_t *src = app_ai_patch_get_crop_image(&w, &h, &len);
    if (!src || !w || !h || !len) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NOT_FOUND;
    }

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    if (s_session.image_count >= APP_AI_SESSION_MAX_IMAGES) {
        strncpy(s_session.error, "max_images", sizeof(s_session.error) - 1);
        s_session.error[sizeof(s_session.error) - 1] = '\0';
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NO_MEM;
    }
    if (s_session.total_rgb_bytes + len > APP_AI_SESSION_MAX_RGB_BYTES) {
        strncpy(s_session.error, "psram_limit", sizeof(s_session.error) - 1);
        s_session.error[sizeof(s_session.error) - 1] = '\0';
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NO_MEM;
    }
    uint32_t idx = s_session.image_count;
    xSemaphoreGive(s_session.lock);

    uint8_t *copy = heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!copy) {
        copy = heap_caps_malloc(len, MALLOC_CAP_8BIT);
    }
    if (!copy) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        strncpy(s_session.error, "no_mem_crop", sizeof(s_session.error) - 1);
        s_session.error[sizeof(s_session.error) - 1] = '\0';
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NO_MEM;
    }
    memcpy(copy, src, len);

    float left = 0, top = 0, width = 100, height = 100;
    app_ai_patch_get_crop_percent(&left, &top, &width, &height);

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    // Re-check after allocation in case another callback slipped through.
    if (s_session.state == SESSION_PROCESSING ||
        s_session.image_count != idx ||
        s_session.total_rgb_bytes + len > APP_AI_SESSION_MAX_RGB_BYTES) {
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        heap_caps_free(copy);
        return ESP_ERR_INVALID_STATE;
    }
    session_image_t *image = &s_session.images[idx];
    image->crop_rgb = copy;
    image->crop_bytes = len;
    image->w = w;
    image->h = h;
    image->crop_left = left;
    image->crop_top = top;
    image->crop_width = width;
    image->crop_height = height;
    s_session.image_count++;
    s_session.total_rgb_bytes += len;
    s_session.state = SESSION_COLLECTING;
    s_session.capture_busy = false;
    s_session.error[0] = '\0';
    xSemaphoreGive(s_session.lock);

    ESP_LOGI(TAG, "P4 image #%u captured: crop=%ux%u RGB=%u bytes, session=%u image(s)",
             (unsigned)(idx + 1), (unsigned)w, (unsigned)h, (unsigned)len,
             (unsigned)(idx + 1));
    app_web_notify_ai_session("capture");
    return ESP_OK;
}

static void capture_task(void *arg)
{
    (void)arg;
    esp_err_t ret = capture_one();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "P4 multi-capture failed: %s", esp_err_to_name(ret));
        app_web_notify_ai_session("capture_error");
    }
    vTaskDelete(NULL);
}

esp_err_t app_ai_session_capture_async(void)
{
    esp_err_t ret = ensure_initialized();
    if (ret != ESP_OK) return ret;

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    if (s_session.state == SESSION_PROCESSING || s_session.processing_image >= 0 || s_session.capture_busy) {
        xSemaphoreGive(s_session.lock);
        ESP_LOGW(TAG, "Ignore P4 while AI session is processing");
        return ESP_ERR_INVALID_STATE;
    }
    if (s_session.state == SESSION_IDLE || s_session.state == SESSION_FINISHED) {
        free_images_locked();
        s_session.state = SESSION_COLLECTING;
    }
    s_session.capture_busy = true;
    xSemaphoreGive(s_session.lock);

    // Ghim vào core 1: WiFi/lwIP luôn chạy trên core 0 trong ESP-IDF, để
    // capture+crop (nặng CPU vì giải JPEG + cắt patch) tránh tranh chấp.
    if (xTaskCreatePinnedToCore(capture_task, "ai_p4_cap", SESSION_STACK_BYTES, NULL,
                                SESSION_WORK_PRIORITY, NULL, 1) != pdPASS) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        s_session.capture_busy = false;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

static void process_task(void *arg)
{
    (void)arg;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    uint32_t count = s_session.image_count;
    s_session.state = SESSION_PROCESSING;
    s_session.processed_count = 0;
    s_session.total_patches = 0;
    s_session.processing_image = 0;
    s_session.error[0] = '\0';
    xSemaphoreGive(s_session.lock);

    if (count == 0) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        s_session.state = SESSION_IDLE;
        strncpy(s_session.error, "no_images", sizeof(s_session.error) - 1);
        s_session.processing_image = -1;
        xSemaphoreGive(s_session.lock);
        app_web_notify_ai_session("finish_error");
        vTaskDelete(NULL);
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        session_image_t *image = &s_session.images[i];
        const uint8_t *crop = image->crop_rgb;
        uint32_t w = image->w;
        uint32_t h = image->h;
        s_session.processing_image = (int32_t)i;
        xSemaphoreGive(s_session.lock);

        esp_err_t ret = app_ai_patch_process_crop_rgb(crop, w, h);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Process image #%u patches failed: %s", (unsigned)(i + 1), esp_err_to_name(ret));
            xSemaphoreTake(s_session.lock, portMAX_DELAY);
            snprintf(s_session.error, sizeof(s_session.error), "patch_image_%u", (unsigned)(i + 1));
            s_session.processing_image = -1;
            xSemaphoreGive(s_session.lock);
            break;
        }

        uint32_t patch_count = app_ai_patch_get_count();
        uint32_t cols = 0, rows = 0;
        app_ai_patch_get_grid(&cols, &rows);

        ret = app_ai_infer_run_all();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "AI inference image #%u failed: %s", (unsigned)(i + 1), esp_err_to_name(ret));
            xSemaphoreTake(s_session.lock, portMAX_DELAY);
            snprintf(s_session.error, sizeof(s_session.error), "infer_image_%u", (unsigned)(i + 1));
            s_session.processing_image = -1;
            xSemaphoreGive(s_session.lock);
            break;
        }

        size_t feature_count = (size_t)patch_count * APP_AI_INFER_FEATURE_DIM;
        size_t feature_bytes = feature_count * sizeof(float);
        float *features = heap_caps_malloc(feature_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!features) {
            features = heap_caps_malloc(feature_bytes, MALLOC_CAP_8BIT);
        }
        if (!features) {
            ESP_LOGE(TAG, "Allocate features for image #%u failed (%u bytes)", (unsigned)(i + 1), (unsigned)feature_bytes);
            xSemaphoreTake(s_session.lock, portMAX_DELAY);
            strncpy(s_session.error, "no_mem_features", sizeof(s_session.error) - 1);
            s_session.processing_image = -1;
            xSemaphoreGive(s_session.lock);
            break;
        }
        bool copy_ok = true;
        for (uint32_t p = 0; p < patch_count; p++) {
            esp_err_t get_ret = app_ai_infer_get_vector(p,
                features + (size_t)p * APP_AI_INFER_FEATURE_DIM,
                APP_AI_INFER_FEATURE_DIM);
            if (get_ret != ESP_OK) {
                copy_ok = false;
                break;
            }
        }
        if (!copy_ok) {
            heap_caps_free(features);
            xSemaphoreTake(s_session.lock, portMAX_DELAY);
            snprintf(s_session.error, sizeof(s_session.error), "feature_copy_image_%u", (unsigned)(i + 1));
            s_session.processing_image = -1;
            xSemaphoreGive(s_session.lock);
            break;
        }

        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        image->patch_count = patch_count;
        image->cols = cols;
        image->rows = rows;
        image->features = features;
        s_session.processed_count = i + 1;
        s_session.total_patches += patch_count;
        s_session.processing_image = (int32_t)i;
        xSemaphoreGive(s_session.lock);

        ESP_LOGI(TAG, "AI session image #%u/%u done: %ux%u patches=%u features=%u-d",
                 (unsigned)(i + 1), (unsigned)count, (unsigned)cols, (unsigned)rows,
                 (unsigned)patch_count, APP_AI_INFER_FEATURE_DIM);
        app_web_notify_ai_session("progress");
        vTaskDelay(1);
    }

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    bool success = (s_session.processed_count == count && s_session.error[0] == '\0');
    s_session.processing_image = -1;
    if (success) {
        s_session.state = SESSION_BUILDING_CORESET;
    }
    xSemaphoreGive(s_session.lock);

    if (success) {
        app_web_notify_ai_session("coreset_start");

        app_ai_coreset_block_t blocks[APP_AI_SESSION_MAX_IMAGES];
        memset(blocks, 0, sizeof(blocks));
        uint32_t block_count = 0;

        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        while (block_count < count && block_count < APP_AI_SESSION_MAX_IMAGES) {
            blocks[block_count].vectors = s_session.images[block_count].features;
            blocks[block_count].count = s_session.images[block_count].patch_count;
            blocks[block_count].source_image = block_count;
            block_count++;
        }
        xSemaphoreGive(s_session.lock);

        esp_err_t coret = app_ai_coreset_build(blocks, block_count);
        if (coret != ESP_OK) {
            success = false;
            xSemaphoreTake(s_session.lock, portMAX_DELAY);
            strncpy(s_session.error, "coreset_failed", sizeof(s_session.error) - 1);
            s_session.error[sizeof(s_session.error) - 1] = '\0';
            xSemaphoreGive(s_session.lock);
            ESP_LOGE(TAG, "Coreset build failed: %s", esp_err_to_name(coret));
        } else {
            ESP_LOGI(TAG, "Memory Bank built from %u image(s): %u/%u feature vectors kept",
                     (unsigned)block_count,
                     (unsigned)app_ai_coreset_get_bank_count(),
                     (unsigned)count);
        }
    }

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    s_session.processing_image = -1;
    s_session.state = SESSION_FINISHED;
    xSemaphoreGive(s_session.lock);
    ESP_LOGI(TAG, "AI session finished: %u/%u images processed, %u total patches",
             (unsigned)s_session.processed_count, (unsigned)count, (unsigned)s_session.total_patches);
    app_web_notify_ai_session(success ? "finish" : "finish_error");
    vTaskDelete(NULL);
}

esp_err_t app_ai_session_finish_async(void)
{
    esp_err_t ret = ensure_initialized();
    if (ret != ESP_OK) return ret;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    if (s_session.state != SESSION_COLLECTING || s_session.image_count == 0 || s_session.capture_busy) {
        xSemaphoreGive(s_session.lock);
        ESP_LOGW(TAG, "P3 ignored: no collecting session (state=%s, images=%u)",
                 state_name(s_session.state), (unsigned)s_session.image_count);
        return ESP_ERR_INVALID_STATE;
    }
    s_session.state = SESSION_PROCESSING;
    xSemaphoreGive(s_session.lock);

    // Ghim vào core 1 cùng lý do: chạy TFLite-Micro Invoke() liên tục cho
    // tới 300 patch, để riêng khỏi core 0 (WiFi/lwIP) cho ổn định & nhanh hơn.
    if (xTaskCreatePinnedToCore(process_task, "ai_p3_proc", SESSION_STACK_BYTES, NULL,
                                SESSION_WORK_PRIORITY, NULL, 1) != pdPASS) {
        xSemaphoreTake(s_session.lock, portMAX_DELAY);
        s_session.state = SESSION_COLLECTING;
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

bool app_ai_session_is_collecting(void)
{
    if (ensure_initialized() != ESP_OK) return false;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    bool v = (s_session.state == SESSION_COLLECTING);
    xSemaphoreGive(s_session.lock);
    return v;
}

bool app_ai_session_is_busy(void)
{
    if (ensure_initialized() != ESP_OK) return false;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    bool v = (s_session.state == SESSION_PROCESSING);
    xSemaphoreGive(s_session.lock);
    return v;
}

uint32_t app_ai_session_get_image_count(void)
{
    if (ensure_initialized() != ESP_OK) return 0;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    uint32_t v = s_session.image_count;
    xSemaphoreGive(s_session.lock);
    return v;
}

uint32_t app_ai_session_get_processed_count(void)
{
    if (ensure_initialized() != ESP_OK) return 0;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    uint32_t v = s_session.processed_count;
    xSemaphoreGive(s_session.lock);
    return v;
}

esp_err_t app_ai_session_get_status_json(char *buf, size_t len)
{
    if (!buf || len == 0 || ensure_initialized() != ESP_OK) return ESP_ERR_INVALID_ARG;

    cJSON *root = cJSON_CreateObject();
    if (!root) return ESP_ERR_NO_MEM;

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    cJSON_AddStringToObject(root, "state", state_name(s_session.state));
    cJSON_AddNumberToObject(root, "image_count", s_session.image_count);
    cJSON_AddNumberToObject(root, "processed_count", s_session.processed_count);
    cJSON_AddNumberToObject(root, "total_patches", s_session.total_patches);
    cJSON_AddNumberToObject(root, "stored_rgb_bytes", (double)s_session.total_rgb_bytes);
    cJSON_AddNumberToObject(root, "max_images", APP_AI_SESSION_MAX_IMAGES);
    cJSON_AddNumberToObject(root, "max_rgb_bytes", APP_AI_SESSION_MAX_RGB_BYTES);
    cJSON_AddNumberToObject(root, "processing_image", s_session.processing_image >= 0 ? s_session.processing_image + 1 : 0);
    if (s_session.error[0]) cJSON_AddStringToObject(root, "error", s_session.error);

    cJSON *images = cJSON_AddArrayToObject(root, "images");
    if (images) {
        for (uint32_t i = 0; i < s_session.image_count; i++) {
            const session_image_t *im = &s_session.images[i];
            cJSON *obj = cJSON_CreateObject();
            if (!obj) continue;
            cJSON_AddNumberToObject(obj, "index", i);
            cJSON_AddNumberToObject(obj, "width", im->w);
            cJSON_AddNumberToObject(obj, "height", im->h);
            cJSON_AddNumberToObject(obj, "patch_count", im->patch_count);
            cJSON_AddNumberToObject(obj, "cols", im->cols);
            cJSON_AddNumberToObject(obj, "rows", im->rows);
            cJSON_AddNumberToObject(obj, "rgb_bytes", (double)im->crop_bytes);
            cJSON_AddItemToArray(images, obj);
        }
    }
    char err[sizeof(s_session.error)];
    memcpy(err, s_session.error, sizeof(err));
    xSemaphoreGive(s_session.lock);

    if (err[0]) {
        cJSON_AddStringToObject(root, "last_error", err);
    }
    bool ok = cJSON_PrintPreallocated(root, buf, (int)len, false);
    cJSON_Delete(root);
    return ok ? ESP_OK : ESP_ERR_NO_MEM;
}

esp_err_t app_ai_session_get_image_features_json(uint32_t image_index, char *buf, size_t len)
{
    if (!buf || len == 0 || ensure_initialized() != ESP_OK) return ESP_ERR_INVALID_ARG;

    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    if (image_index >= s_session.image_count) {
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NOT_FOUND;
    }
    const session_image_t *im = &s_session.images[image_index];
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        xSemaphoreGive(s_session.lock);
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddNumberToObject(root, "image_index", image_index);
    cJSON_AddNumberToObject(root, "image_number", image_index + 1);
    cJSON_AddNumberToObject(root, "patch_count", im->patch_count);
    cJSON_AddNumberToObject(root, "cols", im->cols);
    cJSON_AddNumberToObject(root, "rows", im->rows);
    cJSON_AddNumberToObject(root, "feature_dim", APP_AI_INFER_FEATURE_DIM);
    cJSON_AddNumberToObject(root, "width", im->w);
    cJSON_AddNumberToObject(root, "height", im->h);
    if (im->features && im->patch_count) {
        cJSON *vectors = cJSON_AddArrayToObject(root, "vectors");
        if (!vectors) {
            cJSON_Delete(root);
            xSemaphoreGive(s_session.lock);
            return ESP_ERR_NO_MEM;
        }
        for (uint32_t p = 0; p < im->patch_count; p++) {
            cJSON *vec = cJSON_CreateFloatArray(
                im->features + (size_t)p * APP_AI_INFER_FEATURE_DIM,
                APP_AI_INFER_FEATURE_DIM);
            if (!vec) continue;
            cJSON_AddItemToArray(vectors, vec);
        }
    }
    bool ok = cJSON_PrintPreallocated(root, buf, (int)len, false);
    cJSON_Delete(root);
    xSemaphoreGive(s_session.lock);
    return ok ? ESP_OK : ESP_ERR_NO_MEM;
}

const uint8_t *app_ai_session_get_image_crop(uint32_t image_index,
                                             uint32_t *w, uint32_t *h,
                                             size_t *len)
{
    if (ensure_initialized() != ESP_OK) return NULL;
    xSemaphoreTake(s_session.lock, portMAX_DELAY);
    if (image_index >= s_session.image_count) {
        xSemaphoreGive(s_session.lock);
        return NULL;
    }
    session_image_t *im = &s_session.images[image_index];
    if (w) *w = im->w;
    if (h) *h = im->h;
    if (len) *len = im->crop_bytes;
    const uint8_t *ptr = im->crop_rgb;
    xSemaphoreGive(s_session.lock);
    return ptr;
}
