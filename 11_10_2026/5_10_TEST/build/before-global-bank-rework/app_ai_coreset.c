/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "app_ai_coreset.h"
#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

static const char *TAG = "app_ai_coreset";

#define CORESET_CANDIDATE_CHUNK 128U
#define CORESET_MIN_SCORE_FLOOR 0.001f

// Chỉ dùng cho log chẩn đoán "phân bố dmin" bên dưới - không ảnh hưởng
// thuật toán. Đủ lớn cho LEARN_SAMPLES=5 ảnh hiện tại, có dư chỗ.
#define CORESET_DIAG_MAX_IMAGES 8U
// Ranh giới bucket dùng để soi "vực" giữa nhóm patch gần-trùng-lặp (nền
// đồng nhất) và nhóm patch thực sự khác biệt (linh kiện, đường mạch...).
#define CORESET_DIAG_BUCKET_2 0.05f
#define CORESET_DIAG_BUCKET_3 0.10f
#define CORESET_DIAG_BUCKET_4 0.20f
#define SPATIAL_MAX_IMAGES 5U
#define APPEARANCE_AI_MIN_SCORE 0.65f
#define APPEARANCE_CONFIRM_SCORE 1.35f

static inline float dist2_32(const float *a, const float *b);

static struct {
    SemaphoreHandle_t lock;
    float *bank;                 /* bank_count * feature_dim */
    uint32_t *bank_image;
    uint32_t *bank_patch;
    float *bank_selection_dist;
    uint32_t bank_count;
    uint32_t total_vectors;
    float threshold;
    float train_max_score;
    float epsilon;
    uint32_t target_bank;
    uint32_t min_bank;
    uint32_t source_images;
    int64_t build_us;
    bool ready;
    const float *spatial_samples[SPATIAL_MAX_IMAGES];
    const uint8_t *appearance_samples[SPATIAL_MAX_IMAGES];
    uint32_t spatial_count;
    uint32_t spatial_images;
    uint8_t spatial_mask[APP_AI_PATCH_MAX_COUNT];
    float spatial_limits[APP_AI_PATCH_MAX_COUNT];
    uint8_t appearance_mask[APP_AI_PATCH_MAX_COUNT];
    float appearance_limits[APP_AI_PATCH_MAX_COUNT];
    float appearance_floor;
    float spatial_floor;
} s_core = {0};

static int compare_float(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static float appearance_distance(const uint8_t *a, const uint8_t *b)
{
    int offset[3];
    for (uint32_t c = 0; c < 3; ++c) {
        int differences[16];
        for (uint32_t cell = 0; cell < 16; ++cell) {
            differences[cell] = (int)a[cell * 3 + c] - (int)b[cell * 3 + c];
        }
        for (uint32_t i = 1; i < 16; ++i) {
            int value = differences[i];
            int j = (int)i - 1;
            while (j >= 0 && differences[j] > value) {
                differences[j + 1] = differences[j];
                --j;
            }
            differences[j + 1] = value;
        }
        offset[c] = (differences[7] + differences[8]) / 2;
    }
    float worst = 0.0f;
    for (uint32_t cell = 0; cell < 16; ++cell) {
        const uint32_t base = cell * 3;
        float d = (float)(abs((int)a[base] - (int)b[base] - offset[0]) +
                          abs((int)a[base + 1] - (int)b[base + 1] - offset[1]) +
                          abs((int)a[base + 2] - (int)b[base + 2] - offset[2])) / 3.0f;
        if (d > worst) worst = d;
    }
    return worst;
}

static esp_err_t build_spatial_bank(const app_ai_coreset_block_t *blocks, uint32_t images)
{
    if (images < 3 || images > SPATIAL_MAX_IMAGES || !blocks[0].vectors ||
        blocks[0].count == 0 || blocks[0].count > APP_AI_PATCH_MAX_COUNT) return ESP_ERR_INVALID_ARG;
    const uint32_t count = blocks[0].count;
    for (uint32_t i = 0; i < images; ++i) {
        if (!blocks[i].vectors || !blocks[i].appearance || blocks[i].count != count) {
            ESP_LOGE(TAG, "LEARN image %u grid differs from first image", (unsigned)(i + 1));
            return ESP_ERR_INVALID_SIZE;
        }
    }
    float *calibration = heap_caps_malloc(5U * APP_AI_PATCH_MAX_COUNT * sizeof(float),
                                          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!calibration) calibration = heap_caps_malloc(5U * APP_AI_PATCH_MAX_COUNT * sizeof(float),
                                                     MALLOC_CAP_8BIT);
    if (!calibration) return ESP_ERR_NO_MEM;
    float *noise = calibration;
    float *spread = noise + APP_AI_PATCH_MAX_COUNT;
    float *appearance_noise = spread + APP_AI_PATCH_MAX_COUNT;
    float *appearance_spread = appearance_noise + APP_AI_PATCH_MAX_COUNT;
    float *sorted_noise = appearance_spread + APP_AI_PATCH_MAX_COUNT;
    uint32_t excluded = 0;
    uint32_t appearance_excluded = 0;
    uint32_t excluded_per_image[SPATIAL_MAX_IMAGES] = {0};
    for (uint32_t p = 0; p < count; ++p) {
        float pair[SPATIAL_MAX_IMAGES][SPATIAL_MAX_IMAGES] = {{0}};
        uint32_t medoid = 0;
        float best_sum = FLT_MAX;
        for (uint32_t i = 0; i < images; ++i) {
            float sum = 0;
            const float *a = blocks[i].vectors + (size_t)p * APP_AI_INFER_FEATURE_DIM;
            for (uint32_t j = 0; j < images; ++j) {
                if (i == j) continue;
                const float *b = blocks[j].vectors + (size_t)p * APP_AI_INFER_FEATURE_DIM;
                pair[i][j] = sqrtf(dist2_32(a, b));
                if (!isfinite(pair[i][j])) {
                    heap_caps_free(calibration);
                    return ESP_ERR_INVALID_ARG;
                }
                sum += pair[i][j];
            }
            if (sum < best_sum) { best_sum = sum; medoid = i; }
        }
        float distances[SPATIAL_MAX_IMAGES];
        for (uint32_t i = 0; i < images; ++i) distances[i] = pair[medoid][i];
        float sorted[SPATIAL_MAX_IMAGES];
        memcpy(sorted, distances, images * sizeof(float));
        qsort(sorted, images, sizeof(float), compare_float);
        uint8_t mask = (uint8_t)((1U << images) - 1U);
        // One contaminated LEARN image must not turn a foreign object into
        // the normal example for this position.
        if (sorted[images - 1] > fmaxf(0.12f, sorted[images - 2] * 2.5f)) {
            for (uint32_t i = 0; i < images; ++i) {
                if (distances[i] == sorted[images - 1] && i != medoid) {
                    mask &= (uint8_t)~(1U << i);
                    ++excluded;
                    ++excluded_per_image[i];
                    break;
                }
            }
        }
        s_core.spatial_mask[p] = mask;
        float largest_loo = 0;
        float nearest[SPATIAL_MAX_IMAGES];
        uint32_t nearest_count = 0;
        for (uint32_t i = 0; i < images; ++i) {
            if (!(mask & (1U << i))) continue;
            float d = FLT_MAX;
            for (uint32_t j = 0; j < images; ++j) {
                if (i != j && (mask & (1U << j)) && pair[i][j] < d) d = pair[i][j];
            }
            nearest[nearest_count++] = d;
            if (d > largest_loo) largest_loo = d;
        }
        qsort(nearest, nearest_count, sizeof(float), compare_float);
        noise[p] = nearest[nearest_count / 2];
        spread[p] = largest_loo;

        float color_pair[SPATIAL_MAX_IMAGES][SPATIAL_MAX_IMAGES] = {{0}};
        uint32_t color_medoid = 0;
        float color_best_sum = FLT_MAX;
        for (uint32_t i = 0; i < images; ++i) {
            float sum = 0;
            const uint8_t *a = blocks[i].appearance + (size_t)p * APP_AI_PATCH_APPEARANCE_DIM;
            for (uint32_t j = 0; j < images; ++j) {
                if (i == j) continue;
                const uint8_t *b = blocks[j].appearance + (size_t)p * APP_AI_PATCH_APPEARANCE_DIM;
                color_pair[i][j] = appearance_distance(a, b);
                sum += color_pair[i][j];
            }
            if (sum < color_best_sum) { color_best_sum = sum; color_medoid = i; }
        }
        for (uint32_t i = 0; i < images; ++i) distances[i] = color_pair[color_medoid][i];
        memcpy(sorted, distances, images * sizeof(float));
        qsort(sorted, images, sizeof(float), compare_float);
        mask = (uint8_t)((1U << images) - 1U);
        if (sorted[images - 1] > fmaxf(18.0f, sorted[images - 2] * 2.5f)) {
            for (uint32_t i = 0; i < images; ++i) {
                if (distances[i] == sorted[images - 1] && i != color_medoid) {
                    mask &= (uint8_t)~(1U << i);
                    ++appearance_excluded;
                    ++excluded_per_image[i];
                    break;
                }
            }
        }
        s_core.appearance_mask[p] = mask;
        largest_loo = 0;
        nearest_count = 0;
        for (uint32_t i = 0; i < images; ++i) {
            if (!(mask & (1U << i))) continue;
            float d = FLT_MAX;
            for (uint32_t j = 0; j < images; ++j) {
                if (i != j && (mask & (1U << j)) && color_pair[i][j] < d) d = color_pair[i][j];
            }
            nearest[nearest_count++] = d;
            if (d > largest_loo) largest_loo = d;
        }
        qsort(nearest, nearest_count, sizeof(float), compare_float);
        appearance_noise[p] = nearest[nearest_count / 2];
        appearance_spread[p] = largest_loo;
    }
    memcpy(sorted_noise, noise, count * sizeof(float));
    qsort(sorted_noise, count, sizeof(float), compare_float);
    const float floor = fmaxf(0.10f, sorted_noise[(count * 9U) / 10U] * 1.25f);
    memcpy(sorted_noise, appearance_noise, count * sizeof(float));
    qsort(sorted_noise, count, sizeof(float), compare_float);
    const float appearance_floor = fmaxf(12.0f, sorted_noise[(count * 9U) / 10U] * 1.5f);
    for (uint32_t p = 0; p < count; ++p) {
        s_core.spatial_limits[p] = fmaxf(floor, spread[p] * 1.8f);
        s_core.appearance_limits[p] = fmaxf(appearance_floor, appearance_spread[p] * 2.0f);
    }
    for (uint32_t i = 0; i < images; ++i) {
        s_core.spatial_samples[i] = blocks[i].vectors;
        s_core.appearance_samples[i] = blocks[i].appearance;
    }
    s_core.spatial_count = count;
    s_core.spatial_images = images;
    s_core.spatial_floor = floor;
    s_core.appearance_floor = appearance_floor;
    ESP_LOGI(TAG, "Spatial bank: %u positions x %u images, noise floor %.4f, excluded %u outlier patches",
             (unsigned)count, (unsigned)images, (double)floor, (unsigned)excluded);
    ESP_LOGI(TAG, "Appearance bank: noise floor %.2f, excluded %u outlier patches",
             (double)appearance_floor, (unsigned)appearance_excluded);
    for (uint32_t i = 0; i < images; ++i) {
        if (excluded_per_image[i] > count / 5) {
            ESP_LOGW(TAG, "LEARN image %u differs at %u patch comparisons; inspect/re-capture this sample",
                     (unsigned)(i + 1), (unsigned)excluded_per_image[i]);
        }
    }
    heap_caps_free(calibration);
    return ESP_OK;
}

static esp_err_t ensure_init(void)
{
    if (s_core.lock) return ESP_OK;
    s_core.lock = xSemaphoreCreateMutex();
    if (!s_core.lock) return ESP_ERR_NO_MEM;
    s_core.epsilon = APP_AI_CORESET_EPSILON_MIN;
    return ESP_OK;
}

/* 
 * Tính khoảng cách Euclidean bình phương chuẩn C.
 * Tránh hoàn toàn lỗi Alignment khi dùng mảng tạm với ESP-DSP SIMD trên ESP32-S3.
 */
static inline float dist2_32(const float *a, const float *b)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < APP_AI_INFER_FEATURE_DIM; i++) {
        float diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return sum_sq;
}

static esp_err_t build_candidate_index(const app_ai_coreset_block_t *blocks,
                                       uint32_t block_count,
                                       uint32_t total,
                                       const float ***out_vectors,
                                       uint32_t **out_images,
                                       uint32_t **out_patches)
{
    const float **vectors = heap_caps_malloc((size_t)total * sizeof(*vectors), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!vectors) vectors = heap_caps_malloc((size_t)total * sizeof(*vectors), MALLOC_CAP_8BIT);
    uint32_t *images = heap_caps_malloc((size_t)total * sizeof(*images), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!images) images = heap_caps_malloc((size_t)total * sizeof(*images), MALLOC_CAP_8BIT);
    uint32_t *patches = heap_caps_malloc((size_t)total * sizeof(*patches), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!patches) patches = heap_caps_malloc((size_t)total * sizeof(*patches), MALLOC_CAP_8BIT);
    if (!vectors || !images || !patches) {
        if (vectors) heap_caps_free((void *)vectors);
        if (images) heap_caps_free(images);
        if (patches) heap_caps_free(patches);
        return ESP_ERR_NO_MEM;
    }

    uint32_t n = 0;
    uint32_t b = 0;
    while (b < block_count) {
        uint32_t p = 0;
        while (p < blocks[b].count) {
            vectors[n] = blocks[b].vectors + (size_t)p * APP_AI_INFER_FEATURE_DIM;
            images[n] = blocks[b].source_image;
            patches[n] = p;
            n++;
            p++;
        }
        b++;
    }
    *out_vectors = vectors;
    *out_images = images;
    *out_patches = patches;
    return ESP_OK;
}

static esp_err_t allocate_bank_storage(void)
{
    if (s_core.bank) return ESP_OK;
    const size_t bytes = (size_t)APP_AI_CORESET_MAX_BANK * APP_AI_INFER_FEATURE_DIM * sizeof(float);
    s_core.bank = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_core.bank) s_core.bank = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
    if (!s_core.bank) return ESP_ERR_NO_MEM;

    s_core.bank_image = heap_caps_malloc(APP_AI_CORESET_MAX_BANK * sizeof(uint32_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_core.bank_image) s_core.bank_image = heap_caps_malloc(APP_AI_CORESET_MAX_BANK * sizeof(uint32_t), MALLOC_CAP_8BIT);
    s_core.bank_patch = heap_caps_malloc(APP_AI_CORESET_MAX_BANK * sizeof(uint32_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_core.bank_patch) s_core.bank_patch = heap_caps_malloc(APP_AI_CORESET_MAX_BANK * sizeof(uint32_t), MALLOC_CAP_8BIT);
    s_core.bank_selection_dist = heap_caps_malloc(APP_AI_CORESET_MAX_BANK * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_core.bank_selection_dist) s_core.bank_selection_dist = heap_caps_malloc(APP_AI_CORESET_MAX_BANK * sizeof(float), MALLOC_CAP_8BIT);

    if (!s_core.bank_image || !s_core.bank_patch || !s_core.bank_selection_dist) {
        if (s_core.bank) heap_caps_free(s_core.bank);
        if (s_core.bank_image) heap_caps_free(s_core.bank_image);
        if (s_core.bank_patch) heap_caps_free(s_core.bank_patch);
        if (s_core.bank_selection_dist) heap_caps_free(s_core.bank_selection_dist);
        s_core.bank = NULL;
        s_core.bank_image = NULL;
        s_core.bank_patch = NULL;
        s_core.bank_selection_dist = NULL;
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t app_ai_coreset_init(void)
{
    esp_err_t ret = ensure_init();
    if (ret != ESP_OK) return ret;
    ret = allocate_bank_storage();
    if (ret != ESP_OK) return ret;
    app_ai_coreset_reset();
    return ESP_OK;
}

void app_ai_coreset_reset(void)
{
    if (ensure_init() != ESP_OK) return;
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    s_core.bank_count = 0;
    s_core.total_vectors = 0;
    s_core.threshold = 0.0f;
    s_core.train_max_score = 0.0f;
    s_core.target_bank = 0;
    s_core.min_bank = 0;
    s_core.source_images = 0;
    s_core.build_us = 0;
    s_core.ready = false;
    memset(s_core.spatial_samples, 0, sizeof(s_core.spatial_samples));
    memset(s_core.appearance_samples, 0, sizeof(s_core.appearance_samples));
    s_core.spatial_count = s_core.spatial_images = 0;
    s_core.spatial_floor = 0.0f;
    s_core.appearance_floor = 0.0f;
    xSemaphoreGive(s_core.lock);
}

/* Tính dmin (khoảng cách ngắn nhất) từ 1 vector kiểm tra tới toàn bộ Memory Bank */
static float candidate_score(const float *candidate,
                             const float *bank,
                             uint32_t bank_count)
{
    float best2 = FLT_MAX;
    uint32_t b = 0;
    while (b < bank_count) {
        const float d2 = dist2_32(candidate, bank + (size_t)b * APP_AI_INFER_FEATURE_DIM);
        if (d2 < best2) best2 = d2;
        b++;
    }
    return best2 == FLT_MAX ? 0.0f : sqrtf(best2);
}

float app_ai_coreset_compute_score(const float *vector)
{
    if (!vector || !s_core.ready || s_core.bank_count == 0) return FLT_MAX;
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    float score = candidate_score(vector, s_core.bank, s_core.bank_count);
    xSemaphoreGive(s_core.lock);
    return score;
}

float app_ai_coreset_compute_spatial_score(const float *vector,
                                            const uint8_t *appearance, uint32_t patch,
                                            float *ai_score, float *appearance_score)
{
    if (!vector || !appearance || !s_core.ready || patch >= s_core.spatial_count) return FLT_MAX;
    float best2 = FLT_MAX;
    const uint8_t mask = s_core.spatial_mask[patch];
    for (uint32_t i = 0; i < s_core.spatial_images; ++i) {
        if (!(mask & (1U << i))) continue;
        const float *normal = s_core.spatial_samples[i] +
            (size_t)patch * APP_AI_INFER_FEATURE_DIM;
        const float d2 = dist2_32(vector, normal);
        if (d2 < best2) best2 = d2;
    }
    float color_best = FLT_MAX;
    const uint8_t color_mask = s_core.appearance_mask[patch];
    for (uint32_t i = 0; i < s_core.spatial_images; ++i) {
        if (!(color_mask & (1U << i))) continue;
        const uint8_t *normal = s_core.appearance_samples[i] +
            (size_t)patch * APP_AI_PATCH_APPEARANCE_DIM;
        float d = appearance_distance(appearance, normal);
        if (d < color_best) color_best = d;
    }
    const float ai = sqrtf(best2) / s_core.spatial_limits[patch];
    const float color = color_best / s_core.appearance_limits[patch];
    if (ai_score) *ai_score = ai;
    if (appearance_score) *appearance_score = color;
    // A strong pixel change with unchanged CNN embedding is evidence that
    // the feature extractor needs checking, not proof of an AI anomaly.
    const float corroborated = fminf(ai / APPEARANCE_AI_MIN_SCORE,
                                     color / APPEARANCE_CONFIRM_SCORE);
    return fmaxf(ai, corroborated);
}

esp_err_t app_ai_coreset_build(const app_ai_coreset_block_t *blocks,
                               uint32_t block_count)
{
    if (!blocks || block_count == 0) return ESP_ERR_INVALID_ARG;
    if (ensure_init() != ESP_OK) return ESP_ERR_NO_MEM;
    if (allocate_bank_storage() != ESP_OK) return ESP_ERR_NO_MEM;

    esp_err_t spatial_ret = build_spatial_bank(blocks, block_count);
    if (spatial_ret != ESP_OK) return spatial_ret;

    uint32_t total = 0;
    uint32_t valid_blocks = 0;
    uint32_t bi = 0;
    while (bi < block_count) {
        if (blocks[bi].vectors && blocks[bi].count > 0) {
            total += blocks[bi].count;
            valid_blocks++;
        }
        bi++;
    }
    if (total == 0) return ESP_ERR_NOT_FOUND;

    float *min_dist2 = heap_caps_malloc((size_t)total * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!min_dist2) min_dist2 = heap_caps_malloc((size_t)total * sizeof(float), MALLOC_CAP_8BIT);
    if (!min_dist2) return ESP_ERR_NO_MEM;

    const float **candidate_vectors = NULL;
    uint32_t *candidate_images = NULL;
    uint32_t *candidate_patches = NULL;
    if (build_candidate_index(blocks, block_count, total, &candidate_vectors, &candidate_images, &candidate_patches) != ESP_OK) {
        heap_caps_free(min_dist2);
        return ESP_ERR_NO_MEM;
    }

    uint32_t i = 0;
    while (i < total) {
        min_dist2[i] = FLT_MAX;
        i++;
    }

    /* Step A: Adaptive Epsilon */
    double anchor_sum = 0.0;
    uint32_t anchor_n = 0;
    i = 1;
    while (i < total) {
        anchor_sum += (double)sqrtf(dist2_32(candidate_vectors[0], candidate_vectors[i]));
        anchor_n++;
        i++;
        if ((i % CORESET_CANDIDATE_CHUNK) == 0) vTaskDelay(1);
    }
    const float d_avg = anchor_n ? (float)(anchor_sum / (double)anchor_n) : 0.0f;
    float epsilon = APP_AI_CORESET_EPSILON_ALPHA * d_avg;
    if (epsilon < APP_AI_CORESET_EPSILON_MIN) epsilon = APP_AI_CORESET_EPSILON_MIN;

    /* Step B: Pass 1 - Greedy K-Center Farthest Point chọn MMB */
    uint32_t ratio_target = (uint32_t)ceilf((float)total * APP_AI_CORESET_RATIO);
    if (ratio_target < 1U) ratio_target = 1U;
    uint32_t target_bank = ratio_target;
    if (target_bank < APP_AI_CORESET_MIN_BANK) target_bank = APP_AI_CORESET_MIN_BANK;
    if (target_bank > APP_AI_CORESET_MAX_BANK) target_bank = APP_AI_CORESET_MAX_BANK;
    if (target_bank > total) target_bank = total;

    uint32_t min_bank = (total < APP_AI_CORESET_MIN_BANK) ? total : APP_AI_CORESET_MIN_BANK;
    if (min_bank > target_bank) min_bank = target_bank;

    const int64_t t0 = esp_timer_get_time();
    uint32_t selected = 0;

    const float *first = candidate_vectors[0];
    memcpy(s_core.bank, first, APP_AI_INFER_FEATURE_DIM * sizeof(float));
    s_core.bank_image[0] = candidate_images[0];
    s_core.bank_patch[0] = candidate_patches[0];
    s_core.bank_selection_dist[0] = 0.0f;
    selected = 1;

    i = 0;
    while (i < total) {
        min_dist2[i] = dist2_32(candidate_vectors[i], s_core.bank);
        i++;
        if ((i % CORESET_CANDIDATE_CHUNK) == 0) vTaskDelay(1);
    }

    while (selected < target_bank) {
        float farthest2 = -1.0f;
        uint32_t farthest_index = 0;
        i = 0;
        while (i < total) {
            if (min_dist2[i] > farthest2) {
                farthest2 = min_dist2[i];
                farthest_index = i;
            }
            i++;
        }
        if (farthest2 < 0.0f) break;

        const float farthest_dist = sqrtf(farthest2);
        if (farthest_dist <= epsilon && selected >= min_bank) {
            break;
        }

        const float *chosen = candidate_vectors[farthest_index];
        const uint32_t chosen_image = candidate_images[farthest_index];
        const uint32_t chosen_patch = candidate_patches[farthest_index];

        memcpy(s_core.bank + (size_t)selected * APP_AI_INFER_FEATURE_DIM,
               chosen,
               APP_AI_INFER_FEATURE_DIM * sizeof(float));
        s_core.bank_image[selected] = chosen_image;
        s_core.bank_patch[selected] = chosen_patch;
        s_core.bank_selection_dist[selected] = farthest_dist;
        selected++;

        const float *new_center = s_core.bank + (size_t)(selected - 1) * APP_AI_INFER_FEATURE_DIM;
        i = 0;
        while (i < total) {
            const float d2 = dist2_32(candidate_vectors[i], new_center);
            if (d2 < min_dist2[i]) min_dist2[i] = d2;
            i++;
            if ((i % CORESET_CANDIDATE_CHUNK) == 0) vTaskDelay(1);
        }

        vTaskDelay(1);
    }

    /* Step C: Pass 2 Calibration - Đo dmin_max của từng ảnh huấn luyện để tính Ngưỡng.
     * Đồng thời gom luôn thống kê phân bố dmin (histogram theo bucket + theo
     * từng ảnh) NGAY TRONG vòng lặp này - dùng lại đúng giá trị `score` vừa
     * tính, không tốn thêm một lượt tính khoảng cách nào nữa. Mục đích: soi
     * xem có tồn tại một "vực" (cliff) giữa nhóm patch gần-trùng-lặp (nền
     * PCB đồng nhất -> dmin rất nhỏ) và nhóm patch thực sự khác biệt (linh
     * kiện, đường mạch, chữ in lụa -> dmin lớn) hay không, để biết ngưỡng
     * lọc ảnh sát 0 là do thuật toán Coreset hoạt động đúng (nền đồng nhất
     * thật) hay do model/patch có vấn đề. */
    float global_max_score = 0.0f;
    uint32_t diag_bucket_floor = 0;   /* dmin < CORESET_MIN_SCORE_FLOOR */
    uint32_t diag_bucket_eps = 0;     /* [floor, EPSILON_MIN) */
    uint32_t diag_bucket_2 = 0;       /* [EPSILON_MIN, 0.05) */
    uint32_t diag_bucket_3 = 0;       /* [0.05, 0.10) */
    uint32_t diag_bucket_4 = 0;       /* [0.10, 0.20) */
    uint32_t diag_bucket_5 = 0;       /* >= 0.20 - khác biệt rõ */
    double diag_sum = 0.0;
    float diag_min = FLT_MAX;
    float diag_max = 0.0f;
    uint32_t diag_near_dup_per_image[CORESET_DIAG_MAX_IMAGES] = {0};
    uint32_t diag_total_per_image[CORESET_DIAG_MAX_IMAGES] = {0};

    uint32_t b = 0;
    while (b < block_count) {
        if (blocks[b].vectors && blocks[b].count) {
            float img_max_dmin = 0.0f;
            uint32_t p = 0;
            while (p < blocks[b].count) {
                const float *v = blocks[b].vectors + (size_t)p * APP_AI_INFER_FEATURE_DIM;
                const float score = candidate_score(v, s_core.bank, selected);
                if (score > img_max_dmin) {
                    img_max_dmin = score;
                }

                diag_sum += (double)score;
                if (score < diag_min) diag_min = score;
                if (score > diag_max) diag_max = score;
                if (score < CORESET_MIN_SCORE_FLOOR) diag_bucket_floor++;
                else if (score < APP_AI_CORESET_EPSILON_MIN) diag_bucket_eps++;
                else if (score < CORESET_DIAG_BUCKET_2) diag_bucket_2++;
                else if (score < CORESET_DIAG_BUCKET_3) diag_bucket_3++;
                else if (score < CORESET_DIAG_BUCKET_4) diag_bucket_4++;
                else diag_bucket_5++;
                if (b < CORESET_DIAG_MAX_IMAGES) {
                    diag_total_per_image[b]++;
                    if (score < APP_AI_CORESET_EPSILON_MIN) diag_near_dup_per_image[b]++;
                }

                p++;
            }
            ESP_LOGI(TAG, "Pass 2 - Ảnh mẫu #%lu: dmin_max = %.6f", (unsigned long)(b + 1), (double)img_max_dmin);
            if (img_max_dmin > global_max_score) {
                global_max_score = img_max_dmin;
            }
        }
        b++;
        vTaskDelay(1);
    }

    float threshold = global_max_score * APP_AI_CORESET_THRESHOLD_MARGIN;
    if (threshold < CORESET_MIN_SCORE_FLOOR) {
        threshold = CORESET_MIN_SCORE_FLOOR;
    }

    /* Log phân bố dmin: nếu phần lớn patch rơi vào bucket "gần trùng" và chỉ
     * một nhóm nhỏ nằm ở bucket "khác biệt rõ" => có "vực" thật, ngưỡng sát 0
     * là hợp lý (nền đồng nhất). Nếu GẦN NHƯ TOÀN BỘ đều rơi vào bucket "gần
     * trùng" kể cả những patch có linh kiện/đường mạch => model/pipeline có
     * vấn đề đáng nghi, cần soi lại ảnh input thay vì tin ngay ngưỡng. */
    ESP_LOGI(TAG, "---- Phân bố dmin: %lu patch training so với MMB (%lu vector) ----",
             (unsigned long)total, (unsigned long)selected);
    ESP_LOGI(TAG, "  min=%.6f  mean=%.6f  max=%.6f",
             (double)(diag_min == FLT_MAX ? 0.0f : diag_min),
             (double)(total ? (float)(diag_sum / (double)total) : 0.0f),
             (double)diag_max);
    ESP_LOGI(TAG, "  < %.3f (sàn ngưỡng)        : %4lu/%-4lu (%5.1f%%)",
             (double)CORESET_MIN_SCORE_FLOOR, (unsigned long)diag_bucket_floor, (unsigned long)total,
             total ? 100.0 * diag_bucket_floor / total : 0.0);
    ESP_LOGI(TAG, "  [%.3f, %.2f) gần trùng lặp : %4lu/%-4lu (%5.1f%%)",
             (double)CORESET_MIN_SCORE_FLOOR, (double)APP_AI_CORESET_EPSILON_MIN,
             (unsigned long)diag_bucket_eps, (unsigned long)total,
             total ? 100.0 * diag_bucket_eps / total : 0.0);
    ESP_LOGI(TAG, "  [%.2f, %.2f)               : %4lu/%-4lu (%5.1f%%)",
             (double)APP_AI_CORESET_EPSILON_MIN, (double)CORESET_DIAG_BUCKET_2,
             (unsigned long)diag_bucket_2, (unsigned long)total,
             total ? 100.0 * diag_bucket_2 / total : 0.0);
    ESP_LOGI(TAG, "  [%.2f, %.2f)               : %4lu/%-4lu (%5.1f%%)",
             (double)CORESET_DIAG_BUCKET_2, (double)CORESET_DIAG_BUCKET_3,
             (unsigned long)diag_bucket_3, (unsigned long)total,
             total ? 100.0 * diag_bucket_3 / total : 0.0);
    ESP_LOGI(TAG, "  [%.2f, %.2f)               : %4lu/%-4lu (%5.1f%%)",
             (double)CORESET_DIAG_BUCKET_3, (double)CORESET_DIAG_BUCKET_4,
             (unsigned long)diag_bucket_4, (unsigned long)total,
             total ? 100.0 * diag_bucket_4 / total : 0.0);
    ESP_LOGI(TAG, "  >= %.2f khác biệt rõ       : %4lu/%-4lu (%5.1f%%)",
             (double)CORESET_DIAG_BUCKET_4, (unsigned long)diag_bucket_5, (unsigned long)total,
             total ? 100.0 * diag_bucket_5 / total : 0.0);

    {
        uint32_t im = 0;
        while (im < block_count && im < CORESET_DIAG_MAX_IMAGES) {
            if (diag_total_per_image[im] > 0) {
                ESP_LOGI(TAG, "  Ảnh #%lu: %lu/%lu patch gần trùng lặp (dmin<%.2f, %.1f%%)",
                         (unsigned long)(im + 1),
                         (unsigned long)diag_near_dup_per_image[im],
                         (unsigned long)diag_total_per_image[im],
                         (double)APP_AI_CORESET_EPSILON_MIN,
                         100.0 * diag_near_dup_per_image[im] / diag_total_per_image[im]);
            }
            im++;
        }
    }

    /* Phân bố NGUỒN ẢNH của các vector được giữ lại trong MMB - nếu dồn gần
     * hết vào 1-2 ảnh, các ảnh còn lại gần như không đóng góp gì cho MMB
     * (có thể do ảnh đó thiếu vùng đặc trưng, hoặc bị mờ/lệch góc/thiếu sáng
     * hơn hẳn các ảnh còn lại). */
    {
        uint32_t bank_per_image[CORESET_DIAG_MAX_IMAGES] = {0};
        uint32_t si = 0;
        while (si < selected) {
            uint32_t img = s_core.bank_image[si];
            if (img < CORESET_DIAG_MAX_IMAGES) bank_per_image[img]++;
            si++;
        }
        ESP_LOGI(TAG, "  Vector MMB theo ảnh nguồn:");
        uint32_t im = 0;
        while (im < block_count && im < CORESET_DIAG_MAX_IMAGES) {
            if (diag_total_per_image[im] > 0) {
                ESP_LOGI(TAG, "    Ảnh #%lu: %lu vector", (unsigned long)(im + 1),
                         (unsigned long)bank_per_image[im]);
            }
            im++;
        }
    }
    ESP_LOGI(TAG, "-------------------------------------------------------------------");

    const int64_t t1 = esp_timer_get_time();
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    s_core.bank_count = selected;
    s_core.total_vectors = total;
    s_core.threshold = 1.0f;
    s_core.train_max_score = global_max_score;
    s_core.target_bank = target_bank;
    s_core.min_bank = min_bank;
    s_core.source_images = valid_blocks;
    s_core.build_us = t1 - t0;
    s_core.epsilon = epsilon;
    s_core.ready = (selected > 0);
    xSemaphoreGive(s_core.lock);

    ESP_LOGI(TAG,
             "Coreset ready: %lu/%lu vectors kept (%.1f%%, ratio=%.2f), "
             "d_avg=%.4f, epsilon=%.4f, global_max_dmin=%.6f, threshold=%.6f, build=%.1f ms",
             (unsigned long)selected,
             (unsigned long)total,
             total ? (100.0f * (float)selected / (float)total) : 0.0f,
             (double)APP_AI_CORESET_RATIO,
             (double)d_avg,
             (double)epsilon,
             (double)global_max_score,
             (double)threshold,
             (double)(t1 - t0) / 1000.0);

    heap_caps_free((void *)candidate_vectors);
    heap_caps_free(candidate_images);
    heap_caps_free(candidate_patches);
    heap_caps_free(min_dist2);
    return ESP_OK;
}

bool app_ai_coreset_is_ready(void)
{
    if (ensure_init() != ESP_OK) return false;
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    bool ready = s_core.ready;
    xSemaphoreGive(s_core.lock);
    return ready;
}

uint32_t app_ai_coreset_get_bank_count(void)
{
    if (ensure_init() != ESP_OK) return 0;
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    uint32_t n = s_core.bank_count;
    xSemaphoreGive(s_core.lock);
    return n;
}

float app_ai_coreset_get_threshold(void)
{
    if (ensure_init() != ESP_OK) return 0.0f;
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    float v = s_core.threshold;
    xSemaphoreGive(s_core.lock);
    return v;
}

esp_err_t app_ai_coreset_get_vector(uint32_t index,
                                    float *out,
                                    size_t out_len,
                                    uint32_t *source_image,
                                    uint32_t *source_patch,
                                    float *selection_distance)
{
    if (!out || out_len < APP_AI_INFER_FEATURE_DIM) return ESP_ERR_INVALID_ARG;
    if (ensure_init() != ESP_OK) return ESP_ERR_INVALID_STATE;
    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    if (!s_core.ready || index >= s_core.bank_count) {
        xSemaphoreGive(s_core.lock);
        return ESP_ERR_NOT_FOUND;
    }
    memcpy(out, s_core.bank + (size_t)index * APP_AI_INFER_FEATURE_DIM,
           APP_AI_INFER_FEATURE_DIM * sizeof(float));
    if (source_image) *source_image = s_core.bank_image[index];
    if (source_patch) *source_patch = s_core.bank_patch[index];
    if (selection_distance) *selection_distance = s_core.bank_selection_dist[index];
    xSemaphoreGive(s_core.lock);
    return ESP_OK;
}

esp_err_t app_ai_coreset_get_status_json(char *buf, size_t len)
{
    if (!buf || len == 0 || ensure_init() != ESP_OK) return ESP_ERR_INVALID_ARG;
    cJSON *root = cJSON_CreateObject();
    if (!root) return ESP_ERR_NO_MEM;

    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    cJSON_AddBoolToObject(root, "ready", s_core.ready);
    cJSON_AddNumberToObject(root, "bank_count", s_core.bank_count);
    cJSON_AddNumberToObject(root, "max_bank_count", APP_AI_CORESET_MAX_BANK);
    cJSON_AddNumberToObject(root, "min_bank_count", s_core.ready ? s_core.min_bank : APP_AI_CORESET_MIN_BANK);
    cJSON_AddNumberToObject(root, "total_vectors", s_core.total_vectors);
    cJSON_AddNumberToObject(root, "source_images", s_core.source_images);
    cJSON_AddNumberToObject(root, "feature_dim", APP_AI_INFER_FEATURE_DIM);
    cJSON_AddNumberToObject(root, "epsilon", s_core.epsilon);
    cJSON_AddNumberToObject(root, "train_max_score", s_core.train_max_score);
    cJSON_AddNumberToObject(root, "threshold", s_core.threshold);
    cJSON_AddNumberToObject(root, "coreset_ratio", APP_AI_CORESET_RATIO);
    cJSON_AddNumberToObject(root, "target_bank_count",
                            s_core.ready ? s_core.target_bank :
                            ((s_core.total_vectors < APP_AI_CORESET_MAX_BANK) ? s_core.total_vectors : APP_AI_CORESET_MAX_BANK));
    cJSON_AddNumberToObject(root, "kept_percent",
                            s_core.total_vectors ? (100.0 * (double)s_core.bank_count / (double)s_core.total_vectors) : 0.0);
    cJSON_AddNumberToObject(root, "reduction_percent",
                            s_core.total_vectors ? (100.0 * (1.0 - (double)s_core.bank_count / (double)s_core.total_vectors)) : 0.0);
    cJSON_AddNumberToObject(root, "build_ms", (double)s_core.build_us / 1000.0);
    cJSON_AddNumberToObject(root, "epsilon_alpha", APP_AI_CORESET_EPSILON_ALPHA);
    cJSON_AddStringToObject(root, "algorithm", "Greedy Farthest-Point k-center");
    cJSON_AddStringToObject(root, "distance", "Euclidean");
    cJSON_AddStringToObject(root, "threshold_rule", "spatial leave-one-out, score > 1");
    cJSON_AddStringToObject(root, "decision_rule", "AI embedding primary; color confirms AI >= 0.65");
    cJSON_AddNumberToObject(root, "spatial_positions", s_core.spatial_count);
    cJSON_AddNumberToObject(root, "spatial_noise_floor", s_core.spatial_floor);
    cJSON_AddNumberToObject(root, "appearance_noise_floor", s_core.appearance_floor);

    cJSON *entries = cJSON_AddArrayToObject(root, "entries");
    if (entries && s_core.ready) {
        uint32_t i = 0;
        while (i < s_core.bank_count) {
            cJSON *e = cJSON_CreateObject();
            if (e) {
                cJSON_AddNumberToObject(e, "index", i);
                cJSON_AddNumberToObject(e, "source_image", s_core.bank_image[i] + 1);
                cJSON_AddNumberToObject(e, "source_patch", s_core.bank_patch[i] + 1);
                cJSON_AddNumberToObject(e, "selection_distance", s_core.bank_selection_dist[i]);
                cJSON *pv = cJSON_AddArrayToObject(e, "preview_vector");
                if (pv) {
                    uint32_t d = 0;
                    const float *v = s_core.bank + (size_t)i * APP_AI_INFER_FEATURE_DIM;
                    while (d < APP_AI_CORESET_PREVIEW_DIMS && d < APP_AI_INFER_FEATURE_DIM) {
                        cJSON_AddItemToArray(pv, cJSON_CreateNumber(v[d]));
                        d++;
                    }
                }
                cJSON_AddItemToArray(entries, e);
            }
            i++;
        }
    }
    bool ok = cJSON_PrintPreallocated(root, buf, (int)len, false);
    cJSON_Delete(root);
    xSemaphoreGive(s_core.lock);
    return ok ? ESP_OK : ESP_ERR_NO_MEM;
}

esp_err_t app_ai_coreset_get_bank_json(char *buf, size_t len)
{
    if (!buf || len == 0 || ensure_init() != ESP_OK) return ESP_ERR_INVALID_ARG;
    cJSON *root = cJSON_CreateObject();
    if (!root) return ESP_ERR_NO_MEM;

    xSemaphoreTake(s_core.lock, portMAX_DELAY);
    cJSON_AddStringToObject(root, "algorithm", "Greedy Farthest-Point k-center");
    cJSON_AddNumberToObject(root, "feature_dim", APP_AI_INFER_FEATURE_DIM);
    cJSON_AddNumberToObject(root, "bank_count", s_core.bank_count);
    cJSON_AddNumberToObject(root, "total_vectors", s_core.total_vectors);
    cJSON_AddNumberToObject(root, "epsilon", s_core.epsilon);
    cJSON_AddNumberToObject(root, "epsilon_alpha", APP_AI_CORESET_EPSILON_ALPHA);
    cJSON_AddNumberToObject(root, "threshold", s_core.threshold);
    cJSON_AddNumberToObject(root, "coreset_ratio", APP_AI_CORESET_RATIO);

    cJSON *entries = cJSON_AddArrayToObject(root, "memory_bank");
    if (entries && s_core.ready) {
        uint32_t i = 0;
        while (i < s_core.bank_count) {
            cJSON *e = cJSON_CreateObject();
            if (e) {
                cJSON_AddNumberToObject(e, "index", i);
                cJSON_AddNumberToObject(e, "source_image", s_core.bank_image[i] + 1);
                cJSON_AddNumberToObject(e, "source_patch", s_core.bank_patch[i] + 1);
                cJSON_AddNumberToObject(e, "selection_distance", s_core.bank_selection_dist[i]);
                cJSON *v = cJSON_CreateFloatArray(s_core.bank + (size_t)i * APP_AI_INFER_FEATURE_DIM,
                                                   APP_AI_INFER_FEATURE_DIM);
                if (v) cJSON_AddItemToObject(e, "vector", v);
                cJSON_AddItemToArray(entries, e);
            }
            i++;
        }
    }
    bool ok = cJSON_PrintPreallocated(root, buf, (int)len, false);
    cJSON_Delete(root);
    xSemaphoreGive(s_core.lock);
    return ok ? ESP_OK : ESP_ERR_NO_MEM;
}
