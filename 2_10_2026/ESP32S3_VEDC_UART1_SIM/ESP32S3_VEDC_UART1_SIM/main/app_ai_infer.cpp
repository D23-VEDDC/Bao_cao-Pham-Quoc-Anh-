/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstring>
#include <cstdio>
#include <new>

#include "app_ai_infer.h"
#include "app_ai_patch.h"
#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_task_wdt.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "app_ai_infer";

extern const uint8_t feature_extractor_32d_full_int8_tflite_start[] asm(
    "_binary_feature_extractor_32d_full_int8_tflite_start");
extern const uint8_t feature_extractor_32d_full_int8_tflite_end[] asm(
    "_binary_feature_extractor_32d_full_int8_tflite_end");

#define APP_AI_INFER_NUM_OPS 30
#define APP_AI_INFER_ARENA_BYTES (160 * 1024)

#ifndef APP_AI_INFER_DEBUG_DIAG
#define APP_AI_INFER_DEBUG_DIAG 0
#endif

/*
 * In toàn bộ trace OP.
 * 1 = log tất cả OP
 * 0 = chỉ log OP0..OP5 + các điểm collapse
 */
#ifndef APP_AI_INFER_DEBUG_TRACE_ALL
#define APP_AI_INFER_DEBUG_TRACE_ALL 1
#endif

/*
 * DEDUP must be OFF while validating feature extraction. Reusing a vector
 * from the previous patch can hide whether the model itself produces a
 * duplicate embedding. Enable only after the ESP32 output has been
 * verified against the reference/Python pipeline.
 */
#ifndef APP_AI_INFER_ENABLE_DEDUP
#define APP_AI_INFER_ENABLE_DEDUP 0
#endif

/* Lightweight real-patch I/O diagnostics. Unlike APP_AI_INFER_DEBUG_DIAG,
 * this does NOT enable preserve_all_tensors or the black/white diagnostic. */
#ifndef APP_AI_INFER_DEBUG_PATCH_IO
#define APP_AI_INFER_DEBUG_PATCH_IO 0
#endif

#ifndef APP_AI_INFER_DEBUG_SINGLE_PATCH
#define APP_AI_INFER_DEBUG_SINGLE_PATCH 0
#endif

#define APP_AI_INFER_DEBUG_PATCH_INDEX 13

/*
 * OP cần soi sâu trong diagnostic cũ. Các log mới không còn giả định OP5
 * là điểm collapse; duplicate được kiểm tra trực tiếp ở input/output patch.
 */
#define APP_AI_DEBUG_TARGET_OP 5

#define APP_AI_DEBUG_MAX_TENSORS 256

#define APP_AI_INVOKE_WARN_US 1000000

namespace {
const tflite::Model *s_model = nullptr;
tflite::MicroInterpreter *s_interpreter = nullptr;
TfLiteTensor *s_input = nullptr;
TfLiteTensor *s_output = nullptr;
uint8_t *s_arena = nullptr;

#if APP_AI_INFER_ENABLE_DEDUP
uint8_t *s_cached_input = nullptr;
size_t s_cached_input_bytes = 0;
float s_cached_output[APP_AI_INFER_FEATURE_DIM];
bool s_cached_output_valid = false;
uint32_t s_cached_input_patch = 0;
#endif

alignas(16) uint8_t s_interpreter_storage[
    sizeof(tflite::MicroInterpreter)];
}

typedef struct {
    float raw[APP_AI_INFER_FEATURE_DIM];
    float values[APP_AI_INFER_FEATURE_DIM];
} app_ai_infer_vector_t;

static struct {
    SemaphoreHandle_t lock;
    bool ready;

    app_ai_infer_vector_t *vectors;
    uint32_t count;

    bool last_run_ok;
    char last_error[48];

    int64_t last_run_us;

    uint32_t prev_raw_sum;
    uint32_t prev_input_hash;
    float prev_raw[APP_AI_INFER_FEATURE_DIM];
    bool prev_raw_valid;
} s_ai_infer;

#if APP_AI_INFER_DEBUG_PATCH_IO
static uint32_t s_debug_rgb_hashes[APP_AI_PATCH_MAX_COUNT];
static uint32_t s_debug_input_hashes[APP_AI_PATCH_MAX_COUNT];
static uint32_t s_debug_raw_sums[APP_AI_PATCH_MAX_COUNT];
#endif

/* BƯỚC 2: Thêm buffer giữ nguyên patch 13 lúc đang chạy */
#if APP_AI_INFER_DEBUG_DIAG
static uint8_t s_debug_patch13[APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS];
static bool s_debug_patch13_valid = false;
#endif

/* ================================================================
 * WDT
 * ================================================================ */

static void ai_wdt_feed_and_yield(void)
{
    esp_err_t status = esp_task_wdt_status(NULL);

    if (status == ESP_OK) {
        esp_task_wdt_reset();
    }

    /*
     * Cho IDLE0 chạy. FreeRTOS tick = 1 kHz nên vTaskDelay(1) chỉ nghỉ 1 ms,
     * không đủ khi UVC/USB/WebSocket (prio 5) đang chiếm CPU0 -> IDLE0 bị
     * đói và task watchdog nổ. Nghỉ 20 ms sau mỗi patch (~4 s cho 192 patch).
     */
    vTaskDelay(pdMS_TO_TICKS(20));
}

/* ================================================================
 * INPUT CHECKSUM
 * ================================================================ */

#if APP_AI_INFER_DEBUG_DIAG

static uint32_t calc_input_checksum(void)
{
    if (!s_input || !s_input->data.raw) {
        return 0;
    }

    uint32_t hash = 2166136261u;
    size_t bytes = s_input->bytes;

    for (size_t i = 0; i < bytes; i++) {
        hash ^= (uint8_t)s_input->data.raw[i];
        hash *= 16777619u;
    }

    return hash;
}

#endif

/* ================================================================
 * PIXEL -> INT8 LUT
 * ================================================================ */

static int8_t s_pixel_int8_lut[256];
static bool s_pixel_int8_lut_ready = false;

static void build_pixel_int8_lut(void)
{
    ESP_LOGE(TAG, "[INPUT QPARAM] type=%d scale=%.10f zero_point=%ld",
             s_input->type,
             (double)s_input->params.scale,
             (long)s_input->params.zero_point);

    float scale = 0.0029373318f;
    int32_t zero_point = -128;

    if (s_input) {
        if (s_input->params.scale > 0.0f) {
            scale = s_input->params.scale;
        }
        zero_point = s_input->params.zero_point;
    }

    for (int v = 0; v < 256; v++) {
        float real_value = (float)v / 255.0f;
        int32_t q = (int32_t)lroundf(real_value / scale) + zero_point;

        if (q < -128) q = -128;
        if (q > 127)  q = 127;

        s_pixel_int8_lut[v] = (int8_t)q;
    }

    s_pixel_int8_lut_ready = true;

    ESP_LOGE(TAG, "[QMAP] p=0   -> q=%d", s_pixel_int8_lut[0]);
    ESP_LOGE(TAG, "[QMAP] p=128 -> q=%d", s_pixel_int8_lut[128]);
    ESP_LOGE(TAG, "[QMAP] p=160 -> q=%d", s_pixel_int8_lut[160]);
    ESP_LOGE(TAG, "[QMAP] p=180 -> q=%d", s_pixel_int8_lut[180]);
    ESP_LOGE(TAG, "[QMAP] p=190 -> q=%d", s_pixel_int8_lut[190]);
    ESP_LOGE(TAG, "[QMAP] p=200 -> q=%d", s_pixel_int8_lut[200]);
    ESP_LOGE(TAG, "[QMAP] p=220 -> q=%d", s_pixel_int8_lut[220]);
    ESP_LOGE(TAG, "[QMAP] p=255 -> q=%d", s_pixel_int8_lut[255]);

    ESP_LOGI(TAG, "[AI LUT] built scale=%.9f zp=%d", (double)scale, (int)zero_point);
}

static inline void write_input_pixel(size_t i, uint8_t pixel)
{
    switch (s_input->type) {

    case kTfLiteInt8:
        if (s_pixel_int8_lut_ready) {
            s_input->data.int8[i] = s_pixel_int8_lut[pixel];
        } else {
            float scale = (s_input->params.scale > 0.0f) ? s_input->params.scale : 0.0029373318f;
            int32_t zero_point = s_input->params.zero_point;
            float real_value = (float)pixel / 255.0f;

            int32_t q = (int32_t)lroundf(real_value / scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;

            s_input->data.int8[i] = (int8_t)q;
        }
        break;

    case kTfLiteUInt8:
        s_input->data.uint8[i] = pixel;
        break;

    case kTfLiteFloat32:
    default:
        s_input->data.f[i] = (float)pixel / 255.0f;
        break;
    }
}

/* ================================================================
 * OUTPUT READER
 * ================================================================ */

static inline float read_output_value(size_t i)
{
    switch (s_output->type) {

    case kTfLiteFloat32:
        return s_output->data.f[i];

    case kTfLiteInt8: {
        float scale = (s_output->params.scale > 0.0f) ? s_output->params.scale : 0.02f;
        return ((float)s_output->data.int8[i] - (float)s_output->params.zero_point) * scale;
    }

    case kTfLiteUInt8: {
        float scale = (s_output->params.scale > 0.0f) ? s_output->params.scale : 0.02f;
        return ((float)s_output->data.uint8[i] - (float)s_output->params.zero_point) * scale;
    }

    case kTfLiteInt16: {
        float scale = (s_output->params.scale > 0.0f) ? s_output->params.scale : 0.02f;
        return ((float)s_output->data.i16[i] - (float)s_output->params.zero_point) * scale;
    }

    case kTfLiteInt32: {
        float scale = (s_output->params.scale > 0.0f) ? s_output->params.scale : 0.02f;
        return ((float)s_output->data.i32[i] - (float)s_output->params.zero_point) * scale;
    }

    case kTfLiteInt64: {
        float scale = (s_output->params.scale > 0.0f) ? s_output->params.scale : 0.02f;
        return ((float)s_output->data.i64[i] - (float)s_output->params.zero_point) * scale;
    }

    default:
        return NAN;
    }
}

#if APP_AI_INFER_DEBUG_DIAG

/* ================================================================
 * TENSOR DEBUG BASIC
 * ================================================================ */

static size_t debug_tensor_elements(const TfLiteEvalTensor *t)
{
    if (!t || !t->dims) return 0;

    ssize_t n = 1;
    for (int i = 0; i < t->dims->size; i++) {
        if (t->dims->data[i] <= 0) return 0;
        n *= (size_t)t->dims->data[i];
    }
    return n;
}

static size_t debug_tensor_element_bytes(TfLiteType type)
{
    switch (type) {
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteBool:
        return 1;
    case kTfLiteInt16:
        return 2;
    case kTfLiteInt32:
    case kTfLiteFloat32:
        return 4;
    case kTfLiteInt64:
    case kTfLiteFloat64:
        return 8;
    default:
        return 0;
    }
}

static uint32_t debug_tensor_hash(const TfLiteEvalTensor *t)
{
    if (!t || !t->data.raw) return 0;

    size_t n = debug_tensor_elements(t);
    size_t elem_bytes = debug_tensor_element_bytes(t->type);

    if (n == 0 || elem_bytes == 0) return 0;

    const uint8_t *p = reinterpret_cast<const uint8_t *>(t->data.raw);
    size_t bytes = n * elem_bytes;
    uint32_t hash = 2166136261u;

    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= 16777619u;
    }

    return hash;
}

static const char *debug_tensor_type_name(TfLiteType type)
{
    switch (type) {
    case kTfLiteFloat32: return "F32";
    case kTfLiteFloat64: return "F64";
    case kTfLiteInt8:    return "I8";
    case kTfLiteUInt8:   return "U8";
    case kTfLiteInt16:   return "I16";
    case kTfLiteInt32:   return "I32";
    case kTfLiteInt64:   return "I64";
    case kTfLiteBool:    return "BOOL";
    default:             return "OTHER";
    }
}

static void debug_tensor_shape_string(const TfLiteEvalTensor *t, char *buf, size_t buf_len)
{
    if (!buf || buf_len == 0) return;
    buf[0] = '\0';

    if (!t || !t->dims) {
        snprintf(buf, buf_len, "?");
        return;
    }

    size_t used = 0;
    for (int i = 0; i < t->dims->size; i++) {
        int wrote = snprintf(buf + used, used < buf_len ? buf_len - used : 0,
                             "%s%d", i == 0 ? "" : "x", t->dims->data[i]);
        if (wrote < 0) break;
        used += (size_t)wrote;
        if (used >= buf_len) {
            buf[buf_len - 1] = '\0';
            break;
        }
    }
}

static float debug_tensor_get_float(const TfLiteEvalTensor *t, size_t i)
{
    switch (t->type) {
    case kTfLiteFloat32: return t->data.f[i];
    case kTfLiteInt8:    return (float)t->data.int8[i];
    case kTfLiteUInt8:   return (float)t->data.uint8[i];
    case kTfLiteInt16:   return (float)t->data.i16[i];
    case kTfLiteInt32:   return (float)t->data.i32[i];
    case kTfLiteInt64:   return (float)t->data.i64[i];
    case kTfLiteBool:    return t->data.b[i] ? 1.0f : 0.0f;
    default:             return 0.0f;
    }
}

/* ================================================================
 * TENSOR STATISTICS
 * ================================================================ */

typedef struct {
    bool valid;

    uint32_t hash;
    size_t elements;

    float min_value;
    float max_value;
    float mean_value;

    uint32_t nonzero;

    float first8[8];
} debug_tensor_stats_t;

static void debug_collect_tensor_stats(int tensor_index, debug_tensor_stats_t *stats)
{
    if (!stats) return;

    memset(stats, 0, sizeof(*stats));
    if (!s_interpreter) return;

    TfLiteEvalTensor *t = s_interpreter->GetTensor(tensor_index);
    if (!t || !t->data.raw) return;

    size_t n = debug_tensor_elements(t);
    if (n == 0) return;

    stats->valid = true;
    stats->elements = n;
    stats->hash = debug_tensor_hash(t);

    float min_v = debug_tensor_get_float(t, 0);
    float max_v = min_v;
    double sum = 0.0;
    uint32_t nonzero = 0;
    size_t sample_n = n < 8 ? n : 8;

    for (size_t i = 0; i < n; i++) {
        float v = debug_tensor_get_float(t, i);

        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += (double)v;

        if (v != 0.0f) nonzero++;
        if (i < sample_n) stats->first8[i] = v;
    }

    stats->min_value = min_v;
    stats->max_value = max_v;
    stats->mean_value = (float)(sum / (double)n);
    stats->nonzero = nonzero;
}

static void debug_log_saturation(int tensor_index, const char *label)
{
    if (!s_interpreter) return;

    TfLiteEvalTensor *t = s_interpreter->GetTensor(tensor_index);
    if (!t || !t->data.int8 || t->type != kTfLiteInt8) return;

    size_t n = debug_tensor_elements(t);
    if (n == 0) return;

    uint32_t sat_min = 0;
    uint32_t sat_max = 0;

    for (size_t i = 0; i < n; i++) {
        if (t->data.int8[i] == -128) sat_min++;
        if (t->data.int8[i] == 127) sat_max++;
    }

    ESP_LOGE(TAG,
             "[SAT %s] T%d total=%u min(-128)=%u %.1f%% max(127)=%u %.1f%% total_sat=%u %.1f%%",
             label,
             tensor_index,
             (unsigned)n,
             (unsigned)sat_min,
             100.0 * (double)sat_min / (double)n,
             (unsigned)sat_max,
             100.0 * (double)sat_max / (double)n,
             (unsigned)(sat_min + sat_max),
             100.0 * (double)(sat_min + sat_max) / (double)n);
}

/* ================================================================
 * LOG TENSOR
 * ================================================================ */

static void debug_log_tensor(int tensor_index, const char *tag, bool log_samples)
{
    if (!s_interpreter) return;

    TfLiteEvalTensor *t = s_interpreter->GetTensor(tensor_index);
    if (!t) {
        ESP_LOGE(TAG, "[TENSOR %s] T%d=NULL", tag, tensor_index);
        return;
    }

    debug_tensor_stats_t stats;
    debug_collect_tensor_stats(tensor_index, &stats);

    char shape[48];
    debug_tensor_shape_string(t, shape, sizeof(shape));

    ESP_LOGI(TAG,
        "[TENSOR %s] T%d type=%s "
        "shape=%s n=%u hash=0x%08X "
        "min=%.5f max=%.5f mean=%.5f "
        "nz=%u/%u",
        tag,
        tensor_index,
        debug_tensor_type_name(t->type),
        shape,
        (unsigned)stats.elements,
        (unsigned)stats.hash,
        (double)stats.min_value,
        (double)stats.max_value,
        (double)stats.mean_value,
        (unsigned)stats.nonzero,
        (unsigned)stats.elements);

    if (log_samples) {
        ESP_LOGI(TAG,
            "[TENSOR %s] T%d first8="
            "[%.4f %.4f %.4f %.4f "
            "%.4f %.4f %.4f %.4f]",
            tag,
            tensor_index,
            (double)stats.first8[0],
            (double)stats.first8[1],
            (double)stats.first8[2],
            (double)stats.first8[3],
            (double)stats.first8[4],
            (double)stats.first8[5],
            (double)stats.first8[6],
            (double)stats.first8[7]);
    }
}

/* ================================================================
 * OP DETAILS
 * ================================================================ */

static void debug_log_op_detail(unsigned op_index, const char *tag)
{
    if (!s_model || !s_interpreter || !s_model->subgraphs()) return;

    const auto *sg = s_model->subgraphs()->Get(0);
    const auto *ops = sg->operators();

    if (!ops || op_index >= ops->size()) return;

    const auto *op = ops->Get(op_index);
    uint32_t opcode_index = op->opcode_index();
    const auto *opcode = s_model->operator_codes()->Get(opcode_index);
    const char *name = tflite::EnumNameBuiltinOperator(opcode->builtin_code());

    ESP_LOGE(TAG, "================================================");
    ESP_LOGE(TAG, "[OP%u DETAIL %s] %s", op_index, tag, name ? name : "UNKNOWN");
    ESP_LOGE(TAG, "[OP%u DETAIL] inputs=%u outputs=%u", op_index,
             (unsigned)op->inputs()->size(), (unsigned)op->outputs()->size());

    /* Log tất cả input tensor */
    for (unsigned i = 0; i < op->inputs()->size(); i++) {
        int tid = op->inputs()->Get(i);
        if (tid < 0 || tid >= APP_AI_DEBUG_MAX_TENSORS) continue;

        ESP_LOGE(TAG, "[OP%u INPUT%u] tensor=T%d", op_index, i, tid);
        debug_log_tensor(tid, i == 0 ? "OP_INPUT" : "OP_PARAM", true);
    }

    /* Log output */
    for (unsigned i = 0; i < op->outputs()->size(); i++) {
        int tid = op->outputs()->Get(i);
        if (tid < 0 || tid >= APP_AI_DEBUG_MAX_TENSORS) continue;

        ESP_LOGE(TAG, "[OP%u OUTPUT%u] tensor=T%d", op_index, i, tid);
        debug_log_tensor(tid, "OP_OUTPUT", true);
    }

    ESP_LOGE(TAG, "================================================");
}

/* ================================================================
 * MODEL GRAPH
 * ================================================================ */

static void debug_dump_model_graph(void)
{
#if APP_AI_INFER_DEBUG_SINGLE_PATCH
    ESP_LOGI(TAG, "[AI DEBUG] skipped model graph dump for single-patch run");
    return;
#endif

    if (!s_model || !s_model->subgraphs() || s_model->subgraphs()->size() == 0) return;

    const auto *sg = s_model->subgraphs()->Get(0);
    const auto *ops = sg->operators();
    if (!ops) return;

    ESP_LOGI(TAG, "================ MODEL GRAPH ================");
    ESP_LOGI(TAG, "tensors=%u ops=%u input=%d output=%d",
             (unsigned)sg->tensors()->size(),
             (unsigned)ops->size(),
             (int)sg->inputs()->Get(0),
             (int)sg->outputs()->Get(0));

    for (unsigned op_index = 0; op_index < ops->size(); op_index++) {
        const auto *op = ops->Get(op_index);
        uint32_t opcode_index = op->opcode_index();
        const auto *opcode = s_model->operator_codes()->Get(opcode_index);
        tflite::BuiltinOperator builtin = opcode->builtin_code();
        const char *name = tflite::EnumNameBuiltinOperator(builtin);

        char in_buf[96];
        char out_buf[64];
        in_buf[0] = '\0';
        out_buf[0] = '\0';

        size_t used_in = 0;
        size_t used_out = 0;

        for (unsigned j = 0; j < op->inputs()->size(); j++) {
            int32_t tid = op->inputs()->Get(j);
            int wrote = snprintf(in_buf + used_in,
                                 used_in < sizeof(in_buf) ? sizeof(in_buf) - used_in : 0,
                                 "%s%d", j ? "," : "", (int)tid);
            if (wrote < 0) break;
            used_in += (size_t)wrote;
            if (used_in >= sizeof(in_buf)) {
                in_buf[sizeof(in_buf) - 1] = '\0';
                break;
            }
        }

        for (unsigned j = 0; j < op->outputs()->size(); j++) {
            int32_t tid = op->outputs()->Get(j);
            int wrote = snprintf(out_buf + used_out,
                                 used_out < sizeof(out_buf) ? sizeof(out_buf) - used_out : 0,
                                 "%s%d", j ? "," : "", (int)tid);
            if (wrote < 0) break;
            used_out += (size_t)wrote;
            if (used_out >= sizeof(out_buf)) {
                out_buf[sizeof(out_buf) - 1] = '\0';
                break;
            }
        }

        ESP_LOGI(TAG, "[GRAPH] OP%u %-22s in=[%s] out=[%s]",
                 op_index, name ? name : "UNKNOWN", in_buf, out_buf);

        if ((op_index & 0x07) == 0) {
            vTaskDelay(1);
        }
    }

    ESP_LOGI(TAG, "==============================================");
}

/* ================================================================
 * TRACE CAPTURE
 * ================================================================ */

static uint32_t debug_capture_operator_output_hashes(
    uint32_t *hashes,
    size_t hash_count,
    const char *tag)
{
    if (!hashes || hash_count == 0 || !s_model || !s_interpreter || !s_model->subgraphs()) return 0;

    memset(hashes, 0, sizeof(uint32_t) * hash_count);

    const auto *sg = s_model->subgraphs()->Get(0);
    const auto *ops = sg->operators();
    if (!ops) return 0;

    uint32_t captured = 0;

    for (unsigned op_index = 0; op_index < ops->size(); op_index++) {
        const auto *op = ops->Get(op_index);

        for (unsigned out_index = 0; out_index < op->outputs()->size(); out_index++) {
            int tensor_index = op->outputs()->Get(out_index);
            if (tensor_index < 0 || (size_t)tensor_index >= hash_count) continue;

            TfLiteEvalTensor *tensor = s_interpreter->GetTensor(tensor_index);
            hashes[tensor_index] = debug_tensor_hash(tensor);
            captured++;
        }
    }

    for (unsigned op_index = 0; op_index < ops->size(); op_index++) {
        const auto *op = ops->Get(op_index);
        uint32_t opcode_index = op->opcode_index();
        const auto *opcode = s_model->operator_codes()->Get(opcode_index);
        const char *name = tflite::EnumNameBuiltinOperator(opcode->builtin_code());

        for (unsigned out_index = 0; out_index < op->outputs()->size(); out_index++) {
            int tensor_index = op->outputs()->Get(out_index);
            if (tensor_index < 0 || (size_t)tensor_index >= hash_count) {
                ESP_LOGW(TAG, "[TRACE %s] OP%u %s -> T%d SKIPPED trace_limit=%u",
                         tag, op_index, name ? name : "UNKNOWN", tensor_index, (unsigned)hash_count);
                continue;
            }

#if APP_AI_INFER_DEBUG_TRACE_ALL
            ESP_LOGI(TAG, "[TRACE %s] OP%u %-18s -> T%d hash=0x%08X",
                     tag, op_index, name ? name : "UNKNOWN", tensor_index, (unsigned)hashes[tensor_index]);
#else
            if (op_index <= APP_AI_DEBUG_TARGET_OP) {
                ESP_LOGI(TAG, "[TRACE %s] OP%u %-18s -> T%d hash=0x%08X",
                         tag, op_index, name ? name : "UNKNOWN", tensor_index, (unsigned)hashes[tensor_index]);
            }
#endif
        }

        if ((op_index & 0x07) == 0) {
            vTaskDelay(1);
        }
    }

    return captured;
}

/* ================================================================
 * OP5 QUANTIZATION & SATURATION LOG
 * ================================================================ */

static void debug_log_tensor_qparams(int tensor_index, const char *name)
{
    if (!s_model || !s_model->subgraphs()) return;

    const auto *sg = s_model->subgraphs()->Get(0);
    if (!sg->tensors() || tensor_index < 0 || (size_t)tensor_index >= sg->tensors()->size()) return;

    const auto *t_fb = sg->tensors()->Get(tensor_index);
    float scale = 0.0f;
    int64_t zp = 0;

    if (t_fb->quantization()) {
        if (t_fb->quantization()->scale() && t_fb->quantization()->scale()->size() > 0) {
            scale = t_fb->quantization()->scale()->Get(0);
        }
        if (t_fb->quantization()->zero_point() && t_fb->quantization()->zero_point()->size() > 0) {
            zp = t_fb->quantization()->zero_point()->Get(0);
        }
    }

    ESP_LOGE(TAG, "[QPARAM %s] T%d scale=%.10f, zp=%lld", name, tensor_index, (double)scale, (long long)zp);
}

static void debug_log_tensor_checkpoint(int tensor_index, const char *stage)
{
    if (!s_interpreter || !s_model || !s_model->subgraphs()) return;

    const auto *subgraph = s_model->subgraphs()->Get(0);
    if (!subgraph || !subgraph->tensors() || tensor_index < 0 ||
        (size_t)tensor_index >= subgraph->tensors()->size()) {
        ESP_LOGE(TAG, "[CHECKPOINT %s] T%d outside model tensor range", stage, tensor_index);
        return;
    }

    TfLiteEvalTensor *tensor = s_interpreter->GetTensor(tensor_index);
    if (!tensor) {
        ESP_LOGE(TAG, "[CHECKPOINT %s] T%d=NULL", stage, tensor_index);
        return;
    }

    char shape[48];
    debug_tensor_shape_string(tensor, shape, sizeof(shape));
    size_t bytes = debug_tensor_elements(tensor) * debug_tensor_element_bytes(tensor->type);
    ESP_LOGI(TAG, "[CHECKPOINT %s] T%d tensor=%p data=%p bytes=%u type=%s dims=%s",
             stage, tensor_index, (void *)tensor, (void *)tensor->data.raw,
             (unsigned)bytes, debug_tensor_type_name(tensor->type), shape);
    debug_log_tensor(tensor_index, stage, false);
    debug_log_tensor_qparams(tensor_index, stage);
    debug_log_saturation(tensor_index, stage);
}

static void debug_log_op5_qparams_and_saturation(void)
{
    if (!s_interpreter) return;

    int idx_in  = 114; // T114 (Input)
    int idx_w   = 99;  // T99  (Weight)
    int idx_b   = 98;  // T98  (Bias)
    int idx_out = 115; // T115 (Output)

    ESP_LOGE(TAG, "================ OP5 QPARAM & SATURATION ================");
    debug_log_tensor_qparams(idx_in,  "T114 IN ");
    debug_log_tensor_qparams(idx_w,   "T99  W  ");
    debug_log_tensor_qparams(idx_b,   "T98  B  ");
    debug_log_tensor_qparams(idx_out, "T115 OUT");

    TfLiteEvalTensor *t_out = s_interpreter->GetTensor(idx_out);
    if (t_out && t_out->type == kTfLiteInt8 && t_out->data.int8) {
        size_t n = debug_tensor_elements(t_out);
        if (n > 0) {
            uint32_t sat_min = 0, sat_max = 0;
            for (size_t i = 0; i < n; i++) {
                if (t_out->data.int8[i] == -128) sat_min++;
                else if (t_out->data.int8[i] == 127) sat_max++;
            }

            ESP_LOGE(TAG, "[T115 SATURATION] total=%u | sat_min(-128)=%u (%.1f%%) | sat_max(127)=%u (%.1f%%) | sat_total=%u (%.1f%%)",
                     (unsigned)n,
                     (unsigned)sat_min, 100.0 * sat_min / n,
                     (unsigned)sat_max, 100.0 * sat_max / n,
                     (unsigned)(sat_min + sat_max), 100.0 * (sat_min + sat_max) / n);
        }
    }
    ESP_LOGE(TAG, "=========================================================");
}

/* ================================================================
 * OP5 BLACK/WHITE DETAIL
 * ================================================================ */

static void debug_compare_op5(
    const debug_tensor_stats_t *black_stats,
    const debug_tensor_stats_t *white_stats,
    size_t stats_count)
{
    if (!black_stats || !white_stats || !s_model || !s_model->subgraphs()) {
        return;
    }

    /* Log tham số định lượng QPARAM & Saturation count của OP5 */
    debug_log_op5_qparams_and_saturation();

    const auto *sg = s_model->subgraphs()->Get(0);
    const auto *ops = sg->operators();

    if (!ops || APP_AI_DEBUG_TARGET_OP >= ops->size()) {
        return;
    }

    const auto *op = ops->Get(APP_AI_DEBUG_TARGET_OP);

    ESP_LOGE(TAG, "################################################");
    ESP_LOGE(TAG, "############ OP5 DEEP DEBUG ###################");

    /* INPUTS */
    for (unsigned i = 0; i < op->inputs()->size(); i++) {
        int tid = op->inputs()->Get(i);
        if (tid < 0 || (size_t)tid >= stats_count) continue;

        const debug_tensor_stats_t *b = &black_stats[tid];
        const debug_tensor_stats_t *w = &white_stats[tid];

        ESP_LOGE(TAG, "[OP5 INPUT%u] T%d", i, tid);
        ESP_LOGE(TAG, "[OP5 INPUT%u] B hash=0x%08X min=%.4f max=%.4f mean=%.4f nz=%u/%u",
                 i, (unsigned)b->hash, (double)b->min_value, (double)b->max_value, (double)b->mean_value, (unsigned)b->nonzero, (unsigned)b->elements);
        ESP_LOGE(TAG, "[OP5 INPUT%u] W hash=0x%08X min=%.4f max=%.4f mean=%.4f nz=%u/%u",
                 i, (unsigned)w->hash, (double)w->min_value, (double)w->max_value, (double)w->mean_value, (unsigned)w->nonzero, (unsigned)w->elements);
        ESP_LOGE(TAG, "[OP5 INPUT%u] B first8=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]",
                 i, (double)b->first8[0], (double)b->first8[1], (double)b->first8[2], (double)b->first8[3],
                    (double)b->first8[4], (double)b->first8[5], (double)b->first8[6], (double)b->first8[7]);
        ESP_LOGE(TAG, "[OP5 INPUT%u] W first8=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]",
                 i, (double)w->first8[0], (double)w->first8[1], (double)w->first8[2], (double)w->first8[3],
                    (double)w->first8[4], (double)w->first8[5], (double)w->first8[6], (double)w->first8[7]);
    }

    /* OUTPUT */
    for (unsigned i = 0; i < op->outputs()->size(); i++) {
        int tid = op->outputs()->Get(i);
        if (tid < 0 || (size_t)tid >= stats_count) continue;

        const debug_tensor_stats_t *b = &black_stats[tid];
        const debug_tensor_stats_t *w = &white_stats[tid];

        ESP_LOGE(TAG, "[OP5 OUTPUT%u] T%d", i, tid);
        ESP_LOGE(TAG, "[OP5 OUTPUT%u] B hash=0x%08X min=%.4f max=%.4f mean=%.4f nz=%u/%u",
                 i, (unsigned)b->hash, (double)b->min_value, (double)b->max_value, (double)b->mean_value, (unsigned)b->nonzero, (unsigned)b->elements);
        ESP_LOGE(TAG, "[OP5 OUTPUT%u] W hash=0x%08X min=%.4f max=%.4f mean=%.4f nz=%u/%u",
                 i, (unsigned)w->hash, (double)w->min_value, (double)w->max_value, (double)w->mean_value, (unsigned)w->nonzero, (unsigned)w->elements);
        ESP_LOGE(TAG, "[OP5 OUTPUT%u] B first8=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]",
                 i, (double)b->first8[0], (double)b->first8[1], (double)b->first8[2], (double)b->first8[3],
                    (double)b->first8[4], (double)b->first8[5], (double)b->first8[6], (double)b->first8[7]);
        ESP_LOGE(TAG, "[OP5 OUTPUT%u] W first8=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]",
                 i, (double)w->first8[0], (double)w->first8[1], (double)w->first8[2], (double)w->first8[3],
                    (double)w->first8[4], (double)w->first8[5], (double)w->first8[6], (double)w->first8[7]);

        if (b->hash == w->hash) {
            ESP_LOGE(TAG, "[OP5 OUTPUT%u] BLACK == WHITE EXACTLY", i);
        } else {
            ESP_LOGE(TAG, "[OP5 OUTPUT%u] BLACK != WHITE", i);
        }
    }

    ESP_LOGE(TAG, "################################################");

    ai_wdt_feed_and_yield();
}

/* ================================================================
 * COMPARE BLACK / WHITE
 * ================================================================ */

static void debug_compare_operator_outputs(
    const uint32_t *black_hash,
    const uint32_t *white_hash,
    const debug_tensor_stats_t *black_stats,
    const debug_tensor_stats_t *white_stats,
    size_t hash_count)
{
    if (!black_hash || !white_hash || !black_stats || !white_stats || !s_model || !s_interpreter || !s_model->subgraphs()) {
        return;
    }

    const auto *sg = s_model->subgraphs()->Get(0);
    const auto *ops = sg->operators();
    if (!ops) return;

    ESP_LOGI(TAG, "=========== BLACK vs WHITE ===========");

    bool first_diff_reported = false;
    bool first_collapse_reported = false;
    bool previous_was_diff = false;

    unsigned previous_op = 0;
    int previous_tensor = -1;

    for (unsigned op_index = 0; op_index < ops->size(); op_index++) {
        const auto *op = ops->Get(op_index);
        uint32_t opcode_index = op->opcode_index();
        const auto *opcode = s_model->operator_codes()->Get(opcode_index);
        const char *name = tflite::EnumNameBuiltinOperator(opcode->builtin_code());

        for (unsigned out_index = 0; out_index < op->outputs()->size(); out_index++) {
            int tensor_index = op->outputs()->Get(out_index);
            if (tensor_index < 0 || (size_t)tensor_index >= hash_count) continue;

            const uint32_t bh = black_hash[tensor_index];
            const uint32_t wh = white_hash[tensor_index];
            bool same = (bh == wh);

            TfLiteEvalTensor *tensor = s_interpreter->GetTensor(tensor_index);
            char shape[48];
            debug_tensor_shape_string(tensor, shape, sizeof(shape));

#if APP_AI_INFER_DEBUG_TRACE_ALL
            ESP_LOGI(TAG, "[COMPARE] OP%u %-18s T%d [%s] %s B=0x%08X W=0x%08X",
                     op_index, name ? name : "UNKNOWN", tensor_index, shape, same ? "SAME" : "DIFF", (unsigned)bh, (unsigned)wh);
#else
            if (op_index <= APP_AI_DEBUG_TARGET_OP || same == false) {
                ESP_LOGI(TAG, "[COMPARE] OP%u %-18s T%d [%s] %s B=0x%08X W=0x%08X",
                         op_index, name ? name : "UNKNOWN", tensor_index, shape, same ? "SAME" : "DIFF", (unsigned)bh, (unsigned)wh);
            }
#endif

            if (!same && !first_diff_reported) {
                ESP_LOGW(TAG, "[FIRST DIFF] OP%u %s -> T%d", op_index, name ? name : "UNKNOWN", tensor_index);
                first_diff_reported = true;
            }

            if (same && previous_was_diff && !first_collapse_reported) {
                ESP_LOGE(TAG, "[FIRST COLLAPSE] previous OP%u -> T%d DIFF, but OP%u %s -> T%d SAME!",
                         previous_op, previous_tensor, op_index, name ? name : "UNKNOWN", tensor_index);
                ESP_LOGE(TAG, "[FIRST COLLAPSE] THIS IS THE FIRST KERNEL TO INVESTIGATE.");

                debug_log_tensor(tensor_index, "COLLAPSE_CURRENT", true);
                first_collapse_reported = true;
            }

            previous_was_diff = !same;
            previous_op = op_index;
            previous_tensor = tensor_index;
        }

        /* OP5 deep debug */
        if (op_index == APP_AI_DEBUG_TARGET_OP) {
            debug_compare_op5(black_stats, white_stats, hash_count);
        }

        if ((op_index & 0x03) == 0) {
            ai_wdt_feed_and_yield();
        }
    }

    if (!first_diff_reported) {
        ESP_LOGE(TAG, "[TRACE RESULT] BLACK and WHITE never differ.");
    }

    if (!first_collapse_reported) {
        ESP_LOGI(TAG, "[TRACE RESULT] No immediate DIFF->SAME collapse detected.");
    }

    ESP_LOGI(TAG, "========================================");
}

/* ================================================================
 * CONSTANT INPUT TEST
 * ================================================================ */

static void debug_run_one_constant_test(
    const char *name,
    uint8_t pixel,
    float *out_raw,
    uint32_t *out_hash,
    uint32_t *trace_hashes,
    debug_tensor_stats_t *trace_stats,
    size_t trace_count)
{
    size_t patch_len = (size_t)APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS;

    for (size_t p = 0; p < patch_len; p++) {
        write_input_pixel(p, pixel);
    }

    uint32_t input_hash = calc_input_checksum();
    int first_value = 0;
    int last_value = 0;

    if (s_input->type == kTfLiteInt8) {
        first_value = (int)s_input->data.int8[0];
        last_value  = (int)s_input->data.int8[patch_len - 1];
    } else if (s_input->type == kTfLiteUInt8) {
        first_value = (int)s_input->data.uint8[0];
        last_value  = (int)s_input->data.uint8[patch_len - 1];
    }

    ESP_LOGI(TAG, "[TEST %s] pixel=%u input_hash=0x%08X first=%d last=%d",
             name, (unsigned)pixel, (unsigned)input_hash, first_value, last_value);

    debug_log_tensor(0, name, true);

    int64_t invoke_t0 = esp_timer_get_time();

    if (esp_task_wdt_status(NULL) == ESP_OK) {
        esp_task_wdt_reset();
    }

    TfLiteStatus status = s_interpreter->Invoke();
    int64_t invoke_us = esp_timer_get_time() - invoke_t0;

    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "[TEST %s] Invoke FAILED", name);
        ai_wdt_feed_and_yield();
        return;
    }

    if (invoke_us > APP_AI_INVOKE_WARN_US) {
        ESP_LOGW(TAG, "[WDT WATCH] TEST %s Invoke=%lld us", name, (long long)invoke_us);
    }

    ESP_LOGI(TAG, "[TEST %s] Invoke=%lld us", name, (long long)invoke_us);

    for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
        out_raw[d] = read_output_value((size_t)d);
    }

    if (out_hash) {
        int output_tensor_index = s_model->subgraphs()->Get(0)->outputs()->Get(0);
        *out_hash = debug_tensor_hash(s_interpreter->GetTensor(output_tensor_index));
    }

    if (trace_hashes && trace_count > 0) {
        debug_capture_operator_output_hashes(trace_hashes, trace_count, name);
    }

    /* Capture stats SAU Invoke */
    if (trace_stats && trace_count > 0) {
        for (size_t i = 0; i < trace_count; i++) {
            debug_collect_tensor_stats((int)i, &trace_stats[i]);
        }
    }

    /* Nếu là WHITE thì soi OP5 trực tiếp */
    if (strcmp(name, "WHITE") == 0) {
        debug_log_op_detail(APP_AI_DEBUG_TARGET_OP, "WHITE CURRENT");
    }

    ESP_LOGI(TAG, "[TEST %s] OUTPUT32:", name);
    for (int base = 0; base < APP_AI_INFER_FEATURE_DIM; base += 8) {
        ESP_LOGI(TAG, "[%s] %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f",
                 name,
                 out_raw[base + 0], out_raw[base + 1], out_raw[base + 2], out_raw[base + 3],
                 out_raw[base + 4], out_raw[base + 5], out_raw[base + 6], out_raw[base + 7]);
    }

    ai_wdt_feed_and_yield();
}

/* ================================================================
 * BLACK / GRAY / WHITE
 * ================================================================ */

static void run_black_white_test(void)
{
#if APP_AI_INFER_DEBUG_SINGLE_PATCH
    ESP_LOGI(TAG, "[AI DEBUG] skipped black/white test for single-patch run");
    return;
#endif

    if (!s_interpreter || !s_input || !s_output) return;

    const size_t tensor_count = s_model->subgraphs()->Get(0)->tensors()->size();
    size_t trace_count = tensor_count < APP_AI_DEBUG_MAX_TENSORS ? tensor_count : APP_AI_DEBUG_MAX_TENSORS;

    if (tensor_count > APP_AI_DEBUG_MAX_TENSORS) {
        ESP_LOGW(TAG, "[TEST] tensor_count=%u trace limited to %u",
                 (unsigned)tensor_count, (unsigned)APP_AI_DEBUG_MAX_TENSORS);
    }

    static uint32_t black_trace[APP_AI_DEBUG_MAX_TENSORS];
    static uint32_t white_trace[APP_AI_DEBUG_MAX_TENSORS];

    static debug_tensor_stats_t black_stats[APP_AI_DEBUG_MAX_TENSORS];
    static debug_tensor_stats_t white_stats[APP_AI_DEBUG_MAX_TENSORS];

    float black_raw[APP_AI_INFER_FEATURE_DIM] = {0};
    float gray_raw[APP_AI_INFER_FEATURE_DIM]  = {0};
    float white_raw[APP_AI_INFER_FEATURE_DIM] = {0};

    uint32_t black_hash = 0;
    uint32_t gray_hash  = 0;
    uint32_t white_hash = 0;

    memset(black_stats, 0, sizeof(black_stats));
    memset(white_stats, 0, sizeof(white_stats));

    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, "BLACK / GRAY / WHITE DIAGNOSTIC");
    ESP_LOGI(TAG, "tensor_count=%u preserve_all=%s target_OP=%d",
             (unsigned)tensor_count,
             s_interpreter->preserve_all_tensors() ? "YES" : "NO",
             APP_AI_DEBUG_TARGET_OP);

    debug_run_one_constant_test("BLACK", 0, black_raw, &black_hash, black_trace, black_stats, trace_count);
    debug_run_one_constant_test("GRAY", 128, gray_raw, &gray_hash, nullptr, nullptr, 0);
    debug_run_one_constant_test("WHITE", 255, white_raw, &white_hash, white_trace, white_stats, trace_count);

    float bw_max = 0.0f;
    float bg_max = 0.0f;
    float gw_max = 0.0f;
    float bw_sum_sq = 0.0f;
    bool bw_diff = false;

    for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
        float bw = fabsf(black_raw[d] - white_raw[d]);
        float bg = fabsf(black_raw[d] - gray_raw[d]);
        float gw = fabsf(gray_raw[d] - white_raw[d]);

        if (bw > bw_max) bw_max = bw;
        if (bg > bg_max) bg_max = bg;
        if (gw > gw_max) gw_max = gw;

        bw_sum_sq += bw * bw;
        if (bw > 1e-5f) bw_diff = true;
    }

    ESP_LOGI(TAG, "[FINAL] hash B=0x%08X G=0x%08X W=0x%08X",
             (unsigned)black_hash, (unsigned)gray_hash, (unsigned)white_hash);
    ESP_LOGI(TAG, "[FINAL] BW max=%.8f L2=%.8f BG max=%.8f GW max=%.8f",
             (double)bw_max, (double)sqrtf(bw_sum_sq), (double)bg_max, (double)gw_max);

    debug_compare_operator_outputs(black_trace, white_trace, black_stats, white_stats, trace_count);

    if (bw_diff) {
        ESP_LOGI(TAG, "[TEST RESULT] BLACK/WHITE OUTPUT DIFFER");
    } else {
        ESP_LOGE(TAG, "[TEST RESULT] BLACK/WHITE OUTPUT IDENTICAL");
    }

    ESP_LOGI(TAG, "================================================");
}

/* ================================================================
 * PATCH PAIR TRACE
 * ================================================================ */

static uint32_t s_patch_ref_hash[APP_AI_DEBUG_MAX_TENSORS];
static bool s_patch_ref_valid = false;
static int s_patch_ref_index = -1;
static debug_tensor_stats_t s_patch_ref_stats[APP_AI_DEBUG_MAX_TENSORS];

static void debug_trace_patch_pair(int patch_index)
{
    if (!s_interpreter || !s_model || !s_model->subgraphs()) return;

    const auto* subgraph = s_model->subgraphs()->Get(0);
    if (!subgraph || !subgraph->tensors()) return;

    const int tensor_count = (int)subgraph->tensors()->size();

    if (tensor_count > APP_AI_DEBUG_MAX_TENSORS) {
        ESP_LOGE(TAG, "[PATCH TRACE] tensor_count=%d > max=%d",
                 tensor_count, APP_AI_DEBUG_MAX_TENSORS);
        return;
    }

    uint32_t cur_hash[APP_AI_DEBUG_MAX_TENSORS];
    memset(cur_hash, 0, sizeof(cur_hash));

    for (int i = 0; i < tensor_count; i++) {
        TfLiteEvalTensor* t = s_interpreter->GetTensor(i);
        if (t && t->data.raw) {
            cur_hash[i] = debug_tensor_hash(t);
        }
    }

    if (patch_index == 0) {
        memcpy(s_patch_ref_hash, cur_hash, sizeof(s_patch_ref_hash));
        for (int i = 0; i < tensor_count; i++) {
            debug_collect_tensor_stats(i, &s_patch_ref_stats[i]);
        }
        s_patch_ref_valid = true;
        s_patch_ref_index = 0;

        ESP_LOGE(TAG, "[PATCH TRACE] saved reference patch=0");
        return;
    }

    if (!s_patch_ref_valid) return;

    ESP_LOGE(TAG, "[PATCH TRACE] ref=p%d target=p%d",
             s_patch_ref_index, patch_index);

    if (!subgraph->operators()) return;
    const int op_count = (int)subgraph->operators()->size();

    for (int i = 0; i < op_count; i++) {
        const auto* op = subgraph->operators()->Get(i);
        if (!op) continue;

        bool input_diff = false;
        bool output_same = true;
        int first_diff_input = -1;
        int first_output = -1;

        if (op->inputs()) {
            for (size_t j = 0; j < op->inputs()->size(); j++) {
                const int tid = op->inputs()->Get(j);
                if (tid < 0 || tid >= tensor_count) continue;

                if (s_patch_ref_hash[tid] != cur_hash[tid]) {
                    input_diff = true;
                    if (first_diff_input < 0) first_diff_input = tid;
                }
            }
        }

        if (op->outputs()) {
            for (size_t j = 0; j < op->outputs()->size(); j++) {
                const int tid = op->outputs()->Get(j);
                if (tid < 0 || tid >= tensor_count) continue;

                if (first_output < 0) first_output = tid;

                if (s_patch_ref_hash[tid] != cur_hash[tid]) {
                    output_same = false;
                }
            }
        }

        if (first_diff_input >= 0 && first_output >= 0) {
            TfLiteEvalTensor* tin = s_interpreter->GetTensor(first_diff_input);
            TfLiteEvalTensor* tout = s_interpreter->GetTensor(first_output);

            if (tin && tout) {
                size_t tin_elems = debug_tensor_elements(tin);
                size_t tout_elems = debug_tensor_elements(tout);

                ESP_LOGE(TAG,
                        "[COLLAPSE DETAIL] OP%d T%d -> T%d | IN hash=0x%08X min=%d max=%d | OUT hash=0x%08X min=%d max=%d",
                        i,
                        first_diff_input,
                        first_output,
                        (unsigned int)cur_hash[first_diff_input],
                        (tin->type == kTfLiteInt8 && tin->data.int8 && tin_elems > 0) ? (int)tin->data.int8[0] : 0,
                        (tin->type == kTfLiteInt8 && tin->data.int8 && tin_elems > 0) ? (int)tin->data.int8[tin_elems - 1] : 0,
                        (unsigned int)cur_hash[first_output],
                        (tout->type == kTfLiteInt8 && tout->data.int8 && tout_elems > 0) ? (int)tout->data.int8[0] : 0,
                        (tout->type == kTfLiteInt8 && tout->data.int8 && tout_elems > 0) ? (int)tout->data.int8[tout_elems - 1] : 0);

                if (input_diff && output_same) {
                    debug_log_saturation(first_diff_input, "INPUT");
                    debug_log_saturation(first_output, "OUTPUT");

                    ESP_LOGE(TAG,
                            "[PATCH COLLAPSE] ref=p%d target=p%d OP%d input T%d DIFF -> output T%d SAME",
                            s_patch_ref_index, patch_index, i,
                            first_diff_input, first_output);
                    return;
                }
            }
        }

        if (input_diff && output_same) {
            if (first_diff_input >= 0 && first_output >= 0) {
                debug_tensor_stats_t cur_in_stats;
                debug_tensor_stats_t cur_out_stats;
                debug_collect_tensor_stats(first_diff_input, &cur_in_stats);
                debug_collect_tensor_stats(first_output, &cur_out_stats);
                const debug_tensor_stats_t *ref_in_stats =
                    &s_patch_ref_stats[first_diff_input];
                const debug_tensor_stats_t *ref_out_stats =
                    &s_patch_ref_stats[first_output];
                ESP_LOGE(TAG,
                         "[REF INPUT ] T%d hash=0x%08X min=%.1f max=%.1f mean=%.3f",
                         first_diff_input,
                         (unsigned)ref_in_stats->hash,
                         (double)ref_in_stats->min_value,
                         (double)ref_in_stats->max_value,
                         (double)ref_in_stats->mean_value);
                ESP_LOGE(TAG,
                         "[CUR INPUT ] T%d hash=0x%08X min=%.1f max=%.1f mean=%.3f",
                         first_diff_input,
                         (unsigned)cur_in_stats.hash,
                         (double)cur_in_stats.min_value,
                         (double)cur_in_stats.max_value,
                         (double)cur_in_stats.mean_value);
                ESP_LOGE(TAG,
                         "[REF OUTPUT] T%d hash=0x%08X min=%.1f max=%.1f mean=%.3f",
                         first_output,
                         (unsigned)ref_out_stats->hash,
                         (double)ref_out_stats->min_value,
                         (double)ref_out_stats->max_value,
                         (double)ref_out_stats->mean_value);
                ESP_LOGE(TAG,
                         "[CUR OUTPUT] T%d hash=0x%08X min=%.1f max=%.1f mean=%.3f",
                         first_output,
                         (unsigned)cur_out_stats.hash,
                         (double)cur_out_stats.min_value,
                         (double)cur_out_stats.max_value,
                         (double)cur_out_stats.mean_value);
            }

            ESP_LOGE(TAG,
                     "[PATCH COLLAPSE] ref=p%d target=p%d OP%d input T%d DIFF -> output T%d SAME",
                     s_patch_ref_index, patch_index, i,
                     first_diff_input, first_output);
            return;
        }
    }

    ESP_LOGE(TAG,
             "[PATCH TRACE] no input-DIFF -> output-SAME collapse for p%d vs p%d",
             s_patch_ref_index, patch_index);
}

#endif

/* BƯỚC 4: Thêm hàm dump buffer s_debug_patch13 */
#if APP_AI_INFER_DEBUG_DIAG
static void debug_dump_patch13_copy(void)
{
    if (!s_debug_patch13_valid) {
        ESP_LOGE(TAG, "[PATCH DUMP] p13 copy unavailable");
        return;
    }

    const size_t len =
        (size_t)APP_AI_PATCH_SIZE *
        APP_AI_PATCH_SIZE *
        APP_AI_PATCH_CHANNELS;

    uint32_t hash = 2166136261u;
    uint8_t min_value = 255;
    uint8_t max_value = 0;

    for (size_t i = 0; i < len; i++) {
        const uint8_t value = s_debug_patch13[i];
        hash ^= value;
        hash *= 16777619u;
        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
    }

    ESP_LOGI(TAG, "[PATCH13 RGB] len=%u hash=0x%08X min=%u max=%u",
             (unsigned)len, (unsigned)hash, (unsigned)min_value, (unsigned)max_value);
    ESP_LOGI(TAG, "[PATCH13 RGB] first8=%u,%u,%u,%u,%u,%u,%u,%u",
             (unsigned)s_debug_patch13[0], (unsigned)s_debug_patch13[1],
             (unsigned)s_debug_patch13[2], (unsigned)s_debug_patch13[3],
             (unsigned)s_debug_patch13[4], (unsigned)s_debug_patch13[5],
             (unsigned)s_debug_patch13[6], (unsigned)s_debug_patch13[7]);
    ESP_LOGI(TAG, "[PATCH13 RGB] last8=%u,%u,%u,%u,%u,%u,%u,%u",
             (unsigned)s_debug_patch13[len - 8], (unsigned)s_debug_patch13[len - 7],
             (unsigned)s_debug_patch13[len - 6], (unsigned)s_debug_patch13[len - 5],
             (unsigned)s_debug_patch13[len - 4], (unsigned)s_debug_patch13[len - 3],
             (unsigned)s_debug_patch13[len - 2], (unsigned)s_debug_patch13[len - 1]);
}
#endif

/* ================================================================
 * INIT
 * ================================================================ */

esp_err_t app_ai_infer_init(void)
{
    if (s_ai_infer.lock) {
        return ESP_OK;
    }

    s_ai_infer.lock = xSemaphoreCreateMutex();
    if (!s_ai_infer.lock) {
        ESP_LOGE(TAG, "Create AI infer mutex failed");
        return ESP_ERR_NO_MEM;
    }

    size_t vectors_bytes = (size_t)APP_AI_PATCH_MAX_COUNT * sizeof(app_ai_infer_vector_t);

    s_ai_infer.vectors = (app_ai_infer_vector_t *)heap_caps_malloc(vectors_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_ai_infer.vectors) {
        s_ai_infer.vectors = (app_ai_infer_vector_t *)heap_caps_malloc(vectors_bytes, MALLOC_CAP_8BIT);
    }

    if (!s_ai_infer.vectors) {
        ESP_LOGE(TAG, "Allocate vector buffer failed");
        return ESP_ERR_NO_MEM;
    }

    s_model = tflite::GetModel(feature_extractor_32d_full_int8_tflite_start);
    if (!s_model) {
        ESP_LOGE(TAG, "GetModel failed");
        return ESP_FAIL;
    }

    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu != %d", (unsigned long)s_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    const size_t arena_alignment = 32;

    s_arena = (uint8_t *)heap_caps_aligned_alloc(arena_alignment, APP_AI_INFER_ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!s_arena) {
        ESP_LOGW(TAG, "Internal DRAM full, fallback PSRAM");
        s_arena = (uint8_t *)heap_caps_aligned_alloc(arena_alignment, APP_AI_INFER_ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }

    if (!s_arena) {
        ESP_LOGE(TAG, "Allocate tensor arena failed");
        return ESP_ERR_NO_MEM;
    }

    static tflite::MicroMutableOpResolver<APP_AI_INFER_NUM_OPS> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddSub();
    resolver.AddMul();
    resolver.AddDiv();
    resolver.AddDequantize();
    resolver.AddQuantize();
    resolver.AddSquare();
    resolver.AddSqrt();
    resolver.AddRsqrt();
    resolver.AddSum();
    resolver.AddL2Normalization();
    resolver.AddMaximum();
    resolver.AddMinimum();
    resolver.AddPad();
    resolver.AddLogistic();

    const bool preserve_all_tensors = (APP_AI_INFER_DEBUG_DIAG != 0);

    s_interpreter = new (s_interpreter_storage) tflite::MicroInterpreter(
        s_model,
        resolver,
        s_arena,
        APP_AI_INFER_ARENA_BYTES,
        nullptr,
        nullptr,
        preserve_all_tensors);

    TfLiteStatus alloc_status = s_interpreter->AllocateTensors();

    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed preserve=%s", preserve_all_tensors ? "YES" : "NO");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "[AI ARENA] used=%u/%u (%.1f%%) preserve=%s",
             (unsigned)s_interpreter->arena_used_bytes(),
             (unsigned)APP_AI_INFER_ARENA_BYTES,
             100.0 * (double)s_interpreter->arena_used_bytes() / (double)APP_AI_INFER_ARENA_BYTES,
             preserve_all_tensors ? "YES" : "NO");

    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    ESP_LOGE(TAG, "[MODEL INPUT] type=%d bytes=%d scale=%.10f zero_point=%ld",
             s_input->type,
             (int)s_input->bytes,
             (double)s_input->params.scale,
             (long)s_input->params.zero_point);
    ESP_LOGE(TAG, "[MODEL OUTPUT] type=%d bytes=%d scale=%.10f zero_point=%ld",
             s_output->type,
             (int)s_output->bytes,
             (double)s_output->params.scale,
             (long)s_output->params.zero_point);

    bool shape_ok = s_input && s_output &&
                    s_input->dims->size == 4 &&
                    s_input->dims->data[1] == APP_AI_PATCH_SIZE &&
                    s_input->dims->data[2] == APP_AI_PATCH_SIZE &&
                    s_input->dims->data[3] == APP_AI_PATCH_CHANNELS &&
                    s_output->dims->size == 2 &&
                    s_output->dims->data[1] == APP_AI_INFER_FEATURE_DIM;

    if (!shape_ok) {
        ESP_LOGE(TAG, "Model shape mismatch");
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "=== INPUT TENSOR ===");
    ESP_LOGI(TAG, "ptr=%p bytes=%u type=%d shape=%d,%d,%d,%d scale=%.9f zp=%d",
             (void *)s_input->data.raw, (unsigned)s_input->bytes, (int)s_input->type,
             s_input->dims->data[0], s_input->dims->data[1], s_input->dims->data[2], s_input->dims->data[3],
             (double)s_input->params.scale, (int)s_input->params.zero_point);

    ESP_LOGI(TAG, "=== OUTPUT TENSOR ===");
    ESP_LOGI(TAG, "ptr=%p bytes=%u type=%d shape=%d,%d scale=%.9f zp=%d",
             (void *)s_output->data.raw, (unsigned)s_output->bytes, (int)s_output->type,
             s_output->dims->data[0], s_output->dims->data[1],
             (double)s_output->params.scale, (int)s_output->params.zero_point);

#if APP_AI_INFER_ENABLE_DEDUP
    s_cached_input = (uint8_t *)heap_caps_malloc(s_input->bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!s_cached_input) {
        s_cached_input = (uint8_t *)heap_caps_malloc(s_input->bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    if (s_cached_input) {
        s_cached_input_bytes = s_input->bytes;
    } else {
        ESP_LOGW(TAG, "Input dedup cache unavailable; inference will run for every patch");
    }
#else
    ESP_LOGI(TAG, "[AI DEDUP] DISABLED - every patch executes Invoke()");
#endif

    if (s_input->type == kTfLiteInt8) {
        build_pixel_int8_lut();
    }

#if APP_AI_INFER_DEBUG_DIAG

    ESP_LOGI(TAG, "[AI DEBUG] preserve_all=%s", s_interpreter->preserve_all_tensors() ? "YES" : "NO");

    debug_dump_model_graph();
    run_black_white_test();

#endif

    s_ai_infer.ready = true;
    ESP_LOGI(TAG, "[AI INIT] READY");

    return ESP_OK;
}

/* ================================================================
 * READY
 * ================================================================ */

bool app_ai_infer_is_ready(void)
{
    return s_ai_infer.ready;
}

/* ================================================================
 * VECTOR DIVERSITY CHECK (luôn bật, rẻ: ~0.6M phép tính cho 192 patch)
 *
 * Trả lời câu hỏi: "các patch KHÁC nhau có ra vector KHÁC nhau không?"
 *  - exact_dup_diff_input : 2 patch có RGB khác nhau nhưng vector giống hệt
 *                           => model/kernel bị sập (lỗi cũ). PHẢI = 0.
 *  - exact_dup_same_input : RGB giống hệt nên vector giống là bình thường.
 * ================================================================ */

static uint32_t s_patch_rgb_hash[APP_AI_PATCH_MAX_COUNT];

static void log_vector_diversity(uint32_t n)
{
    if (n < 2) return;

    uint32_t dup_diff_input = 0;
    uint32_t dup_same_input = 0;
    uint32_t near_dup_diff_input = 0;
    uint32_t first_bad_a = 0, first_bad_b = 0;
    double sum_l2 = 0.0;
    float min_l2 = 1e30f, max_l2 = 0.0f;
    uint64_t pairs = 0;

    for (uint32_t i = 0; i < n; i++) {
        const app_ai_infer_vector_t *vi = &s_ai_infer.vectors[i];
        for (uint32_t j = i + 1; j < n; j++) {
            const app_ai_infer_vector_t *vj = &s_ai_infer.vectors[j];
            bool exact = true;
            float d2 = 0.0f;
            for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
                if (vi->raw[d] != vj->raw[d]) exact = false;
                float diff = vi->values[d] - vj->values[d];
                d2 += diff * diff;
            }
            float l2 = sqrtf(d2);
            const bool same_input = (s_patch_rgb_hash[i] == s_patch_rgb_hash[j]);

            if (exact) {
                if (same_input) {
                    dup_same_input++;
                } else {
                    if (dup_diff_input == 0) { first_bad_a = i; first_bad_b = j; }
                    dup_diff_input++;
                }
            } else if (l2 < 1e-3f && !same_input) {
                near_dup_diff_input++;
            }

            if (l2 < min_l2) min_l2 = l2;
            if (l2 > max_l2) max_l2 = l2;
            sum_l2 += l2;
            pairs++;
        }
        if ((i & 0x1F) == 0x1F) {
            ai_wdt_feed_and_yield();
        }
    }

    ESP_LOGI(TAG,
             "[AI DIVERSITY] n=%u pairs=%llu L2(normalized): min=%.5f mean=%.5f max=%.5f",
             (unsigned)n, (unsigned long long)pairs,
             (double)min_l2, (double)(sum_l2 / (double)pairs), (double)max_l2);
    ESP_LOGI(TAG,
             "[AI DIVERSITY] exact_dup(same RGB)=%u  exact_dup(DIFFERENT RGB)=%u  near_dup(DIFFERENT RGB)=%u",
             (unsigned)dup_same_input, (unsigned)dup_diff_input, (unsigned)near_dup_diff_input);

    if (dup_diff_input > 0) {
        ESP_LOGE(TAG,
                 "[AI DIVERSITY] FAIL: %u cap patch RGB khac nhau nhung vector GIONG HET (vd p%u == p%u) -> kernel/model sap",
                 (unsigned)dup_diff_input, (unsigned)first_bad_a, (unsigned)first_bad_b);
    } else if (near_dup_diff_input > 0) {
        ESP_LOGW(TAG, "[AI DIVERSITY] WARN: %u cap patch khac RGB nhung vector gan nhu trung (L2<1e-3)",
                 (unsigned)near_dup_diff_input);
    } else {
        ESP_LOGI(TAG, "[AI DIVERSITY] PASS: patch khac nhau -> vector khac nhau");
    }
}

/* ================================================================
 * INFERENCE
 * ================================================================ */

esp_err_t app_ai_infer_run_all(void)
{
    if (!s_ai_infer.lock || !s_ai_infer.ready) {
        ESP_LOGW(TAG, "AI infer skipped: model not ready");
        return ESP_ERR_INVALID_STATE;
    }

    uint32_t count = app_ai_patch_get_count();

    if (count == 0) {
        ESP_LOGW(TAG, "AI infer skipped: no patches");

        xSemaphoreTake(s_ai_infer.lock, portMAX_DELAY);
        s_ai_infer.last_run_ok = false;
        s_ai_infer.count = 0;
        snprintf(s_ai_infer.last_error, sizeof(s_ai_infer.last_error), "no_patches");
        xSemaphoreGive(s_ai_infer.lock);

        return ESP_ERR_INVALID_STATE;
    }

    if (count > APP_AI_PATCH_MAX_COUNT) {
        count = APP_AI_PATCH_MAX_COUNT;
    }

    uint32_t run_count = count;
#if APP_AI_INFER_DEBUG_DIAG && APP_AI_INFER_DEBUG_SINGLE_PATCH
    if (count <= APP_AI_INFER_DEBUG_PATCH_INDEX) {
        ESP_LOGE(TAG, "[AI DEBUG] patch p%d unavailable; captured patch count=%u",
                 APP_AI_INFER_DEBUG_PATCH_INDEX, (unsigned)count);
        xSemaphoreTake(s_ai_infer.lock, portMAX_DELAY);
        s_ai_infer.last_run_ok = false;
        s_ai_infer.count = 0;
        snprintf(s_ai_infer.last_error, sizeof(s_ai_infer.last_error), "patch13_unavailable");
        xSemaphoreGive(s_ai_infer.lock);
        return ESP_ERR_NOT_FOUND;
    }
    run_count = 1;
#endif

    xSemaphoreTake(s_ai_infer.lock, portMAX_DELAY);

    int64_t t0 = esp_timer_get_time();
    esp_err_t ret = ESP_OK;
    const char *err_code = NULL;
    uint32_t done = 0;
    uint32_t reused_inputs = 0;
    bool added_to_wdt = false;

    if (esp_task_wdt_status(NULL) != ESP_OK) {
        esp_err_t wdt_add_status = esp_task_wdt_add(NULL);
        if (wdt_add_status == ESP_OK) {
            added_to_wdt = true;
        } else {
            ESP_LOGW(TAG, "Could not subscribe inference task to WDT: %s", esp_err_to_name(wdt_add_status));
        }
    }

    s_ai_infer.prev_raw_valid = false;

#if APP_AI_INFER_ENABLE_DEDUP
    s_cached_output_valid = false;
#endif

#if APP_AI_INFER_DEBUG_PATCH_IO
    memset(s_debug_rgb_hashes, 0, sizeof(s_debug_rgb_hashes));
    memset(s_debug_input_hashes, 0, sizeof(s_debug_input_hashes));
    memset(s_debug_raw_sums, 0, sizeof(s_debug_raw_sums));
#endif

    ESP_LOGI(TAG, "[AI START] patches=%u selected=%u core=%d",
             (unsigned)count, (unsigned)run_count, xPortGetCoreID());
    ESP_LOGI(TAG, "[AI WDT] task_status=%s", esp_task_wdt_status(NULL) == ESP_OK ? "SUBSCRIBED" : "NOT_SUBSCRIBED");

    for (uint32_t run_index = 0; run_index < run_count; run_index++) {
        uint32_t i = run_index;
        const uint32_t vector_index = run_index;
#if APP_AI_INFER_DEBUG_DIAG && APP_AI_INFER_DEBUG_SINGLE_PATCH
        i = APP_AI_INFER_DEBUG_PATCH_INDEX;
#endif
        size_t len = 0;
        const uint8_t *patch = app_ai_patch_get_data(i, &len);

        if (!patch || len != (size_t)APP_AI_PATCH_SIZE * APP_AI_PATCH_SIZE * APP_AI_PATCH_CHANNELS) {
            ret = ESP_ERR_INVALID_SIZE;
            err_code = "patch_size_mismatch";
            break;
        }

/* BƯỚC 3: Copy patch13 ngay lúc ESP32 lấy patch thành công */
#if APP_AI_INFER_DEBUG_DIAG
        if (i == 13) {
            memcpy(s_debug_patch13, patch, len);
            s_debug_patch13_valid = true;
            debug_dump_patch13_copy();
        }
#endif

        s_input  = s_interpreter->input(0);
        s_output = s_interpreter->output(0);

#if APP_AI_INFER_DEBUG_PATCH_IO || APP_AI_INFER_DEBUG_DIAG
        ESP_LOGE(TAG, "[MODEL INPUT] type=%d bytes=%d scale=%.10f zero_point=%ld",
                 s_input->type,
                 (int)s_input->bytes,
                 (double)s_input->params.scale,
                 (long)s_input->params.zero_point);
        ESP_LOGE(TAG, "[MODEL OUTPUT] type=%d bytes=%d scale=%.10f zero_point=%ld",
                 s_output->type,
                 (int)s_output->bytes,
                 (double)s_output->params.scale,
                 (long)s_output->params.zero_point);
#endif

        uint32_t raw_sum = 0;
        uint32_t rgb_hash = 2166136261u;
        for (ssize_t p = 0; p < len; p++) {
            raw_sum += patch[p];
            rgb_hash ^= patch[p];
            rgb_hash *= 16777619u;
            write_input_pixel(p, patch[p]);
        }
        s_patch_rgb_hash[vector_index] = rgb_hash;

        uint32_t input_hash = 0;
        uint32_t q_sat_min = 0;
        uint32_t q_sat_max = 0;
        int q_min = 127;
        int q_max = -128;

#if APP_AI_INFER_DEBUG_PATCH_IO || APP_AI_INFER_DEBUG_DIAG
        if (s_input && s_input->data.raw) {
            uint32_t hash = 2166136261u;
            if (s_input->type == kTfLiteInt8) {
                for (size_t q = 0; q < s_input->bytes; q++) {
                    const uint8_t v = (uint8_t)s_input->data.raw[q];
                    const int qi = (int)s_input->data.int8[q];
                    hash ^= v;
                    hash *= 16777619u;
                    if (qi < q_min) q_min = qi;
                    if (qi > q_max) q_max = qi;
                    if (qi == -128) q_sat_min++;
                    if (qi == 127) q_sat_max++;
                }
            } else {
                for (size_t q = 0; q < s_input->bytes; q++) {
                    const uint8_t v = (uint8_t)s_input->data.raw[q];
                    hash ^= v;
                    hash *= 16777619u;
                }
            }
            input_hash = hash;
        }
#endif

#if APP_AI_INFER_DEBUG_PATCH_IO
        if (i < APP_AI_PATCH_MAX_COUNT) {
            s_debug_rgb_hashes[i] = rgb_hash;
            s_debug_input_hashes[i] = input_hash;
            s_debug_raw_sums[i] = raw_sum;
        }
#endif

#if APP_AI_INFER_DEBUG_DIAG
        if (i < 3 || i == count / 2 || i == count - 1) {
            int b0 = 0, b1 = 0, b2 = 0, b100 = 0, b1000 = 0;
            if (s_input->type == kTfLiteInt8) {
                b0 = s_input->data.int8[0];
                b1 = s_input->data.int8[1];
                b2 = s_input->data.int8[2];
                b100 = s_input->data.int8[100];
                b1000 = s_input->data.int8[1000];
            }
            ESP_LOGI(TAG, "[AI INPUT] patch=%u sum=%u hash=0x%08X b0=%d b1=%d b2=%d b100=%d b1000=%d",
                     (unsigned)i, (unsigned)raw_sum, (unsigned)input_hash,
                     b0, b1, b2, b100, b1000);
        }
#endif

        bool reused_input = false;
#if APP_AI_INFER_ENABLE_DEDUP
        reused_input = s_cached_input && s_cached_output_valid &&
                       s_cached_input_bytes == s_input->bytes &&
                       memcmp(s_cached_input, s_input->data.raw, s_input->bytes) == 0;
#endif
        int64_t invoke_us = 0;

#if APP_AI_INFER_ENABLE_DEDUP
        if (reused_input) {
            memcpy(s_ai_infer.vectors[vector_index].raw, s_cached_output, sizeof(s_cached_output));
            reused_inputs++;
            ESP_LOGI(TAG, "[AI DEDUP] patch=%u same quantized input as patch=%u; reused embedding",
                     (unsigned)i, (unsigned)s_cached_input_patch);
        } else
#endif
        {
#if APP_AI_INFER_DEBUG_DIAG
            if (i == APP_AI_INFER_DEBUG_PATCH_INDEX) {
                debug_log_tensor_checkpoint(0, "PRE_INVOKE");
            }
#endif
            if (esp_task_wdt_status(NULL) == ESP_OK) {
                esp_task_wdt_reset();
            }

            int64_t invoke_t0 = esp_timer_get_time();
            TfLiteStatus invoke_status = s_interpreter->Invoke();
            invoke_us = esp_timer_get_time() - invoke_t0;

            if (invoke_status != kTfLiteOk) {
                ESP_LOGE(TAG, "Invoke failed patch=%u", (unsigned)i);
                ret = ESP_FAIL;
                err_code = "invoke_failed";
                break;
            }

#if APP_AI_INFER_DEBUG_DIAG
            if (i == APP_AI_INFER_DEBUG_PATCH_INDEX) {
                const int checkpoint_tensors[] = {110, 111, 112, 113};
                for (size_t tensor = 0;
                     tensor < sizeof(checkpoint_tensors) / sizeof(checkpoint_tensors[0]);
                     tensor++) {
                    debug_log_tensor_checkpoint(checkpoint_tensors[tensor], "POST_INVOKE");
                }
            }
#endif

#if APP_AI_INFER_DEBUG_DIAG
            debug_trace_patch_pair((int)i);
#endif

            if (invoke_us > APP_AI_INVOKE_WARN_US) {
                ESP_LOGW(TAG, "[AI SLOW] patch=%u Invoke=%lld us", (unsigned)i, (long long)invoke_us);
            }

            for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
                s_ai_infer.vectors[vector_index].raw[d] = read_output_value((size_t)d);
            }
#if APP_AI_INFER_ENABLE_DEDUP
                 memcpy(s_cached_output, s_ai_infer.vectors[vector_index].raw,
                     sizeof(s_cached_output));
            s_cached_output_valid = true;
            s_cached_input_patch = i;
            if (s_cached_input && s_cached_input_bytes == s_input->bytes) {
                memcpy(s_cached_input, s_input->data.raw, s_input->bytes);
            }
#endif
        }

        float sum_sq = 0.0f;

        for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
            float val = s_ai_infer.vectors[vector_index].raw[d];
            sum_sq += val * val;
        }

        float norm = sqrtf(sum_sq);

        if (norm < 1e-12f) {
            ESP_LOGE(TAG, "[AI] Patch=%u ZERO VECTOR norm=%.12e raw_sum=%u",
                     (unsigned)i, (double)norm, (unsigned)raw_sum);
            ret = ESP_FAIL;
            err_code = "zero_feature_vector";
            break;
        }

        for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
            s_ai_infer.vectors[vector_index].values[d] =
                s_ai_infer.vectors[vector_index].raw[d] / norm;
        }

#if APP_AI_INFER_DEBUG_PATCH_IO
        /* Compare against ALL previous embeddings, not just the previous patch.
         * This catches non-adjacent collapse and also tells us whether the
         * duplicate came from equal quantized input or from the model/kernel. */
        bool exact_duplicate = false;
        uint32_t duplicate_of = 0;
        for (uint32_t j = 0; j < done; j++) {
            bool same = true;
            for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
                if (s_ai_infer.vectors[j].raw[d] !=
                    s_ai_infer.vectors[vector_index].raw[d]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                exact_duplicate = true;
                duplicate_of = j;
                break;
            }
        }

        if (i == APP_AI_INFER_DEBUG_PATCH_INDEX || i < 8 ||
            (i % 32) == 0 || exact_duplicate) {
            uint32_t raw_hash = 2166136261u;
            for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
                uint32_t bits = 0;
                memcpy(&bits, &s_ai_infer.vectors[vector_index].raw[d], sizeof(bits));
                raw_hash ^= (uint8_t)(bits);
                raw_hash *= 16777619u;
                raw_hash ^= (uint8_t)(bits >> 8);
                raw_hash *= 16777619u;
                raw_hash ^= (uint8_t)(bits >> 16);
                raw_hash *= 16777619u;
                raw_hash ^= (uint8_t)(bits >> 24);
                raw_hash *= 16777619u;
            }
            ESP_LOGI(TAG, "[PATCH IO] p=%u rgb_hash=0x%08X rgb_sum=%u qhash=0x%08X qmin=%d qmax=%d sat_lo=%u sat_hi=%u raw_hash=0x%08X norm=%.7f",
                     (unsigned)i, (unsigned)rgb_hash, (unsigned)raw_sum, (unsigned)input_hash,
                     q_min, q_max, (unsigned)q_sat_min, (unsigned)q_sat_max,
                     (unsigned)raw_hash, (double)norm);
        }

        if (exact_duplicate && !reused_input) {
            ESP_LOGE(TAG, "[AI EXACT DUP] patch=%u == patch=%u AFTER Invoke | rgb=%08X/%08X q=%08X/%08X sum=%u/%u",
                     (unsigned)i, (unsigned)duplicate_of,
                     (unsigned)rgb_hash, (unsigned)s_debug_rgb_hashes[duplicate_of],
                     (unsigned)input_hash, (unsigned)s_debug_input_hashes[duplicate_of],
                     (unsigned)raw_sum, (unsigned)s_debug_raw_sums[duplicate_of]);
        }
#endif

#if APP_AI_INFER_DEBUG_DIAG
    if (done > 0 && s_ai_infer.prev_raw_valid) {
            float max_diff = 0.0f;
            float diff_sum = 0.0f;
            int diff_count = 0;

            for (int d = 0; d < APP_AI_INFER_FEATURE_DIM; d++) {
                float diff = fabsf(s_ai_infer.vectors[vector_index].raw[d] -
                                   s_ai_infer.vectors[vector_index - 1].raw[d]);
                if (diff > max_diff) max_diff = diff;
                diff_sum += diff;
                if (diff > 1e-7f) diff_count++;
            }

            if (diff_count == 0 && !reused_input) {
                ESP_LOGE(TAG, "[AI RAW TRUNG] patch %u == %u max_diff=%.12g input=0x%08X sum=%u prev=0x%08X sum=%u",
                         (unsigned)i, (unsigned)(i - 1), (double)max_diff,
                         (unsigned)input_hash, (unsigned)raw_sum,
                         (unsigned)s_ai_infer.prev_input_hash, (unsigned)s_ai_infer.prev_raw_sum);
            } else if (i < 5) {
                ESP_LOGI(TAG, "[AI RAW] patch=%u vs prev diff_count=%d max=%.8g sum=%.8g",
                         (unsigned)i, diff_count, (double)max_diff, (double)diff_sum);
            }
        }

        memcpy(s_ai_infer.prev_raw, s_ai_infer.vectors[vector_index].raw,
               sizeof(s_ai_infer.prev_raw));
        s_ai_infer.prev_raw_valid = true;
        s_ai_infer.prev_raw_sum = raw_sum;
        s_ai_infer.prev_input_hash = input_hash;
#endif

        done = run_index + 1;
        ai_wdt_feed_and_yield();

        if (done == 1 || (done % 5) == 0 || done == run_count) {
            int64_t elapsed_us = esp_timer_get_time() - t0;
            uint32_t remaining = run_count - done;
            uint32_t eta_seconds = 0;

            if (done > 0) {
                eta_seconds = (uint32_t)((elapsed_us * remaining) / ((int64_t)done * 1000000LL));
            }

            ESP_LOGI(TAG, "[AI PROGRESS] %u/%u %.1f%% invoke=%.3f s elapsed=%.1f s eta~%u s",
                     (unsigned)done, (unsigned)run_count, 100.0 * (double)done / (double)run_count,
                     (double)invoke_us / 1000000.0, (double)elapsed_us / 1000000.0, (unsigned)eta_seconds);
        }
    }

    int64_t t1 = esp_timer_get_time();
    s_ai_infer.last_run_us = t1 - t0;
    s_ai_infer.count = done;
    s_ai_infer.last_run_ok = (ret == ESP_OK);

    if (ret != ESP_OK && err_code) {
        snprintf(s_ai_infer.last_error, sizeof(s_ai_infer.last_error), "%s", err_code);
    } else {
        s_ai_infer.last_error[0] = '\0';
    }

    if (ret == ESP_OK) {
        log_vector_diversity(done);
    }

    ESP_LOGI(TAG, "[AI DONE] %u/%u status=%s elapsed=%.1f s",
             (unsigned)done, (unsigned)run_count,
             ret == ESP_OK ? "ok" : (err_code ? err_code : "failed"),
             (double)s_ai_infer.last_run_us / 1000000.0);

#if APP_AI_INFER_ENABLE_DEDUP
    ESP_LOGI(TAG, "[AI DEDUP] reused=%u/%u patches", (unsigned)reused_inputs, (unsigned)done);
#else
    (void)reused_inputs;
#endif

    if (added_to_wdt) {
        esp_task_wdt_delete(NULL);
    }

    xSemaphoreGive(s_ai_infer.lock);

    return ret;
}

/* ================================================================
 * GET COUNT
 * ================================================================ */

uint32_t app_ai_infer_get_count(void)
{
    if (!s_ai_infer.lock) return 0;

    xSemaphoreTake(s_ai_infer.lock, portMAX_DELAY);
    uint32_t count = s_ai_infer.last_run_ok ? s_ai_infer.count : 0;
    xSemaphoreGive(s_ai_infer.lock);

    return count;
}

/* ================================================================
 * GET VECTOR
 * ================================================================ */

esp_err_t app_ai_infer_get_vector(uint32_t index, float *out, size_t out_len)
{
    if (!out || out_len < APP_AI_INFER_FEATURE_DIM) return ESP_ERR_INVALID_ARG;
    if (!s_ai_infer.lock) return ESP_ERR_INVALID_STATE;

    esp_err_t ret = ESP_ERR_NOT_FOUND;

    xSemaphoreTake(s_ai_infer.lock, portMAX_DELAY);

    if (s_ai_infer.last_run_ok && index < s_ai_infer.count) {
        memcpy(out, s_ai_infer.vectors[index].values, sizeof(float) * APP_AI_INFER_FEATURE_DIM);
        ret = ESP_OK;
    }

    xSemaphoreGive(s_ai_infer.lock);

    return ret;
}

/* ================================================================
 * STATUS JSON
 * ================================================================ */

esp_err_t app_ai_infer_get_status_json(char *buf, size_t len, bool include_vectors)
{
    if (!buf || len == 0) return ESP_ERR_INVALID_ARG;

    cJSON *root = cJSON_CreateObject();
    if (!root) return ESP_ERR_NO_MEM;

    bool ready = s_ai_infer.ready;

    cJSON_AddBoolToObject(root, "ready", ready);
    cJSON_AddNumberToObject(root, "feature_dim", APP_AI_INFER_FEATURE_DIM);

    if (!ready || !s_ai_infer.lock) {
        cJSON_AddBoolToObject(root, "ok", false);
        cJSON_AddStringToObject(root, "error", ready ? "not_initialized" : "model_not_ready");

        bool printed = cJSON_PrintPreallocated(root, buf, (int)len, false);
        cJSON_Delete(root);

        return printed ? ESP_OK : ESP_ERR_NO_MEM;
    }

    xSemaphoreTake(s_ai_infer.lock, portMAX_DELAY);

    bool ok = s_ai_infer.last_run_ok;
    uint32_t count = s_ai_infer.count;
    int64_t run_us = s_ai_infer.last_run_us;

    char err[sizeof(s_ai_infer.last_error)];
    memcpy(err, s_ai_infer.last_error, sizeof(err));

    cJSON_AddBoolToObject(root, "ok", ok);
    cJSON_AddNumberToObject(root, "count", count);
    cJSON_AddNumberToObject(root, "last_run_ms", (double)(run_us / 1000.0));

    if (!ok && err[0]) {
        cJSON_AddStringToObject(root, "error", err);
    }

    if (include_vectors && ok && count > 0) {
        cJSON *vectors = cJSON_AddArrayToObject(root, "vectors");
        if (vectors) {
            for (uint32_t i = 0; i < count; i++) {
                cJSON *vec = cJSON_CreateFloatArray(s_ai_infer.vectors[i].values, APP_AI_INFER_FEATURE_DIM);
                if (vec) {
                    cJSON_AddItemToArray(vectors, vec);
                }
            }
        }
    }

    xSemaphoreGive(s_ai_infer.lock);

    bool printed = cJSON_PrintPreallocated(root, buf, (int)len, false);
    cJSON_Delete(root);

    return printed ? ESP_OK : ESP_ERR_NO_MEM;
}