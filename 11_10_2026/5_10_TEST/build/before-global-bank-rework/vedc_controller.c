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
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_system.h"

#include "app_protocol.h"
#include "app_ai_patch.h"
#include "app_ai_infer.h"
#include "app_ai_coreset.h"
#include "app_remote.h"
#include "app_uvc_manager.h"
#include "cJSON.h"
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
static volatile bool g_model_trained = false;

/* Memory lưu Feature Vector 5 mẫu ảnh */
static float *g_train_features[LEARN_SAMPLES] = {NULL};
static uint8_t *g_train_appearance[LEARN_SAMPLES] = {NULL};
static uint32_t g_train_feature_count[LEARN_SAMPLES] = {0};

static uint8_t g_check_board = 1;
static int64_t g_check_board_start_us = 0;

static QueueHandle_t g_frame_queue = NULL;
static volatile bool g_runtime_ready = false;

/* Struct kết quả kiểm tra */
typedef struct {
    uint8_t  result;
    uint8_t  error_code;
    uint16_t patch_total;
    uint16_t patch_fail_count;
    uint32_t elapsed_ms;
    float    max_dmin; /* điểm lỗi lớn nhất; tên trường giữ tương thích UART */
} visual_check_result_t;

typedef struct {
    uint8_t  result;
    float    imax_measured;
    float    imax_threshold;
    float    iavg_measured;
    float    iavg_threshold;
    uint32_t elapsed_ms;
} power_check_result_t;


/* ================================================================
 * WEB/UI STATE
 *
 * The web app is observation-only.  STM32 is the master controller: the
 * ESP32-S3 changes stage whenever it receives a valid STM32 frame, and the
 * browser only reads the snapshot below.  This keeps the web UI from ever
 * changing the production sequence.
 * ================================================================ */
typedef enum {
    VEDC_UI_STANDBY = 0,
    VEDC_UI_LEARN_WAIT_VISUAL,
    VEDC_UI_LEARN_CAPTURING,
    VEDC_UI_LEARN_WAIT_POWER,
    VEDC_UI_LEARN_RECEIVING_POWER,
    VEDC_UI_LEARN_READY_TRAIN,
    VEDC_UI_TRAINING,
    VEDC_UI_TRAINED,
    VEDC_UI_CHECK_WAIT_PCB,
    VEDC_UI_CHECK_VISUAL,
    VEDC_UI_CHECK_VISUAL_PASS,
    VEDC_UI_CHECK_VISUAL_FAIL,
    VEDC_UI_CHECK_POWER,
    VEDC_UI_CHECK_POWER_PASS,
    VEDC_UI_CHECK_POWER_FAIL,
    VEDC_UI_ERROR
} vedc_ui_stage_t;

static SemaphoreHandle_t g_ui_lock = NULL;
static vedc_ui_stage_t g_ui_stage = VEDC_UI_STANDBY;
static uint32_t g_ui_event_seq = 0;
static uint32_t g_last_capture_id = 0;
static uint8_t g_ui_board = 0;
static uint8_t g_ui_learn_visual_count = 0;
static uint8_t g_ui_learn_power_count = 0;
static uint8_t g_ui_visual_working = 0;
static uint8_t g_ui_power_working = 0;
static uint8_t g_ui_visual_result = 0xFF;
static uint8_t g_ui_power_result = 0xFF;
static uint8_t g_ui_final_result = 0xFF;
static uint8_t g_ui_error_code = 0;
static char g_ui_last_event[96] = "Sẵn sàng - chờ STM32";

static uint32_t g_train_bank_count = 0;
static float g_train_image_threshold = 0.0f;
static float g_train_imax_max_dev = 0.0f;
static float g_train_iavg_max_dev = 0.0f;
static uint32_t g_train_elapsed_ms = 0;

/* Dữ liệu dòng của lần CHECK gần nhất để web hiển thị trực tiếp. */
static float g_ui_imax_measured = 0.0f;
static float g_ui_imax_threshold = 0.0f;
static float g_ui_iavg_measured = 0.0f;
static float g_ui_iavg_threshold = 0.0f;
static uint32_t g_ui_power_elapsed_ms = 0;

/* Kết quả ngoại quan cuối của PCB hiện tại. Scores giữ nguyên mapping row-major
 * với grid 40x40 của app_ai_patch (max 20x15 = 300 patch). */
static uint32_t g_visual_capture_id = 0;
static uint32_t g_visual_cols = 0;
static uint32_t g_visual_rows = 0;
static uint32_t g_visual_crop_width = 0;
static uint32_t g_visual_crop_height = 0;
static int g_visual_crop_x = 0;
static int g_visual_crop_y = 0;
static uint16_t g_visual_patch_total = 0;
static uint16_t g_visual_patch_fail_count = 0;
static uint32_t g_visual_elapsed_ms = 0;
static float g_visual_max_dmin = 0.0f;
static float g_visual_threshold = 0.0f;
static float g_visual_ai_max = 0.0f;
static float g_visual_appearance_max = 0.0f;
static uint32_t g_visual_ai_blind_patches = 0;
static float g_visual_scores[APP_AI_PATCH_MAX_COUNT];

/* ---- TỔNG KẾT TỪNG PCB ------------------------------------------------
 * ui_reset_current_check() xóa trạng thái "đang đo" mỗi khi STM32 gửi CHECK_NEXT,
 * nhưng giữ dữ liệu inspection gần nhất cho web. Tổng kết và lịch sử PCB
 * được lưu riêng ở đây. Mọi truy cập phải giữ
 * g_ui_lock. */
#define CHECK_HISTORY_MAX 8
typedef struct {
    uint8_t  board;
    uint8_t  visual;                 /* RESULT_PASS / RESULT_FAIL */
    uint8_t  power;                  /* RESULT_PASS / RESULT_FAIL / 0xFF = không đo */
    uint8_t  final_result;           /* RESULT_PASS / RESULT_FAIL */
    uint16_t patch_total, patch_fail;
    uint16_t visual_ms, power_ms;
    float    max_dmin, v_threshold;
    float    imax, imax_mean, imax_dev;
    float    iavg, iavg_mean, iavg_dev;
} check_summary_t;

static check_summary_t g_check_history[CHECK_HISTORY_MAX];
static uint8_t  g_check_hist_count = 0;
static uint8_t  g_check_hist_head  = 0;
static check_summary_t g_ui_summary;
static bool     g_ui_summary_valid = false;
static uint32_t g_check_total = 0, g_check_pass = 0, g_check_fail = 0;

/* Cache transaction để đáp ứng đúng yêu cầu RETRY/IDEMPOTENT của STM32.
 * Khi STM32 mất ACK/RESULT và gửi lại cùng lệnh, ESP32 không được chụp/
 * đo/tính lại hoặc làm tăng index lần nữa. */
static uint32_t g_capture_elapsed_ms[LEARN_SAMPLES];
static bool g_train_done_valid = false;
static uint8_t g_train_done_data[26];
static bool g_visual_cache_valid = false;
static uint8_t g_visual_cache_board = 0;
static uint8_t g_visual_cache_data[11];
static bool g_power_cache_valid = false;
static uint8_t g_power_cache_board = 0;
static uint8_t g_power_cache_request[9];
static uint8_t g_power_cache_data[19];
static bool g_next_cache_valid = false;
static uint8_t g_next_cache_board = 0;

static const char *ui_stage_name(vedc_ui_stage_t stage)
{
    switch (stage) {
    case VEDC_UI_STANDBY:              return "STANDBY";
    case VEDC_UI_LEARN_WAIT_VISUAL:   return "LEARN_WAIT_VISUAL";
    case VEDC_UI_LEARN_CAPTURING:     return "LEARN_CAPTURING";
    case VEDC_UI_LEARN_WAIT_POWER:    return "LEARN_WAIT_POWER";
    case VEDC_UI_LEARN_RECEIVING_POWER:return "LEARN_POWER_RESULT";
    case VEDC_UI_LEARN_READY_TRAIN:   return "LEARN_READY_TRAIN";
    case VEDC_UI_TRAINING:            return "TRAINING";
    case VEDC_UI_TRAINED:             return "TRAINED";
    case VEDC_UI_CHECK_WAIT_PCB:      return "CHECK_WAIT_PCB";
    case VEDC_UI_CHECK_VISUAL:        return "CHECK_VISUAL";
    case VEDC_UI_CHECK_VISUAL_PASS:   return "CHECK_VISUAL_PASS";
    case VEDC_UI_CHECK_VISUAL_FAIL:   return "CHECK_VISUAL_FAIL";
    case VEDC_UI_CHECK_POWER:         return "CHECK_POWER";
    case VEDC_UI_CHECK_POWER_PASS:    return "CHECK_POWER_PASS";
    case VEDC_UI_CHECK_POWER_FAIL:    return "CHECK_POWER_FAIL";
    case VEDC_UI_ERROR:               return "ERROR";
    default:                          return "UNKNOWN";
    }
}

__attribute__((unused)) static void ui_set_event(vedc_ui_stage_t stage, uint8_t board, const char *event)
{
    if (!g_ui_lock) return;
    xSemaphoreTake(g_ui_lock, portMAX_DELAY);
    g_ui_stage = stage;
    g_ui_board = board;
    if (event) {
        snprintf(g_ui_last_event, sizeof(g_ui_last_event), "%s", event);
    }
    ++g_ui_event_seq;
    xSemaphoreGive(g_ui_lock);
}

static uint32_t ui_next_capture_id(void)
{
    if (!g_ui_lock) return 0;
    xSemaphoreTake(g_ui_lock, portMAX_DELAY);
    ++g_last_capture_id;
    uint32_t id = g_last_capture_id;
    xSemaphoreGive(g_ui_lock);
    return id;
}

static uint32_t ui_get_last_capture_id(void)
{
    if (!g_ui_lock) return 0;
    xSemaphoreTake(g_ui_lock, portMAX_DELAY);
    uint32_t id = g_last_capture_id;
    xSemaphoreGive(g_ui_lock);
    return id;
}

static void ui_reset_current_check(bool clear_inspection)
{
    if (!g_ui_lock) return;
    xSemaphoreTake(g_ui_lock, portMAX_DELAY);
    /* CHECK_NEXT comes immediately after a visual FAIL. Keep the last
     * completed inspection until the next result so a browser that missed
     * that short interval can still fetch its scores and draw its patches. */
    if (clear_inspection) {
        g_visual_capture_id = 0;
        g_visual_cols = g_visual_rows = 0;
        g_visual_crop_width = g_visual_crop_height = 0;
        g_visual_crop_x = g_visual_crop_y = 0;
        g_visual_patch_total = g_visual_patch_fail_count = 0;
        g_visual_elapsed_ms = 0;
        g_visual_max_dmin = 0.0f;
        g_visual_threshold = 0.0f;
        g_visual_ai_max = g_visual_appearance_max = 0.0f;
        g_visual_ai_blind_patches = 0;
        memset(g_visual_scores, 0, sizeof(g_visual_scores));
    }
    g_ui_visual_result = 0xFF;
    g_ui_power_result = 0xFF;
    g_ui_final_result = 0xFF;
    g_ui_visual_working = 0;
    g_ui_power_working = 0;
    /* Số đo dòng của PCB trước không được còn lại khi đang đo PCB kế tiếp. */
    g_ui_imax_measured = g_ui_imax_threshold = 0.0f;
    g_ui_iavg_measured = g_ui_iavg_threshold = 0.0f;
    g_ui_power_elapsed_ms = 0;
    xSemaphoreGive(g_ui_lock);
}

/* BEGIN SUMMARY HELPERS */
static uint16_t summary_u16(uint32_t v) { return (v > 65535u) ? 65535u : (uint16_t)v; }

/* Gọi khi ĐÃ giữ g_ui_lock. */
static void check_history_clear_locked(void)
{
    memset(g_check_history, 0, sizeof(g_check_history));
    memset(&g_ui_summary, 0, sizeof(g_ui_summary));
    g_check_hist_count = 0;
    g_check_hist_head = 0;
    g_ui_summary_valid = false;
    g_check_total = g_check_pass = g_check_fail = 0;
}

/* Gọi khi ĐÃ giữ g_ui_lock, sau khi g_visual_* đã chứa kết quả ngoại quan của PCB. */
static void check_summary_commit_locked(uint8_t board, uint8_t visual, uint8_t power, uint8_t final_result,
                                        uint32_t power_ms, float imax, float iavg)
{
    check_summary_t s;
    memset(&s, 0, sizeof(s));
    s.board = board;
    s.visual = visual;
    s.power = power;
    s.final_result = final_result;
    s.patch_total = g_visual_patch_total;
    s.patch_fail = g_visual_patch_fail_count;
    s.visual_ms = summary_u16(g_visual_elapsed_ms);
    s.power_ms = summary_u16(power_ms);
    s.max_dmin = g_visual_max_dmin;
    s.v_threshold = g_visual_threshold;
    s.imax = imax;
    s.iavg = iavg;
    s.imax_mean = g_imax_mean;
    s.imax_dev = g_imax_max_dev;
    s.iavg_mean = g_iavg_mean;
    s.iavg_dev = g_iavg_max_dev;

    g_ui_summary = s;
    g_ui_summary_valid = true;
    g_check_history[g_check_hist_head] = s;
    g_check_hist_head = (uint8_t)((g_check_hist_head + 1u) % CHECK_HISTORY_MAX);
    if (g_check_hist_count < CHECK_HISTORY_MAX) g_check_hist_count++;
    g_check_total++;
    if (final_result == RESULT_PASS) g_check_pass++; else g_check_fail++;
}

static double summary_r3(float v) { return round((double)v * 1000.0) / 1000.0; }

static void summary_fill_json(cJSON *o, const check_summary_t *s)
{
    cJSON_AddNumberToObject(o, "board", s->board);
    cJSON_AddNumberToObject(o, "visual", s->visual);
    cJSON_AddNumberToObject(o, "power", s->power);
    cJSON_AddNumberToObject(o, "final", s->final_result);
    cJSON_AddNumberToObject(o, "patch_total", s->patch_total);
    cJSON_AddNumberToObject(o, "patch_fail", s->patch_fail);
    cJSON_AddNumberToObject(o, "visual_ms", s->visual_ms);
    cJSON_AddNumberToObject(o, "power_ms", s->power_ms);
    cJSON_AddNumberToObject(o, "max_dmin", summary_r3(s->max_dmin));
    cJSON_AddNumberToObject(o, "v_threshold", summary_r3(s->v_threshold));
    cJSON_AddNumberToObject(o, "imax", summary_r3(s->imax));
    cJSON_AddNumberToObject(o, "imax_mean", summary_r3(s->imax_mean));
    cJSON_AddNumberToObject(o, "imax_dev", summary_r3(s->imax_dev));
    cJSON_AddNumberToObject(o, "iavg", summary_r3(s->iavg));
    cJSON_AddNumberToObject(o, "iavg_mean", summary_r3(s->iavg_mean));
    cJSON_AddNumberToObject(o, "iavg_dev", summary_r3(s->iavg_dev));
}
/* END SUMMARY HELPERS */

static void ui_set_error(uint8_t error_code, const char *event)
{
    if (!g_ui_lock) return;
    xSemaphoreTake(g_ui_lock, portMAX_DELAY);
    g_ui_stage = VEDC_UI_ERROR;
    g_ui_error_code = error_code;
    if (event) snprintf(g_ui_last_event, sizeof(g_ui_last_event), "%s", event);
    ++g_ui_event_seq;
    xSemaphoreGive(g_ui_lock);
}

static void ui_snapshot_copy(char *buf, size_t len)
{
    if (!buf || len == 0) return;

    cJSON *root = cJSON_CreateObject();
    if (!root) {
        snprintf(buf, len, "{\"stage\":\"ERROR\",\"error\":\"no_mem_json\"}");
        return;
    }

    vedc_ui_stage_t stage;
    uint32_t event_seq, capture_id, board, visual_capture_id;
    uint32_t learn_visual_count, learn_power_count;
    uint8_t visual_result, power_result, final_result, error_code;
    uint8_t visual_working, power_working;
    char last_event[96];
    uint32_t train_bank_count, train_elapsed_ms;
    float train_threshold, train_imax_dev, train_iavg_dev;
    uint32_t cols, rows, crop_w, crop_h;
    int crop_x, crop_y;
    uint16_t patch_total, patch_fail;
    uint32_t visual_elapsed;
    float visual_max_dmin, visual_threshold;
    float visual_ai_max, visual_appearance_max;
    uint32_t visual_ai_blind_patches;
    float imax_measured, imax_threshold, iavg_measured, iavg_threshold;
    uint32_t power_elapsed_ms;
    bool model_trained;
    float scores[APP_AI_PATCH_MAX_COUNT];
    check_summary_t hist[CHECK_HISTORY_MAX];
    check_summary_t summ;
    uint8_t hist_count, hist_head;
    bool summ_valid;
    uint32_t tot_checked, tot_pass, tot_fail;

    if (g_ui_lock) xSemaphoreTake(g_ui_lock, portMAX_DELAY);
    stage = g_ui_stage;
    event_seq = g_ui_event_seq;
    capture_id = g_last_capture_id;
    board = g_ui_board;
    learn_visual_count = g_ui_learn_visual_count;
    learn_power_count = g_ui_learn_power_count;
    visual_result = g_ui_visual_result;
    power_result = g_ui_power_result;
    final_result = g_ui_final_result;
    error_code = g_ui_error_code;
    visual_working = g_ui_visual_working;
    power_working = g_ui_power_working;
    snprintf(last_event, sizeof(last_event), "%s", g_ui_last_event);
    train_bank_count = g_train_bank_count;
    train_elapsed_ms = g_train_elapsed_ms;
    train_threshold = g_train_image_threshold;
    train_imax_dev = g_train_imax_max_dev;
    train_iavg_dev = g_train_iavg_max_dev;
    imax_measured = g_ui_imax_measured;
    imax_threshold = g_ui_imax_threshold;
    iavg_measured = g_ui_iavg_measured;
    iavg_threshold = g_ui_iavg_threshold;
    power_elapsed_ms = g_ui_power_elapsed_ms;
    model_trained = g_model_trained;
    visual_capture_id = g_visual_capture_id;
    cols = g_visual_cols;
    rows = g_visual_rows;
    crop_w = g_visual_crop_width;
    crop_h = g_visual_crop_height;
    crop_x = g_visual_crop_x;
    crop_y = g_visual_crop_y;
    patch_total = g_visual_patch_total;
    patch_fail = g_visual_patch_fail_count;
    visual_elapsed = g_visual_elapsed_ms;
    visual_max_dmin = g_visual_max_dmin;
    visual_threshold = g_visual_threshold;
    visual_ai_max = g_visual_ai_max;
    visual_appearance_max = g_visual_appearance_max;
    visual_ai_blind_patches = g_visual_ai_blind_patches;
    memcpy(scores, g_visual_scores, sizeof(scores));
    memcpy(hist, g_check_history, sizeof(hist));
    summ = g_ui_summary;
    summ_valid = g_ui_summary_valid;
    hist_count = g_check_hist_count;
    hist_head = g_check_hist_head;
    tot_checked = g_check_total;
    tot_pass = g_check_pass;
    tot_fail = g_check_fail;
    if (g_ui_lock) xSemaphoreGive(g_ui_lock);

    cJSON_AddStringToObject(root, "stage", ui_stage_name(stage));
    cJSON_AddNumberToObject(root, "event_seq", event_seq);
    cJSON_AddNumberToObject(root, "uptime_s", (double)(esp_timer_get_time() / 1000000));
    {
        static const char *const names[] = {"UNKNOWN","POWERON","EXT","SW","PANIC","INT_WDT","TASK_WDT",
                                            "WDT","DEEPSLEEP","BROWNOUT","SDIO","USB","JTAG","EFUSE",
                                            "PWR_GLITCH","CPU_LOCKUP"};
        unsigned r = (unsigned)esp_reset_reason();
        cJSON_AddStringToObject(root, "reset_reason", r < sizeof(names)/sizeof(names[0]) ? names[r] : "?");
    }
    cJSON_AddStringToObject(root, "last_event", last_event);
    cJSON_AddNumberToObject(root, "board", board);
    cJSON_AddNumberToObject(root, "last_capture_id", capture_id);
    cJSON_AddBoolToObject(root, "model_trained", model_trained);
    cJSON_AddBoolToObject(root, "camera", app_uvc_is_streaming());
    cJSON_AddBoolToObject(root, "ai_ready", app_ai_infer_is_ready());
    cJSON_AddBoolToObject(root, "bank_ready", app_ai_coreset_is_ready());

    cJSON *learn = cJSON_CreateObject();
    if (learn) {
        cJSON_AddNumberToObject(learn, "visual_done", learn_visual_count);
        cJSON_AddNumberToObject(learn, "power_done", learn_power_count);
        cJSON_AddNumberToObject(learn, "total", LEARN_SAMPLES);
        cJSON_AddNumberToObject(learn, "visual_working", visual_working);
        cJSON_AddNumberToObject(learn, "power_working", power_working);
        cJSON_AddItemToObject(root, "learn", learn);
    }

    cJSON *train = cJSON_CreateObject();
    if (train) {
        cJSON_AddBoolToObject(train, "done", model_trained);
        cJSON_AddNumberToObject(train, "bank_count", train_bank_count);
        cJSON_AddNumberToObject(train, "image_threshold", train_threshold);
        cJSON_AddNumberToObject(train, "imax_max_dev", train_imax_dev);
        cJSON_AddNumberToObject(train, "iavg_max_dev", train_iavg_dev);
        cJSON_AddNumberToObject(train, "elapsed_ms", train_elapsed_ms);
        cJSON_AddItemToObject(root, "train", train);
    }

    cJSON *check = cJSON_CreateObject();
    if (check) {
        cJSON_AddNumberToObject(check, "visual_result", visual_result);
        cJSON_AddNumberToObject(check, "power_result", power_result);
        cJSON_AddNumberToObject(check, "final_result", final_result);
        cJSON_AddNumberToObject(check, "capture_id", visual_capture_id);
        cJSON_AddNumberToObject(check, "patch_total", patch_total);
        cJSON_AddNumberToObject(check, "patch_fail_count", patch_fail);
        cJSON_AddNumberToObject(check, "elapsed_ms", visual_elapsed);
        cJSON_AddNumberToObject(check, "max_dmin", visual_max_dmin);
        cJSON_AddNumberToObject(check, "threshold", visual_threshold);
        cJSON_AddNumberToObject(check, "ai_max_score", visual_ai_max);
        cJSON_AddNumberToObject(check, "appearance_max_score", visual_appearance_max);
        cJSON_AddNumberToObject(check, "ai_blind_patch_count", visual_ai_blind_patches);
        cJSON_AddNumberToObject(check, "cols", cols);
        cJSON_AddNumberToObject(check, "rows", rows);
        cJSON_AddNumberToObject(check, "crop_width", crop_w);
        cJSON_AddNumberToObject(check, "crop_height", crop_h);
        cJSON_AddNumberToObject(check, "crop_x", crop_x);
        cJSON_AddNumberToObject(check, "crop_y", crop_y);
        cJSON_AddNumberToObject(check, "imax_measured", imax_measured);
        cJSON_AddNumberToObject(check, "imax_threshold", imax_threshold);
        cJSON_AddNumberToObject(check, "iavg_measured", iavg_measured);
        cJSON_AddNumberToObject(check, "iavg_threshold", iavg_threshold);
        cJSON_AddNumberToObject(check, "power_elapsed_ms", power_elapsed_ms);
        cJSON_AddNumberToObject(check, "imax_mean", summary_r3(g_imax_mean));
        cJSON_AddNumberToObject(check, "imax_dev", summary_r3(g_imax_max_dev));
        cJSON_AddNumberToObject(check, "iavg_mean", summary_r3(g_iavg_mean));
        cJSON_AddNumberToObject(check, "iavg_dev", summary_r3(g_iavg_max_dev));

        cJSON *score_array = cJSON_CreateArray();
        if (score_array) {
            uint32_t n = cols * rows;
            if (n > APP_AI_PATCH_MAX_COUNT) n = APP_AI_PATCH_MAX_COUNT;
            for (uint32_t i = 0; i < n; ++i) {
                cJSON_AddItemToArray(score_array, cJSON_CreateNumber(scores[i]));
            }
            cJSON_AddItemToObject(check, "scores", score_array);
        }
        cJSON_AddItemToObject(root, "check", check);
    }

    /* Tổng kết PCB gần nhất + lịch sử (cũ -> mới) của phiên CHECK hiện tại. */
    cJSON *summary = cJSON_CreateObject();
    if (summary) {
        cJSON_AddBoolToObject(summary, "valid", summ_valid);
        cJSON_AddNumberToObject(summary, "total", tot_checked);
        cJSON_AddNumberToObject(summary, "pass", tot_pass);
        cJSON_AddNumberToObject(summary, "fail", tot_fail);
        if (summ_valid) summary_fill_json(summary, &summ);
        cJSON_AddItemToObject(root, "summary", summary);
    }
    cJSON *history = cJSON_CreateArray();
    if (history) {
        for (uint8_t i = 0; i < hist_count; ++i) {
            uint8_t idx = (uint8_t)((hist_head + CHECK_HISTORY_MAX - hist_count + i) % CHECK_HISTORY_MAX);
            cJSON *o = cJSON_CreateObject();
            if (!o) break;
            summary_fill_json(o, &hist[idx]);
            cJSON_AddItemToArray(history, o);
        }
        cJSON_AddItemToObject(root, "history", history);
    }

    cJSON *error = cJSON_CreateObject();
    if (error) {
        cJSON_AddBoolToObject(error, "active", stage == VEDC_UI_ERROR);
        cJSON_AddNumberToObject(error, "code", error_code);
        cJSON_AddItemToObject(root, "error", error);
    }

    char *json = cJSON_PrintUnformatted(root);
    if (json) {
        snprintf(buf, len, "%s", json);
        cJSON_free(json);
    } else {
        snprintf(buf, len, "{\"stage\":\"ERROR\",\"error\":\"json_serialize_failed\"}");
    }
    cJSON_Delete(root);
}

esp_err_t vedc_controller_get_status_json(char *buf, size_t len)
{
    if (!buf || len == 0) return ESP_ERR_INVALID_ARG;
    if (!g_ui_lock) return ESP_ERR_INVALID_STATE;
    ui_snapshot_copy(buf, len);
    return (strlen(buf) + 1 < len) ? ESP_OK : ESP_ERR_NO_MEM;
}

/* Reset dữ liệu học */
static void clear_learn_storage(void) {
    memset(g_learn_imax, 0, sizeof(g_learn_imax));
    memset(g_learn_iavg, 0, sizeof(g_learn_iavg));
    for (uint32_t i = 0; i < LEARN_SAMPLES; ++i) {
        if (g_train_features[i]) {
            heap_caps_free(g_train_features[i]);
            g_train_features[i] = NULL;
        }
        if (g_train_appearance[i]) {
            heap_caps_free(g_train_appearance[i]);
            g_train_appearance[i] = NULL;
        }
        g_train_feature_count[i] = 0;
    }
    app_ai_coreset_reset();
    app_ai_patch_reset_registration_reference();

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

    memset(g_capture_elapsed_ms, 0, sizeof(g_capture_elapsed_ms));
    g_train_done_valid = false;
    memset(g_train_done_data, 0, sizeof(g_train_done_data));
    g_visual_cache_valid = false;
    g_visual_cache_board = 0;
    memset(g_visual_cache_data, 0, sizeof(g_visual_cache_data));
    g_power_cache_valid = false;
    g_power_cache_board = 0;
    memset(g_power_cache_request, 0, sizeof(g_power_cache_request));
    memset(g_power_cache_data, 0, sizeof(g_power_cache_data));
    g_next_cache_valid = false;
    g_next_cache_board = 0;

    if (g_ui_lock) {
        xSemaphoreTake(g_ui_lock, portMAX_DELAY);
        g_ui_stage = VEDC_UI_STANDBY;
        g_ui_board = 0;
        g_ui_learn_visual_count = 0;
        g_ui_learn_power_count = 0;
        g_ui_visual_working = 0;
        g_ui_power_working = 0;
        g_ui_visual_result = 0xFF;
        g_ui_power_result = 0xFF;
        g_ui_final_result = 0xFF;
        g_ui_error_code = 0;
        g_train_bank_count = 0;
        g_train_image_threshold = 0.0f;
        g_train_imax_max_dev = 0.0f;
        g_train_iavg_max_dev = 0.0f;
        g_train_elapsed_ms = 0;
        g_ui_imax_measured = 0.0f;
        g_ui_imax_threshold = 0.0f;
        g_ui_iavg_measured = 0.0f;
        g_ui_iavg_threshold = 0.0f;
        g_ui_power_elapsed_ms = 0;
        g_visual_capture_id = 0;
        g_visual_cols = g_visual_rows = 0;
        g_visual_crop_width = g_visual_crop_height = 0;
        g_visual_crop_x = g_visual_crop_y = 0;
        g_visual_patch_total = g_visual_patch_fail_count = 0;
        g_visual_elapsed_ms = 0;
        g_visual_max_dmin = 0.0f;
        g_visual_threshold = 0.0f;
        g_visual_ai_max = g_visual_appearance_max = 0.0f;
        g_visual_ai_blind_patches = 0;
        memset(g_visual_scores, 0, sizeof(g_visual_scores));
        check_history_clear_locked();
        ++g_ui_event_seq;
        snprintf(g_ui_last_event, sizeof(g_ui_last_event), "Đã reset Learn - chờ mẫu đầu tiên");
        xSemaphoreGive(g_ui_lock);
    }
}

/* Chờ các module camera + AI khởi tạo xong. UART1 vẫn được mở sớm để không
 * mất CMD từ STM32, nhưng không cho worker đụng vào UVC/AI trước main(). */
static esp_err_t wait_runtime_ready(uint32_t timeout_ms)
{
    const TickType_t step = pdMS_TO_TICKS(50);
    uint32_t waited = 0;
    while (!g_runtime_ready && waited < timeout_ms) {
        vTaskDelay(step);
        waited += 50;
    }
    return g_runtime_ready ? ESP_OK : ESP_ERR_INVALID_STATE;
}

/* Chụp ảnh và trích xuất Feature Block */
static esp_err_t capture_real_feature_block(float **out, uint32_t *count, uint32_t *out_elapsed_ms) {
    if (!out || !count) return ESP_ERR_INVALID_ARG;
    *out = NULL;
    *count = 0;
    if (out_elapsed_ms) *out_elapsed_ms = 0;

    int64_t t_start_us = esp_timer_get_time();

    esp_err_t ready_ret = wait_runtime_ready(60000);
    if (ready_ret != ESP_OK) {
        ESP_LOGE(TAG, "Camera/AI chưa sẵn sàng sau 60s");
        return ready_ret;
    }

    /* CAPTURE luôn do STM32 trigger. Nếu browser chưa mở /stream thì vẫn
     * phải chủ động bật UVC stream để capture không phụ thuộc Web UI. */
    esp_err_t cam_ret = app_uvc_set_enabled(true);
    if (cam_ret != ESP_OK && cam_ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGW(TAG, "Không bật được UVC stream trước CAPTURE: %s", esp_err_to_name(cam_ret));
    }
    /* set_enabled() chỉ gửi lệnh (bất đồng bộ). Chờ stream thực sự chạy (tối đa 6 s)
     * trước khi chụp, nếu không CAPTURE đầu tiên sẽ trượt "camera_not_streaming". */
    for (int w = 0; w < 120 && !app_uvc_is_streaming(); ++w) {
        vTaskDelay(pdMS_TO_TICKS(50));
    }

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
    uint32_t capture_id = ui_next_capture_id();
    app_ai_patch_mark_jpeg_capture(capture_id);
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
        out->error_code = (err == ESP_ERR_INVALID_STATE || err == ESP_ERR_TIMEOUT) ? ERR_CAMERA_FAIL : ERR_FEATURE_FAIL;
        return;
    }

    const uint32_t bank_count = app_ai_coreset_get_bank_count();
    const float threshold = app_ai_coreset_get_threshold();
    if (!app_ai_coreset_is_ready() || bank_count == 0 || threshold <= 0.0f) {
        heap_caps_free(features);
        out->error_code = ERR_TRAIN_FAIL;
        return;
    }

    uint8_t *appearance = heap_caps_malloc((size_t)count * APP_AI_PATCH_APPEARANCE_DIM,
                                            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!appearance) appearance = heap_caps_malloc((size_t)count * APP_AI_PATCH_APPEARANCE_DIM,
                                                   MALLOC_CAP_8BIT);
    if (!appearance || app_ai_patch_extract_appearance(appearance,
                            (size_t)count * APP_AI_PATCH_APPEARANCE_DIM) != ESP_OK) {
        heap_caps_free(appearance);
        heap_caps_free(features);
        out->error_code = ERR_FEATURE_FAIL;
        return;
    }

    float max_score = 0.0f;
    float ai_max_score = 0.0f;
    float appearance_max_score = 0.0f;
    uint32_t ai_blind_patches = 0;
    uint32_t fail_count = 0;
    uint32_t cols = 0, rows = 0;
    app_ai_patch_get_grid(&cols, &rows);
    uint8_t *error_matrix = calloc((size_t)rows * cols, sizeof(uint8_t));
    float score_cache[APP_AI_PATCH_MAX_COUNT] = {0};
    if (!error_matrix || count > APP_AI_PATCH_MAX_COUNT) {
        free(error_matrix);
        heap_caps_free(appearance);
        heap_caps_free(features);
        out->error_code = ERR_FEATURE_FAIL;
        return;
    }
    for (uint32_t p = 0; p < count; ++p) {
        const float *v = features + (size_t)p * APP_AI_INFER_FEATURE_DIM;
        float ai_score = 0.0f, appearance_score = 0.0f;
        float score = app_ai_patch_is_patch_visible(p) ?
            app_ai_coreset_compute_spatial_score(v, appearance +
                (size_t)p * APP_AI_PATCH_APPEARANCE_DIM, p,
                &ai_score, &appearance_score) : 0.0f;
        if (ai_score > ai_max_score) ai_max_score = ai_score;
        if (appearance_score > appearance_max_score) appearance_max_score = appearance_score;
        if (appearance_score > 1.35f && ai_score < 0.65f) ++ai_blind_patches;
        score_cache[p] = score;
        if (score > threshold) {
            error_matrix[p] = 1;
            fail_count++;
        }
        if (score > max_score) max_score = score;
    }

    cJSON *matrix_root = cJSON_CreateArray();
    if (matrix_root) {
        for (uint32_t r = 0; r < rows; ++r) {
            cJSON *row = cJSON_CreateArray();
            if (!row) break;
            for (uint32_t c = 0; c < cols; ++c) {
                cJSON_AddItemToArray(row, cJSON_CreateNumber(error_matrix[r * cols + c]));
            }
            cJSON_AddItemToArray(matrix_root, row);
        }
        char *matrix_json = cJSON_PrintUnformatted(matrix_root);
        if (matrix_json) {
            esp_err_t upload_ret = app_remote_upload_inspection(matrix_json);
            if (upload_ret != ESP_OK) {
                ESP_LOGW(TAG, "Inspection upload failed: %s", esp_err_to_name(upload_ret));
            }
        } else {
            ESP_LOGW(TAG, "Inspection upload skipped: JPEG or matrix unavailable");
        }
        cJSON_free(matrix_json);
        cJSON_Delete(matrix_root);
    }
    free(error_matrix);
    heap_caps_free(appearance);
    heap_caps_free(features);

    uint32_t elapsed_ms = (uint32_t)((esp_timer_get_time() - t_start_us) / 1000);

    out->result = (fail_count == 0) ? RESULT_PASS : RESULT_FAIL;
    out->patch_total = (uint16_t)clamp_u16(count);
    out->patch_fail_count = (uint16_t)clamp_u16(fail_count);
    out->elapsed_ms = elapsed_ms;
    out->max_dmin = max_score;

    /* Publish the exact patch scores/geometry used by this visual check so
     * the browser can draw red boxes on the exact same captured frame. */
    {
        uint32_t capture_id = ui_get_last_capture_id();
        uint32_t n = count;
        if (n > APP_AI_PATCH_MAX_COUNT) n = APP_AI_PATCH_MAX_COUNT;
        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_visual_capture_id = capture_id;
            g_visual_cols = cols;
            g_visual_rows = rows;
            /* The fixed crop is exactly cols×rows cells of PATCH_SIZE pixels.
             * Do not depend on the optional RGB crop-preview allocation: it
             * may fail under PSRAM pressure although scoring and JPEG succeed. */
            g_visual_crop_width = cols * APP_AI_PATCH_SIZE;
            g_visual_crop_height = rows * APP_AI_PATCH_SIZE;
            app_ai_patch_get_crop_origin(&g_visual_crop_x, &g_visual_crop_y);
            g_visual_patch_total = out->patch_total;
            g_visual_patch_fail_count = out->patch_fail_count;
            g_visual_elapsed_ms = out->elapsed_ms;
            g_visual_max_dmin = out->max_dmin;
            g_visual_threshold = threshold;
            g_visual_ai_max = ai_max_score;
            g_visual_appearance_max = appearance_max_score;
            g_visual_ai_blind_patches = ai_blind_patches;
            memset(g_visual_scores, 0, sizeof(g_visual_scores));
            if (n > 0) memcpy(g_visual_scores, score_cache, (size_t)n * sizeof(float));
            g_ui_visual_result = out->result;
            g_ui_visual_working = 0;
            xSemaphoreGive(g_ui_lock);
        }
    }

    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, " [ESP32] VISUAL CHECK BOARD %u RESULT: %s (%s)", 
             g_check_board,
             (out->result == RESULT_PASS) ? "PASS" : "FAIL",
             (out->result == RESULT_PASS) ? "OK" : "NG");
    ESP_LOGI(TAG, "  + Số patch bị lỗi : %u / %u patches", out->patch_fail_count, out->patch_total);
    ESP_LOGI(TAG, "  + Điểm lỗi lớn nhất: %.6f (Ngưỡng = %.6f)", out->max_dmin, threshold);
    ESP_LOGI(TAG, "  + AI max %.3f, color max %.3f, AI blind candidates %u",
             (double)ai_max_score, (double)appearance_max_score, (unsigned)ai_blind_patches);
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
    case CMD_LEARN_START: {
        /* STM32 chỉ retry LEARN_START cho tới khi nhận ACK. Vì vậy một lệnh
         * LEARN_START đến sau khi đã có sample/đang WAIT_POWER/READY_TRAIN
         * chắc chắn là lần vào LEARN mới, không phải retry của transaction cũ. */
        bool new_learn_session =
            (g_learn_state == LEARN_STATE_IDLE) ||
            (g_learn_state == LEARN_STATE_TRAINED) ||
            (g_learn_state == LEARN_STATE_WAIT_VISION && g_vision_count > 0) ||
            (g_learn_state == LEARN_STATE_WAIT_POWER) ||
            (g_learn_state == LEARN_STATE_READY_TRAIN);

        if (!new_learn_session) {
            /* Chỉ có WAIT_VISION + 0 sample là retry của LEARN_START ban đầu. */
            app_protocol_send_frame(CMD_ACK, NULL, 0);
            break;
        }

        clear_learn_storage();
        g_learn_state = LEARN_STATE_WAIT_VISION;
        g_check_state = CHECK_STATE_IDLE;
        g_check_board = 1;
        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_learn_visual_count = 0;
            g_ui_learn_power_count = 0;
            g_ui_error_code = 0;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "STM32 → LEARN_START: bắt đầu 5 mẫu ngoại quan");
            g_ui_stage = VEDC_UI_LEARN_WAIT_VISUAL;
            g_ui_board = 0;
            xSemaphoreGive(g_ui_lock);
        }
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;
    }

    case CMD_CAPTURE_SAMPLE: {
        if (f->len != 1) {
            app_protocol_send_error(ERR_UNKNOWN, "CAPTURE_SAMPLE LEN != 1");
            break;
        }

        uint8_t idx = f->data[0];
        if (idx == 0 || idx > LEARN_SAMPLES) {
            app_protocol_send_error(ERR_UNKNOWN, "CAPTURE sample index ngoài 1..5");
            break;
        }

        /* Idempotent retry: if STM32 repeats a sample whose ACK was lost,
         * resend the cached ACK without grabbing another frame. */
        if (idx <= g_vision_count && g_train_features[idx - 1U]) {
            uint8_t ack[7];
            ack[0] = idx;
            ack[1] = RESULT_PASS;
            ack[2] = g_vision_count;
            pack_u16_le(&ack[3], clamp_u16(g_train_feature_count[idx - 1U]));
            pack_u16_le(&ack[5], clamp_u16(g_capture_elapsed_ms[idx - 1U]));
            app_protocol_send_frame(CMD_CAPTURE_ACK, ack, sizeof(ack));
            break;
        }

        if (g_learn_state != LEARN_STATE_WAIT_VISION || idx != g_expected_vision) {
            app_protocol_send_error(ERR_UNKNOWN, "CAPTURE sai thứ tự");
            break;
        }

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = VEDC_UI_LEARN_CAPTURING;
            g_ui_visual_working = idx;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "LEARN: ESP32 đang chụp mẫu ngoại quan %u/5", idx);
            g_ui_error_code = 0;
            xSemaphoreGive(g_ui_lock);
        }

        float *features = NULL;
        uint32_t feature_count = 0;
        uint32_t elapsed_ms = 0;
        esp_err_t err = capture_real_feature_block(&features, &feature_count, &elapsed_ms);
        if (err != ESP_OK) {
            ui_set_error((err == ESP_ERR_INVALID_STATE || err == ESP_ERR_TIMEOUT) ? ERR_CAMERA_FAIL : ERR_FEATURE_FAIL,
                         "LEARN: chụp/trích xuất mẫu thất bại");
            app_protocol_send_error((err == ESP_ERR_INVALID_STATE || err == ESP_ERR_TIMEOUT) ? ERR_CAMERA_FAIL : ERR_FEATURE_FAIL,
                                    "Chụp ảnh thất bại");
            break;
        }

        if (idx > 1 && feature_count != g_train_feature_count[0]) {
            heap_caps_free(features);
            ui_set_error(ERR_TRAIN_FAIL, "LEARN: kích thước lưới patch thay đổi giữa các ảnh");
            app_protocol_send_error(ERR_TRAIN_FAIL, "Lưới patch LEARN không đồng nhất");
            break;
        }
        if (idx == 1 && app_ai_patch_set_registration_reference() != ESP_OK) {
            heap_caps_free(features);
            ui_set_error(ERR_TRAIN_FAIL, "LEARN: không lưu được ảnh tham chiếu PCB");
            app_protocol_send_error(ERR_TRAIN_FAIL, "Không tạo được ảnh tham chiếu");
            break;
        }

        uint8_t *appearance = heap_caps_malloc((size_t)feature_count * APP_AI_PATCH_APPEARANCE_DIM,
                                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!appearance) appearance = heap_caps_malloc((size_t)feature_count * APP_AI_PATCH_APPEARANCE_DIM,
                                                       MALLOC_CAP_8BIT);
        if (!appearance || app_ai_patch_extract_appearance(appearance,
                    (size_t)feature_count * APP_AI_PATCH_APPEARANCE_DIM) != ESP_OK) {
            heap_caps_free(appearance);
            heap_caps_free(features);
            ui_set_error(ERR_FEATURE_FAIL, "LEARN: không trích xuất được đặc trưng màu");
            app_protocol_send_error(ERR_FEATURE_FAIL, "Lỗi đặc trưng màu");
            break;
        }

        g_train_features[idx - 1U] = features;
        g_train_appearance[idx - 1U] = appearance;
        g_train_feature_count[idx - 1U] = feature_count;
        g_capture_elapsed_ms[idx - 1U] = elapsed_ms;
        g_vision_count = idx;

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_learn_visual_count = g_vision_count;
            g_ui_visual_working = 0;
            g_ui_error_code = 0;
            ++g_ui_event_seq;
            if (g_vision_count == LEARN_SAMPLES) {
                g_ui_stage = VEDC_UI_LEARN_WAIT_POWER;
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "LEARN: đủ 5/5 ảnh, chuyển sang 5 mẫu dòng điện");
            } else {
                g_ui_stage = VEDC_UI_LEARN_WAIT_VISUAL;
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "LEARN: mẫu ảnh %u/5 đã nhận", idx);
            }
            xSemaphoreGive(g_ui_lock);
        }

        /* UI đã cập nhật xong (ảnh + trạng thái) rồi mới báo ACK cho STM32 */
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
        if (idx == 0 || idx > LEARN_SAMPLES) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_SAMPLE index ngoài 1..5");
            break;
        }

        /* Idempotent retry: the same sample can be acknowledged repeatedly
         * without changing the learned statistics or sample counter. */
        if (idx <= g_power_count) {
            uint8_t ack[3] = { idx, RESULT_PASS, g_power_count };
            app_protocol_send_frame(CMD_POWER_ACK, ack, sizeof(ack));
            break;
        }

        if (g_learn_state != LEARN_STATE_WAIT_POWER || idx != g_expected_power) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_SAMPLE sai thứ tự");
            break;
        }

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = VEDC_UI_LEARN_RECEIVING_POWER;
            g_ui_power_working = idx;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "LEARN: nhận mẫu dòng %u/5 từ STM32", idx);
            g_ui_error_code = 0;
            xSemaphoreGive(g_ui_lock);
        }

        g_learn_imax[idx - 1] = imax;
        g_learn_iavg[idx - 1] = iavg;
        g_power_count = idx;
        g_expected_power++;

        uint8_t ack[3] = { idx, RESULT_PASS, g_power_count };
        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_learn_power_count = g_power_count;
            g_ui_power_working = 0;
            ++g_ui_event_seq;
            if (g_power_count == LEARN_SAMPLES) {
                g_ui_stage = VEDC_UI_LEARN_READY_TRAIN;
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "LEARN: đủ 5/5 dòng, STM32 đã sẵn sàng TRAIN");
            } else {
                g_ui_stage = VEDC_UI_LEARN_WAIT_POWER;
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "LEARN: mẫu dòng %u/5 đã nhận", idx);
            }
            xSemaphoreGive(g_ui_lock);
        }

        if (g_power_count == LEARN_SAMPLES) {
            g_learn_state = LEARN_STATE_READY_TRAIN;
            ESP_LOGI(TAG, "Thu đủ 5 ảnh + 5 dòng -> Sẵn sàng TRAIN (CMD 0x13)!");
        }

        app_protocol_send_frame(CMD_POWER_ACK, ack, sizeof(ack));
        break;
    }

    case CMD_TRAIN_START: { /* Lệnh 0x13 từ STM32 */
        /* Retry-safe: if STM32 did not see CMD_TRAIN_DONE, it resends
         * CMD_TRAIN_START. The model must not be rebuilt a second time. */
        if (g_train_done_valid && g_learn_state == LEARN_STATE_TRAINED) {
            app_protocol_send_frame(CMD_TRAIN_DONE, g_train_done_data, 26);
            break;
        }
        if (g_learn_state != LEARN_STATE_READY_TRAIN) {
            app_protocol_send_error(ERR_TRAIN_FAIL, "Chưa đủ 5 ảnh + 5 mẫu dòng");
            break;
        }

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = VEDC_UI_TRAINING;
            g_ui_board = 0;
            g_ui_error_code = 0;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "TRAIN: ESP32 đang xây Memory Bank");
            xSemaphoreGive(g_ui_lock);
        }

        int64_t t_train_start_us = esp_timer_get_time();

        app_ai_coreset_block_t blocks[LEARN_SAMPLES] = {0};
        for (uint32_t i = 0; i < LEARN_SAMPLES; ++i) {
            blocks[i].vectors = g_train_features[i];
            blocks[i].count = g_train_feature_count[i];
            blocks[i].source_image = i;
            blocks[i].appearance = g_train_appearance[i];
        }

        // 1. TÍNH TOÁN CORESET MEMORY BANK
        if (app_ai_coreset_build(blocks, LEARN_SAMPLES) != ESP_OK) {
            ui_set_error(ERR_TRAIN_FAIL, "TRAIN: Build Coreset thất bại");
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

        memcpy(g_train_done_data, data, sizeof(data));
        g_train_done_valid = true;
        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_train_bank_count = bank_count;
            g_train_image_threshold = image_threshold;
            g_train_imax_max_dev = g_imax_max_dev;
            g_train_iavg_max_dev = g_iavg_max_dev;
            g_train_elapsed_ms = train_elapsed_ms;
            g_ui_stage = VEDC_UI_TRAINED;
            g_ui_error_code = 0;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "TRAIN: hoàn tất - Memory Bank đã sẵn sàng CHECK");
            xSemaphoreGive(g_ui_lock);
        }

        app_protocol_send_frame(CMD_TRAIN_DONE, data, sizeof(data));
        break;
    }

    case CMD_PING:
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;

    case CMD_CHECK_START:
        if (!g_model_trained) {
            ui_set_error(ERR_TRAIN_FAIL, "CHECK bị từ chối: chưa TRAIN Memory Bank");
            app_protocol_send_error(ERR_TRAIN_FAIL, "Chưa TRAIN mô hình");
            break;
        }

        /* Retry-safe nhưng vẫn cho phép người vận hành rời CHECK rồi vào CHECK
         * lại mà không cần reboot ESP32. STM32 chỉ gửi CHECK_START một lần khi
         * vào mode mới; retry của transaction ban đầu luôn thấy board #1 +
         * WAIT_VISUAL. Vì vậy WAIT_POWER/WAIT_NEXT hoặc WAIT_VISUAL với board>1
         * được coi là một CHECK session mới. */
        bool new_check_session =
            (g_check_state == CHECK_STATE_IDLE) ||
            (g_check_state == CHECK_STATE_WAIT_POWER) ||
            (g_check_state == CHECK_STATE_WAIT_NEXT) ||
            (g_check_state == CHECK_STATE_WAIT_VISUAL && g_check_board != 1);

        if (new_check_session) {
            g_check_state = CHECK_STATE_WAIT_VISUAL;
            g_check_board = 1;
            g_check_board_start_us = esp_timer_get_time();
            g_visual_cache_valid = false;
            g_power_cache_valid = false;
            g_next_cache_valid = false;
            ui_reset_current_check(true);
            if (g_ui_lock) {
                xSemaphoreTake(g_ui_lock, portMAX_DELAY);
                check_history_clear_locked();   /* phiên CHECK mới: xóa tổng kết/lịch sử cũ */
                g_ui_stage = VEDC_UI_CHECK_WAIT_PCB;
                g_ui_board = 1;
                g_ui_error_code = 0;
                ++g_ui_event_seq;
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "CHECK: chờ PCB #1 từ cảm biến IR của STM32");
                xSemaphoreGive(g_ui_lock);
            }
        }
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;

    case CMD_VISUAL_CHECK: { /* Lệnh 0x21 từ STM32 */
        if (f->len != 1) {
            app_protocol_send_error(ERR_UNKNOWN, "VISUAL_CHECK LEN != 1");
            break;
        }
        uint8_t visual_board = f->data[0];
        if (visual_board == 0) {
            app_protocol_send_error(ERR_UNKNOWN, "VISUAL_CHECK board = 0");
            break;
        }

        /* Retry-safe: same PCB visual transaction => resend cached 0xA1. */
        if (g_visual_cache_valid && visual_board == g_visual_cache_board) {
            app_protocol_send_frame(CMD_VISUAL_RESULT, g_visual_cache_data, sizeof(g_visual_cache_data));
            break;
        }

        if (g_check_state != CHECK_STATE_WAIT_VISUAL || visual_board != g_check_board) {
            app_protocol_send_error(ERR_UNKNOWN, "VISUAL_CHECK không hợp lệ");
            break;
        }

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = VEDC_UI_CHECK_VISUAL;
            g_ui_board = g_check_board;
            g_ui_visual_working = g_check_board;
            g_ui_power_working = 0;
            g_ui_visual_result = 0xFF;
            g_ui_power_result = 0xFF;
            g_ui_final_result = 0xFF;
            g_ui_error_code = 0;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "CHECK PCB #%u: ESP32 đang chụp + AI ngoại quan", g_check_board);
            xSemaphoreGive(g_ui_lock);
        }

        visual_check_result_t vres;
        perform_real_visual_check(&vres);
        if (vres.result == 0xFF) {
            uint8_t err_code = vres.error_code ? vres.error_code : ERR_FEATURE_FAIL;
            ui_set_error(err_code, "CHECK: xử lý ngoại quan thất bại - xem log ESP32");
            app_protocol_send_error(err_code, "CHECK: chụp/AI ngoại quan thất bại");
            break;
        }

        /* Đóng gói 11 bytes dữ liệu trả về cho STM32 qua CMD_VISUAL_RESULT (0xA1) */
        uint8_t data[11];
        data[0] = vres.result;
        pack_u16_le(&data[1], vres.patch_fail_count);
        pack_u16_le(&data[3], vres.patch_total);
        pack_u16_le(&data[5], clamp_u16(vres.elapsed_ms));
        pack_float_le(&data[7], vres.max_dmin); /* Bổ sung max dmin */

        memcpy(g_visual_cache_data, data, sizeof(data));
        g_visual_cache_board = g_check_board;
        g_visual_cache_valid = true;

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = (vres.result == RESULT_PASS) ? VEDC_UI_CHECK_VISUAL_PASS : VEDC_UI_CHECK_VISUAL_FAIL;
            g_ui_visual_working = 0;
            g_ui_visual_result = vres.result;
            g_ui_final_result = (vres.result == RESULT_FAIL) ? RESULT_FAIL : 0xFF;
            ++g_ui_event_seq;
            if (vres.result == RESULT_PASS) {
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "PCB #%u: NGOẠI QUAN PASS (%u/%u patch lỗi) → chờ đo dòng",
                         g_check_board, (unsigned)vres.patch_fail_count, (unsigned)vres.patch_total);
            } else {
                /* Ngoại quan FAIL: không đo dòng -> tổng kết PCB ngay tại đây. */
                check_summary_commit_locked(g_check_board, vres.result, 0xFF, RESULT_FAIL, 0, 0.0f, 0.0f);
                snprintf(g_ui_last_event, sizeof(g_ui_last_event), "PCB #%u: TỔNG KẾT FAIL · ngoại quan %u/%u patch lỗi · bỏ qua đo dòng",
                         g_check_board, (unsigned)vres.patch_fail_count, (unsigned)vres.patch_total);
            }
            xSemaphoreGive(g_ui_lock);
        }

        app_protocol_send_frame(CMD_VISUAL_RESULT, data, sizeof(data));

        g_check_state = (vres.result == RESULT_PASS) ? CHECK_STATE_WAIT_POWER : CHECK_STATE_WAIT_NEXT;
        break;
    }

    case CMD_POWER_CHECK: {
        if (f->len != 9) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_CHECK LEN != 9");
            break;
        }
        if (f->data[0] == 0) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_CHECK board = 0");
            break;
        }

        /* Retry-safe: same board + same 9-byte request => resend cached 0xA2. */
        if (g_power_cache_valid && f->data[0] == g_power_cache_board &&
            memcmp(f->data, g_power_cache_request, sizeof(g_power_cache_request)) == 0) {
            app_protocol_send_frame(CMD_POWER_RESULT, g_power_cache_data, sizeof(g_power_cache_data));
            break;
        }

        if (g_check_state != CHECK_STATE_WAIT_POWER || f->data[0] != g_check_board) {
            app_protocol_send_error(ERR_UNKNOWN, "POWER_CHECK không hợp lệ");
            break;
        }

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = VEDC_UI_CHECK_POWER;
            g_ui_board = g_check_board;
            g_ui_power_working = g_check_board;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "CHECK PCB #%u: nhận Imax/Iavg từ STM32", g_check_board);
            xSemaphoreGive(g_ui_lock);
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
        memcpy(g_power_cache_data, data, sizeof(data));
        memcpy(g_power_cache_request, f->data, sizeof(g_power_cache_request));
        g_power_cache_board = g_check_board;
        g_power_cache_valid = true;

        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_power_working = 0;
            g_ui_power_result = pres.result;
            g_ui_final_result = pres.result;
            g_ui_imax_measured = pres.imax_measured;
            g_ui_imax_threshold = pres.imax_threshold;
            g_ui_iavg_measured = pres.iavg_measured;
            g_ui_iavg_threshold = pres.iavg_threshold;
            g_ui_power_elapsed_ms = pres.elapsed_ms;
            g_ui_stage = (pres.result == RESULT_PASS) ? VEDC_UI_CHECK_POWER_PASS : VEDC_UI_CHECK_POWER_FAIL;
            ++g_ui_event_seq;
            check_summary_commit_locked(g_check_board, g_ui_visual_result, pres.result, pres.result,
                                        pres.elapsed_ms, pres.imax_measured, pres.iavg_measured);
            snprintf(g_ui_last_event, sizeof(g_ui_last_event),
                     "PCB #%u: TỔNG KẾT %s · patch lỗi %u/%u · dòng %s",
                     g_check_board, (pres.result == RESULT_PASS) ? "PASS" : "FAIL",
                     (unsigned)g_visual_patch_fail_count, (unsigned)g_visual_patch_total,
                     (pres.result == RESULT_PASS) ? "PASS" : "FAIL");
            xSemaphoreGive(g_ui_lock);
        }

        app_protocol_send_frame(CMD_POWER_RESULT, data, sizeof(data));

        g_check_state = CHECK_STATE_WAIT_NEXT;
        break;
    }

    case CMD_CHECK_NEXT: {
        if (f->len != 1) {
            app_protocol_send_error(ERR_UNKNOWN, "CHECK_NEXT LEN != 1");
            break;
        }
        uint8_t next_board = f->data[0];
        if (next_board == 0) {
            app_protocol_send_error(ERR_UNKNOWN, "CHECK_NEXT board = 0");
            break;
        }

        /* STM32 may resend CHECK_NEXT until ACK arrives. A duplicate for the
         * board already advanced must only resend ACK, never increment twice. */
        if (g_next_cache_valid && next_board == g_next_cache_board) {
            app_protocol_send_frame(CMD_ACK, NULL, 0);
            break;
        }

        if (g_check_state != CHECK_STATE_WAIT_NEXT || next_board != g_check_board) {
            app_protocol_send_error(ERR_UNKNOWN, "CHECK_NEXT không hợp lệ");
            break;
        }

        g_next_cache_valid = true;
        g_next_cache_board = g_check_board;
        g_check_board++;
        g_check_state = CHECK_STATE_WAIT_VISUAL;
        g_check_board_start_us = esp_timer_get_time();
        g_visual_cache_valid = false;
        g_power_cache_valid = false;
        ui_reset_current_check(false);
        if (g_ui_lock) {
            xSemaphoreTake(g_ui_lock, portMAX_DELAY);
            g_ui_stage = VEDC_UI_CHECK_WAIT_PCB;
            g_ui_board = g_check_board;
            g_ui_visual_working = 0;
            g_ui_power_working = 0;
            ++g_ui_event_seq;
            snprintf(g_ui_last_event, sizeof(g_ui_last_event), "CHECK: đã chuyển sang PCB #%u - chờ IR", g_check_board);
            xSemaphoreGive(g_ui_lock);
        }
        app_protocol_send_frame(CMD_ACK, NULL, 0);
        break;
    }

    default:
        app_protocol_send_error(ERR_UNKNOWN, "CMD không xác định");
        break;
    }
}

void vedc_controller_set_runtime_ready(bool ready)
{
    g_runtime_ready = ready;
    ESP_LOGI(TAG, "Runtime camera/AI ready = %s", ready ? "YES" : "NO");
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
    /* UI snapshot dùng chung giữa command task và HTTP task. */
    if (!g_ui_lock) {
        g_ui_lock = xSemaphoreCreateMutex();
        if (!g_ui_lock) {
            ESP_LOGE(TAG, "Không tạo được mutex UI");
            return ESP_ERR_NO_MEM;
        }
    }

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
