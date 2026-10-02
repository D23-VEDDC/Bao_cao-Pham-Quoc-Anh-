/*
 * SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cJSON.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "app_web.h"
#include "app_uac_manager.h"
#include "app_uvc_manager.h"
#include "app_ai_patch.h"
#include "app_ai_infer.h"
#include "app_ai_session.h"
#include "app_ai_coreset.h"

#define APP_WEB_JSON_BUF_SIZE        8192
#define APP_WEB_AI_PATCH_JSON_BUF_SIZE 1024
// Big enough for up to APP_AI_PATCH_MAX_COUNT feature vectors of
// APP_AI_INFER_FEATURE_DIM floats each, plus JSON punctuation/keys. Computed
// from those two constants (32 bytes/number covers cJSON's worst-case
// scientific-notation float formatting plus separators) so this buffer
// automatically stays correct if the patch grid ceiling in app_ai_patch.h
// ever changes again - it no longer needs to be hand-updated in sync.
// Allocated on the heap (spills into PSRAM automatically, not the httpd
// worker's stack) since it's much larger than the other JSON buffers on
// this page.
#define APP_WEB_AI_INFER_JSON_BUF_SIZE \
    ((size_t)APP_AI_PATCH_MAX_COUNT * APP_AI_INFER_FEATURE_DIM * 32u + 4096u)
#define APP_WEB_AI_CORESET_JSON_BUF_SIZE (64 * 1024)
#define APP_WEB_WS_RX_BUF_SIZE       4096
#define APP_WEB_AUDIO_HEADER_SIZE    4
#define APP_WEB_AUDIO_MIC_TYPE       0x01
#define APP_WEB_AUDIO_SPK_TYPE       0x02
#define APP_WEB_CAPTIVE_PORTAL_URL   "http://192.168.4.1/"
#define APP_WEB_CAMERA_HEADER_SIZE   16
#define APP_WEB_CAMERA_FORMAT_MJPEG  1
#define APP_WEB_CAMERA_FRAME_WAIT_MS 500
#define APP_WEB_CAMERA_WS_RX_BUF_SIZE 256

extern const uint8_t web_index_html_start[] asm("_binary_index_html_start");
extern const uint8_t web_index_html_end[] asm("_binary_index_html_end");
extern const uint8_t web_style_css_start[] asm("_binary_style_css_start");
extern const uint8_t web_style_css_end[] asm("_binary_style_css_end");
extern const uint8_t web_app_js_start[] asm("_binary_app_js_start");
extern const uint8_t web_app_js_end[] asm("_binary_app_js_end");
extern const uint8_t web_mic_player_worklet_js_start[] asm("_binary_mic_player_worklet_js_start");
extern const uint8_t web_mic_player_worklet_js_end[] asm("_binary_mic_player_worklet_js_end");

typedef enum {
    APP_AUDIO_WS_WORK_STATE,
    APP_AUDIO_WS_WORK_TEXT,
    APP_AUDIO_WS_WORK_BINARY,
} app_audio_ws_work_type_t;

typedef struct {
    app_audio_ws_work_type_t type;
    bool reset_spk_on_fail;
    size_t len;
    uint8_t data[];
} app_audio_ws_work_msg_t;

static const char *TAG = "app_web";
static httpd_handle_t s_server;
static portMUX_TYPE s_ws_mux = portMUX_INITIALIZER_UNLOCKED;
static int s_audio_ws_fd = -1;
static int s_camera_ws_fd = -1;

static bool cjson_get_bool(const cJSON *root, const char *key, bool *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsBool(item)) {
        return false;
    }
    *value = cJSON_IsTrue(item);
    return true;
}

static bool cjson_get_u32(const cJSON *root, const char *key, uint32_t *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsNumber(item) || item->valuedouble < 0 || item->valuedouble > UINT32_MAX) {
        return false;
    }
    *value = (uint32_t)item->valuedouble;
    return true;
}

static bool cjson_get_float(const cJSON *root, const char *key, float *value)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (!cJSON_IsNumber(item)) {
        return false;
    }
    *value = (float)item->valuedouble;
    return true;
}

static const char *cjson_get_type(const cJSON *root)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, "type");
    return cJSON_IsString(item) ? item->valuestring : NULL;
}

static const char *cjson_get_string(const cJSON *root, const char *key)
{
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    return cJSON_IsString(item) ? item->valuestring : NULL;
}

static esp_err_t app_web_get_state_json(char *buf, size_t len)
{
    if (!buf || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    char *audio_json = calloc(1, len);
    if (!audio_json) {
        ESP_LOGE(TAG, "Allocate audio state JSON failed");
        return ESP_ERR_NO_MEM;
    }
    esp_err_t ret = app_uac_get_state_json(audio_json, len);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Build audio state JSON failed: %s", esp_err_to_name(ret));
        free(audio_json);
        return ret;
    }

    cJSON *root = cJSON_Parse(audio_json);
    free(audio_json);
    if (!root) {
        ESP_LOGE(TAG, "Parse audio state JSON failed");
        return ESP_FAIL;
    }

    ret = app_uvc_get_state_json(root);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Build camera state JSON failed: %s", esp_err_to_name(ret));
        cJSON_Delete(root);
        return ret;
    }
    if (!cJSON_PrintPreallocated(root, buf, len, false)) {
        ESP_LOGE(TAG, "Print combined state JSON failed");
        cJSON_Delete(root);
        return ESP_ERR_NO_MEM;
    }
    cJSON_Delete(root);
    return ESP_OK;
}

static const char *ws_info_name(httpd_ws_client_info_t info)
{
    switch (info) {
    case HTTPD_WS_CLIENT_INVALID:
        return "invalid";
    case HTTPD_WS_CLIENT_HTTP:
        return "http";
    case HTTPD_WS_CLIENT_WEBSOCKET:
        return "websocket";
    default:
        return "unknown";
    }
}

static int ws_get_fd(int *slot)
{
    portENTER_CRITICAL(&s_ws_mux);
    int fd = *slot;
    portEXIT_CRITICAL(&s_ws_mux);
    return fd;
}

static bool ws_is_current(int *slot, int fd)
{
    bool current = false;
    portENTER_CRITICAL(&s_ws_mux);
    current = fd >= 0 && *slot == fd;
    portEXIT_CRITICAL(&s_ws_mux);
    return current;
}

static bool ws_clear_if_current(int *slot, int fd)
{
    bool cleared = false;
    portENTER_CRITICAL(&s_ws_mux);
    if (fd >= 0 && *slot == fd) {
        *slot = -1;
        cleared = true;
    }
    portEXIT_CRITICAL(&s_ws_mux);
    return cleared;
}

static int ws_set_current(int *slot, int fd)
{
    if (fd < 0) {
        return -1;
    }

    int old_fd = -1;
    portENTER_CRITICAL(&s_ws_mux);
    old_fd = *slot;
    *slot = fd;
    portEXIT_CRITICAL(&s_ws_mux);
    return old_fd;
}

static int audio_ws_get_fd(void)
{
    return ws_get_fd(&s_audio_ws_fd);
}

static bool audio_ws_is_current(int fd)
{
    return ws_is_current(&s_audio_ws_fd, fd);
}

static void audio_ws_open(int fd)
{
    int old_fd = ws_set_current(&s_audio_ws_fd, fd);
    if (old_fd >= 0 && old_fd != fd) {
        ESP_LOGI(TAG, "Replacing audio WebSocket fd=%d with fd=%d", old_fd, fd);
        app_uac_reset_spk_refill_budget();
        if (s_server) {
            httpd_sess_trigger_close(s_server, old_fd);
        }
    }
}

static bool audio_ws_clear_if_current(int fd)
{
    bool cleared = ws_clear_if_current(&s_audio_ws_fd, fd);
    if (cleared) {
        ESP_LOGI(TAG, "Audio WebSocket session cleared fd=%d", fd);
        app_uac_reset_spk_refill_budget();
    }
    return cleared;
}

static bool camera_ws_is_current(int fd)
{
    return ws_is_current(&s_camera_ws_fd, fd);
}

static void camera_ws_open(int fd)
{
    int old_fd = ws_set_current(&s_camera_ws_fd, fd);
    if (old_fd >= 0 && old_fd != fd) {
        ESP_LOGI(TAG, "Replacing camera WebSocket fd=%d with fd=%d", old_fd, fd);
        if (s_server) {
            httpd_sess_trigger_close(s_server, old_fd);
        }
    }
}

static bool camera_ws_clear_if_current(int fd)
{
    bool cleared = ws_clear_if_current(&s_camera_ws_fd, fd);
    if (cleared) {
        ESP_LOGI(TAG, "Camera WebSocket session cleared fd=%d", fd);
    }
    return cleared;
}

static esp_err_t audio_ws_send_frame_to_current(httpd_ws_type_t type, const uint8_t *data, size_t len)
{
    int fd = audio_ws_get_fd();
    if (!s_server) {
        ESP_LOGW(TAG, "Send audio WS frame skipped: server is not started type=%d len=%u", type, (unsigned)len);
        return ESP_ERR_INVALID_STATE;
    }
    if (fd < 0) {
        ESP_LOGD(TAG, "Send audio WS frame skipped: no active session type=%d len=%u", type, (unsigned)len);
        return ESP_ERR_NOT_FOUND;
    }
    httpd_ws_client_info_t info = httpd_ws_get_fd_info(s_server, fd);
    if (info != HTTPD_WS_CLIENT_WEBSOCKET) {
        ESP_LOGW(TAG, "Send audio WS frame skipped: fd=%d info=%s type=%d len=%u", fd, ws_info_name(info), type, (unsigned)len);
        audio_ws_clear_if_current(fd);
        return ESP_ERR_INVALID_STATE;
    }

    httpd_ws_frame_t frame = {
        .type = type,
        .payload = (uint8_t *)data,
        .len = len,
    };
    esp_err_t ret = httpd_ws_send_frame_async(s_server, fd, &frame);
    if (ret == ESP_OK) {
        if (type == HTTPD_WS_TYPE_TEXT && len > 0 && data[0] == '{') {
            ESP_LOGD(TAG, "Sent audio WS text fd=%d len=%u", fd, (unsigned)len);
        }
        return ESP_OK;
    }

    ESP_LOGW(TAG, "Send audio WS frame failed fd=%d: %s", fd, esp_err_to_name(ret));
    audio_ws_clear_if_current(fd);
    if (s_server) {
        httpd_sess_trigger_close(s_server, fd);
    }
    return ret;
}

static void audio_ws_work_handler(void *arg)
{
    app_audio_ws_work_msg_t *msg = (app_audio_ws_work_msg_t *)arg;
    if (!msg) {
        return;
    }

    esp_err_t ret = ESP_OK;
    if (msg->type == APP_AUDIO_WS_WORK_STATE) {
        char *json = calloc(1, APP_WEB_JSON_BUF_SIZE);
        if (!json) {
            ESP_LOGE(TAG, "Allocate state JSON failed");
            free(msg);
            return;
        }
        ret = app_web_get_state_json(json, APP_WEB_JSON_BUF_SIZE);
        if (ret == ESP_OK) {
            ret = audio_ws_send_frame_to_current(HTTPD_WS_TYPE_TEXT, (const uint8_t *)json, strlen(json));
        } else {
            ESP_LOGW(TAG, "Build state JSON for WS failed: %s", esp_err_to_name(ret));
        }
        free(json);
    } else if (msg->type == APP_AUDIO_WS_WORK_TEXT) {
        ret = audio_ws_send_frame_to_current(HTTPD_WS_TYPE_TEXT, msg->data, msg->len);
    } else if (msg->type == APP_AUDIO_WS_WORK_BINARY) {
        ret = audio_ws_send_frame_to_current(HTTPD_WS_TYPE_BINARY, msg->data, msg->len);
    }

    if (ret != ESP_OK) {
        if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGD(TAG, "Audio WS work skipped type=%d because no client is connected", msg->type);
        } else {
            ESP_LOGW(TAG, "Audio WS work send failed type=%d current_fd=%d: %s", msg->type, audio_ws_get_fd(), esp_err_to_name(ret));
        }
        if (msg->reset_spk_on_fail) {
            app_uac_reset_spk_refill_budget();
        }
    }
    free(msg);
}

static esp_err_t queue_audio_ws_work(app_audio_ws_work_type_t type, const uint8_t *data, size_t len, bool reset_spk_on_fail)
{
    if (!s_server) {
        return ESP_ERR_INVALID_STATE;
    }
    if (type != APP_AUDIO_WS_WORK_STATE && (!data || len == 0)) {
        return ESP_ERR_INVALID_ARG;
    }

    app_audio_ws_work_msg_t *msg = calloc(1, sizeof(app_audio_ws_work_msg_t) + len);
    if (!msg) {
        ESP_LOGE(TAG, "Allocate WS work failed");
        return ESP_ERR_NO_MEM;
    }
    msg->type = type;
    msg->reset_spk_on_fail = reset_spk_on_fail;
    msg->len = len;
    if (len > 0) {
        memcpy(msg->data, data, len);
    }

    esp_err_t ret = httpd_queue_work(s_server, audio_ws_work_handler, msg);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Queue WS work failed type=%d: %s", type, esp_err_to_name(ret));
        free(msg);
    }
    return ret;
}

static esp_err_t queue_audio_ws_mic_pcm(const uint8_t *data, size_t len)
{
    if (!s_server) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!data || len == 0 || len > UINT16_MAX) {
        return ESP_ERR_INVALID_ARG;
    }

    app_audio_ws_work_msg_t *msg = calloc(1, sizeof(app_audio_ws_work_msg_t) + APP_WEB_AUDIO_HEADER_SIZE + len);
    if (!msg) {
        ESP_LOGE(TAG, "Allocate MIC WS work failed len=%u", (unsigned)len);
        return ESP_ERR_NO_MEM;
    }
    msg->type = APP_AUDIO_WS_WORK_BINARY;
    msg->len = APP_WEB_AUDIO_HEADER_SIZE + len;
    msg->data[0] = APP_WEB_AUDIO_MIC_TYPE;
    msg->data[1] = 0;
    msg->data[2] = len & 0xff;
    msg->data[3] = (len >> 8) & 0xff;
    memcpy(msg->data + APP_WEB_AUDIO_HEADER_SIZE, data, len);

    esp_err_t ret = httpd_queue_work(s_server, audio_ws_work_handler, msg);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Queue MIC WS work failed len=%u: %s", (unsigned)len, esp_err_to_name(ret));
        free(msg);
    }
    return ret;
}

static esp_err_t send_text_to_req_ws(httpd_req_t *req, const char *text)
{
    if (!req || !text) {
        return ESP_ERR_INVALID_ARG;
    }
    httpd_ws_frame_t frame = {
        .type = HTTPD_WS_TYPE_TEXT,
        .payload = (uint8_t *)text,
        .len = strlen(text),
    };
    esp_err_t ret = httpd_ws_send_frame(req, &frame);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Send WS text response failed: %s", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t send_embedded_file(httpd_req_t *req, const char *type, const uint8_t *start, const uint8_t *end)
{
    httpd_resp_set_type(req, type);
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, (const char *)start, end - start);
}

static esp_err_t index_handler(httpd_req_t *req)
{
    return send_embedded_file(req, "text/html", web_index_html_start, web_index_html_end);
}

static esp_err_t style_handler(httpd_req_t *req)
{
    return send_embedded_file(req, "text/css", web_style_css_start, web_style_css_end);
}

static esp_err_t app_js_handler(httpd_req_t *req)
{
    return send_embedded_file(req, "application/javascript", web_app_js_start, web_app_js_end);
}

static esp_err_t mic_player_worklet_js_handler(httpd_req_t *req)
{
    return send_embedded_file(req, "application/javascript", web_mic_player_worklet_js_start, web_mic_player_worklet_js_end);
}

static esp_err_t captive_probe_handler(httpd_req_t *req)
{
    // Redirect captive portal probes to the local web console.
    httpd_resp_set_status(req, "302 Found");
    httpd_resp_set_hdr(req, "Location", APP_WEB_CAPTIVE_PORTAL_URL);
    httpd_resp_set_hdr(req, "Connection", "close");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t captive_404_handler(httpd_req_t *req, httpd_err_code_t error)
{
    (void)error;
    return captive_probe_handler(req);
}

static esp_err_t devices_handler(httpd_req_t *req)
{
    char *json = calloc(1, APP_WEB_JSON_BUF_SIZE);
    if (!json) {
        ESP_LOGE(TAG, "Allocate devices JSON failed");
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no memory");
        return ESP_ERR_NO_MEM;
    }
    esp_err_t ret = app_web_get_state_json(json, APP_WEB_JSON_BUF_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Build devices JSON failed: %s", esp_err_to_name(ret));
        free(json);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "state json failed");
        return ret;
    }
    httpd_resp_set_type(req, "application/json");
    ret = httpd_resp_sendstr(req, json);
    free(json);
    return ret;
}

static esp_err_t ai_patch_capture_handler(httpd_req_t *req)
{
    // Multi-photo capture is deliberately controlled only by physical P4/P3.
    // Keep the URI for compatibility with old clients, but never launch a
    // competing capture/inference worker from an HTTP request.
    (void)req;
    return ESP_ERR_INVALID_STATE;
}

static esp_err_t ai_patch_status_handler(httpd_req_t *req)
{
    char json[APP_WEB_AI_PATCH_JSON_BUF_SIZE];
    esp_err_t json_ret = app_ai_patch_get_status_json(json, sizeof(json));
    if (json_ret != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ai patch status json failed");
        return json_ret;
    }
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

// Serves the feature vectors produced by the LAST app_ai_infer_run_all()
// call (triggered automatically right after a successful patch capture -
// see main.c's button task and ai_patch_capture_handler() above) - GET
// /api/ai_infer/features. Does NOT run inference itself, same read-back
// pattern as /api/ai_patch/status: it just reports whatever the device
// already computed, so refreshing the page never re-runs the model.
static esp_err_t ai_session_status_handler(httpd_req_t *req)
{
    char json[8192];
    esp_err_t ret = app_ai_session_get_status_json(json, sizeof(json));
    if (ret != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ai session status failed");
        return ret;
    }
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, json);
}

static esp_err_t ai_session_features_handler(httpd_req_t *req)
{
    char query[64];
    char image_value[16] = {0};
    uint32_t image_index = 0;
    size_t qlen = httpd_req_get_url_query_len(req);
    if (qlen == 0 || qlen >= sizeof(query)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "missing image query");
        return ESP_ERR_INVALID_ARG;
    }
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "image", image_value, sizeof(image_value)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "missing image query");
        return ESP_ERR_INVALID_ARG;
    }
    char *endptr = NULL;
    unsigned long parsed = strtoul(image_value, &endptr, 10);
    if (endptr == image_value || *endptr != '\0' || parsed > UINT32_MAX) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "invalid image index");
        return ESP_ERR_INVALID_ARG;
    }
    image_index = (uint32_t)parsed;
    // One image's patches never exceed APP_AI_PATCH_MAX_COUNT; double the
    // base buffer for margin.
    char *json = malloc(APP_WEB_AI_INFER_JSON_BUF_SIZE * 2);
    if (!json) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "out of memory");
        return ESP_ERR_NO_MEM;
    }
    esp_err_t ret = app_ai_session_get_image_features_json(image_index, json, APP_WEB_AI_INFER_JSON_BUF_SIZE * 2);
    if (ret != ESP_OK) {
        free(json);
        httpd_resp_send_err(req, ret == ESP_ERR_NOT_FOUND ? HTTPD_404_NOT_FOUND : HTTPD_500_INTERNAL_SERVER_ERROR,
                            "image features unavailable");
        return ret;
    }
    httpd_resp_set_type(req, "application/json");
    ret = httpd_resp_sendstr(req, json);
    free(json);
    return ret;
}

static esp_err_t ai_coreset_status_handler(httpd_req_t *req)
{
    char *json = malloc(APP_WEB_AI_CORESET_JSON_BUF_SIZE);
    if (!json) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no memory");
    esp_err_t ret = app_ai_coreset_get_status_json(json, APP_WEB_AI_CORESET_JSON_BUF_SIZE);
    if (ret == ESP_OK) {
        httpd_resp_set_type(req, "application/json");
        ret = httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
    } else {
        ret = httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "coreset status unavailable");
    }
    free(json);
    return ret;
}

static esp_err_t ai_coreset_bank_handler(httpd_req_t *req)
{
    char *json = malloc(APP_WEB_AI_CORESET_JSON_BUF_SIZE);
    if (!json) return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no memory");
    esp_err_t ret = app_ai_coreset_get_bank_json(json, APP_WEB_AI_CORESET_JSON_BUF_SIZE);
    if (ret == ESP_OK) {
        httpd_resp_set_type(req, "application/json");
        ret = httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
    } else {
        ret = httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "memory bank unavailable");
    }
    free(json);
    return ret;
}

static esp_err_t ai_infer_features_handler(httpd_req_t *req)
{
    char *json = calloc(1, APP_WEB_AI_INFER_JSON_BUF_SIZE);
    if (!json) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no memory");
        return ESP_ERR_NO_MEM;
    }
    esp_err_t json_ret = app_ai_infer_get_status_json(json, APP_WEB_AI_INFER_JSON_BUF_SIZE, true);
    if (json_ret != ESP_OK) {
        free(json);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ai infer status json failed");
        return json_ret;
    }
    httpd_resp_set_type(req, "application/json");
    esp_err_t ret = httpd_resp_sendstr(req, json);
    free(json);
    return ret;
}

// Serves one captured 40x40 RGB888 patch as an uncompressed BMP so it can be
// dropped straight into an <img> tag for visual sanity-checking - GET
// /api/ai_patch/patch?index=N (N is 0-based, row-major over the cols x rows
// grid reported by /api/ai_patch/status).
static esp_err_t ai_patch_patch_handler(httpd_req_t *req)
{
    int index = 0;
    char query[32];
    if (httpd_req_get_url_query_len(req) > 0 &&
        httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[8] = {0};
        if (httpd_query_key_value(query, "index", val, sizeof(val)) == ESP_OK) {
            index = atoi(val);
        }
    }
    if (index < 0) {
        index = 0;
    }

    size_t data_len = 0;
    const uint8_t *data = app_ai_patch_get_data((uint32_t)index, &data_len);
    if (!data) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "patch not available");
        return ESP_FAIL;
    }

    const int w = APP_AI_PATCH_SIZE;
    const int h = APP_AI_PATCH_SIZE;
    const int src_row_bytes = w * 3;
    // 40 * 3 = 120 bytes/row, already a multiple of 4 so BMP needs no row padding.
    const uint32_t pixel_bytes = (uint32_t)(src_row_bytes * h);
    const uint32_t file_size = 54 + pixel_bytes;

    uint8_t header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    header[2] = (uint8_t)(file_size);
    header[3] = (uint8_t)(file_size >> 8);
    header[4] = (uint8_t)(file_size >> 16);
    header[5] = (uint8_t)(file_size >> 24);
    header[10] = 54;           // pixel data offset
    header[14] = 40;           // DIB (BITMAPINFOHEADER) size
    header[18] = (uint8_t)(w);
    header[19] = (uint8_t)(w >> 8);
    header[22] = (uint8_t)(h); // positive height => bottom-up row order
    header[23] = (uint8_t)(h >> 8);
    header[26] = 1;            // color planes
    header[28] = 24;           // bits per pixel
    header[34] = (uint8_t)(pixel_bytes);
    header[35] = (uint8_t)(pixel_bytes >> 8);
    header[36] = (uint8_t)(pixel_bytes >> 16);
    header[37] = (uint8_t)(pixel_bytes >> 24);

    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t ret = httpd_resp_send_chunk(req, (const char *)header, sizeof(header));

    uint8_t row[APP_AI_PATCH_SIZE * 3];
    for (int y = h - 1; y >= 0 && ret == ESP_OK; y--) {
        const uint8_t *src_row = data + (size_t)y * src_row_bytes;
        for (int x = 0; x < w; x++) {
            // Patch data is RGB888; BMP pixel rows are stored as BGR.
            row[x * 3 + 0] = src_row[x * 3 + 2];
            row[x * 3 + 1] = src_row[x * 3 + 1];
            row[x * 3 + 2] = src_row[x * 3 + 0];
        }
        ret = httpd_resp_send_chunk(req, (const char *)row, sizeof(row));
    }
    if (ret == ESP_OK) {
        ret = httpd_resp_send_chunk(req, NULL, 0);
    }
    return ret;
}

// Serves the exact crop-rectangle image that the last successful AI-patch
// capture cut its 40x40 patches from, as an uncompressed BMP - GET
// /api/ai_patch/photo. This is the SAME decoded pixels app_ai_patch.c cut
// into patches (see app_ai_patch_get_crop_image()), so what shows up in the
// "Anh da chup" panel is guaranteed to be the actual captured photo, not a
// separately/live-grabbed frame: it fixes the earlier mismatch where the
// browser's own canvas snapshot of the live preview and the ESP32's patch
// cut could come from two different camera frames if anything moved
// in-between.
static esp_err_t ai_patch_photo_handler(httpd_req_t *req)
{
    uint32_t w = 0, h = 0;
    const uint8_t *data = app_ai_patch_get_crop_image(&w, &h, NULL);
    if (!data || w == 0 || h == 0) {
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "captured photo not available");
        return ESP_FAIL;
    }

    const size_t src_row_bytes = (size_t)w * 3;
    // BMP pixel rows must be padded to a 4-byte boundary (unlike the fixed
    // 40x40 patches served above, this width is arbitrary).
    const size_t dst_row_bytes = (src_row_bytes + 3) & ~((size_t)3);
    const uint32_t pixel_bytes = (uint32_t)(dst_row_bytes * h);
    const uint32_t file_size = 54 + pixel_bytes;

    uint8_t header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    header[2] = (uint8_t)(file_size);
    header[3] = (uint8_t)(file_size >> 8);
    header[4] = (uint8_t)(file_size >> 16);
    header[5] = (uint8_t)(file_size >> 24);
    header[10] = 54;           // pixel data offset
    header[14] = 40;           // DIB (BITMAPINFOHEADER) size
    header[18] = (uint8_t)(w);
    header[19] = (uint8_t)(w >> 8);
    header[20] = (uint8_t)(w >> 16);
    header[21] = (uint8_t)(w >> 24);
    header[22] = (uint8_t)(h); // positive height => bottom-up row order
    header[23] = (uint8_t)(h >> 8);
    header[24] = (uint8_t)(h >> 16);
    header[25] = (uint8_t)(h >> 24);
    header[26] = 1;            // color planes
    header[28] = 24;           // bits per pixel
    header[34] = (uint8_t)(pixel_bytes);
    header[35] = (uint8_t)(pixel_bytes >> 8);
    header[36] = (uint8_t)(pixel_bytes >> 16);
    header[37] = (uint8_t)(pixel_bytes >> 24);

    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t ret = httpd_resp_send_chunk(req, (const char *)header, sizeof(header));

    uint8_t *row = malloc(dst_row_bytes);
    if (!row) {
        httpd_resp_send_chunk(req, NULL, 0);
        return ESP_ERR_NO_MEM;
    }
    // Any alignment-padding bytes at the end of the row stay zero; only the
    // first src_row_bytes get overwritten below on every iteration.
    memset(row, 0, dst_row_bytes);

    for (int y = (int)h - 1; y >= 0 && ret == ESP_OK; y--) {
        const uint8_t *src_row = data + (size_t)y * src_row_bytes;
        for (uint32_t x = 0; x < w; x++) {
            // Crop image is RGB888; BMP pixel rows are stored as BGR.
            row[x * 3 + 0] = src_row[x * 3 + 2];
            row[x * 3 + 1] = src_row[x * 3 + 1];
            row[x * 3 + 2] = src_row[x * 3 + 0];
        }
        ret = httpd_resp_send_chunk(req, (const char *)row, dst_row_bytes);
    }
    free(row);
    if (ret == ESP_OK) {
        ret = httpd_resp_send_chunk(req, NULL, 0);
    }
    return ret;
}

static esp_err_t send_state_to_req_ws(httpd_req_t *req)
{
    char *json = calloc(1, APP_WEB_JSON_BUF_SIZE);
    if (!json) {
        ESP_LOGE(TAG, "Allocate state JSON failed");
        return ESP_ERR_NO_MEM;
    }
    esp_err_t ret = app_web_get_state_json(json, APP_WEB_JSON_BUF_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Build state JSON failed: %s", esp_err_to_name(ret));
        free(json);
        return ret;
    }
    ret = send_text_to_req_ws(req, json);
    free(json);
    return ret;
}

static esp_err_t send_spk_refill_budget_to_req_ws(httpd_req_t *req, uint32_t bytes)
{
    if (bytes == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    char json[64];
    snprintf(json, sizeof(json), "{\"type\":\"spk_refill_budget\",\"bytes\":%"PRIu32"}", bytes);
    return send_text_to_req_ws(req, json);
}

static esp_err_t handle_ws_text(httpd_req_t *req, const char *msg)
{
    bool b = false;
    uint32_t v = 0;
    esp_err_t ret = ESP_OK;
    cJSON *root = cJSON_Parse(msg);
    if (!root) {
        ESP_LOGW(TAG, "Parse WS JSON failed");
        app_web_send_error("invalid_json", "Invalid JSON command");
        return ESP_ERR_INVALID_ARG;
    }
    const char *type = cjson_get_type(root);
    if (!type) {
        ESP_LOGW(TAG, "WS command missing type");
        ret = ESP_ERR_INVALID_ARG;
    } else if (strcmp(type, "get_state") == 0) {
        ret = send_state_to_req_ws(req);
    } else if (strcmp(type, "set_mic_enabled") == 0 && cjson_get_bool(root, "enabled", &b)) {
        ret = app_uac_set_enabled(APP_UAC_STREAM_MIC, b);
    } else if (strcmp(type, "set_spk_enabled") == 0 && cjson_get_bool(root, "enabled", &b)) {
        ret = app_uac_set_enabled(APP_UAC_STREAM_SPK, b);
    } else if (strcmp(type, "set_mic_mute") == 0 && cjson_get_bool(root, "mute", &b)) {
        ret = app_uac_set_mute(APP_UAC_STREAM_MIC, b);
    } else if (strcmp(type, "set_spk_mute") == 0 && cjson_get_bool(root, "mute", &b)) {
        ret = app_uac_set_mute(APP_UAC_STREAM_SPK, b);
    } else if (strcmp(type, "set_mic_volume") == 0 && cjson_get_u32(root, "volume", &v)) {
        ret = app_uac_set_volume(APP_UAC_STREAM_MIC, (uint8_t)v);
    } else if (strcmp(type, "set_spk_volume") == 0 && cjson_get_u32(root, "volume", &v)) {
        ret = app_uac_set_volume(APP_UAC_STREAM_SPK, (uint8_t)v);
    } else if (strcmp(type, "set_mic_format") == 0) {
        uint32_t alt = 0;
        uint32_t freq = 0;
        bool valid = cjson_get_u32(root, "alt", &alt) && cjson_get_u32(root, "sample_freq", &freq);
        ret = valid ? app_uac_set_format(APP_UAC_STREAM_MIC, (uint8_t)alt, freq) : ESP_ERR_INVALID_ARG;
    } else if (strcmp(type, "set_spk_format") == 0) {
        uint32_t alt = 0;
        uint32_t freq = 0;
        bool valid = cjson_get_u32(root, "alt", &alt) && cjson_get_u32(root, "sample_freq", &freq);
        ret = valid ? app_uac_set_format(APP_UAC_STREAM_SPK, (uint8_t)alt, freq) : ESP_ERR_INVALID_ARG;
    } else if (strcmp(type, "request_spk_refill_budget") == 0) {
        uint32_t bytes = 0;
        ret = app_uac_request_spk_refill_budget(&bytes);
        if (ret == ESP_OK && bytes > 0) {
            ret = send_spk_refill_budget_to_req_ws(req, bytes);
            if (ret != ESP_OK) {
                app_uac_reset_spk_refill_budget();
            }
        }
    } else if (strcmp(type, "stop_spk_playback") == 0 || strcmp(type, "clear_spk_queue") == 0) {
        ret = app_uac_clear_spk_queue();
        if (ret == ESP_OK) {
            ret = send_text_to_req_ws(req, "{\"type\":\"spk_reset_done\"}");
        }
    } else {
        ESP_LOGW(TAG, "Unsupported WS command: %s", type ? type : "unknown");
        ret = ESP_ERR_NOT_SUPPORTED;
    }

    if (ret != ESP_OK) {
        app_web_send_error("command_failed", esp_err_to_name(ret));
    }
    cJSON_Delete(root);
    return ret;
}

static esp_err_t handle_ws_binary(httpd_req_t *req, const uint8_t *payload, size_t len)
{
    // Binary frames are reserved for browser-to-speaker PCM chunks.
    if (len <= APP_WEB_AUDIO_HEADER_SIZE || payload[0] != APP_WEB_AUDIO_SPK_TYPE) {
        ESP_LOGW(TAG, "Invalid WS binary frame");
        return ESP_ERR_INVALID_ARG;
    }
    int ws_fd = httpd_req_to_sockfd(req);
    if (!audio_ws_is_current(ws_fd)) {
        ESP_LOGW(TAG, "Drop SPK PCM from stale fd=%d current=%d", ws_fd, audio_ws_get_fd());
        return ESP_ERR_INVALID_STATE;
    }
    uint16_t payload_len = payload[2] | (payload[3] << 8);
    if (payload_len != len - APP_WEB_AUDIO_HEADER_SIZE) {
        ESP_LOGW(TAG, "Invalid PCM payload length");
        return ESP_ERR_INVALID_SIZE;
    }
    esp_err_t ret = app_uac_queue_spk_pcm(payload + APP_WEB_AUDIO_HEADER_SIZE, payload_len);
    if (ret != ESP_OK) {
        app_web_send_error("spk_pcm_failed", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t audio_ws_handler(httpd_req_t *req)
{
    int ws_fd = httpd_req_to_sockfd(req);
    httpd_ws_client_info_t info = httpd_ws_get_fd_info(s_server, ws_fd);
    if (info == HTTPD_WS_CLIENT_WEBSOCKET && !audio_ws_is_current(ws_fd)) {
        audio_ws_open(ws_fd);
        ESP_LOGI(TAG, "Audio WebSocket session active fd=%d", ws_fd);
    }
    if (req->method == HTTP_GET) {
        audio_ws_open(ws_fd);
        ESP_LOGI(TAG, "Audio WebSocket connected fd=%d", ws_fd);
        app_web_broadcast_state();
        return ESP_OK;
    }

    httpd_ws_frame_t frame = {0};
    esp_err_t ret = httpd_ws_recv_frame(req, &frame, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Get audio WS frame length failed: %s", esp_err_to_name(ret));
        return ret;
    }
    if (frame.len > APP_WEB_WS_RX_BUF_SIZE) {
        ESP_LOGW(TAG, "Audio WS frame too large: %u", frame.len);
        return ESP_ERR_INVALID_SIZE;
    }
    uint8_t *buf = calloc(1, frame.len + 1);
    if (!buf) {
        ESP_LOGE(TAG, "Allocate WS rx buffer failed");
        return ESP_ERR_NO_MEM;
    }
    frame.payload = buf;
    ret = httpd_ws_recv_frame(req, &frame, frame.len);
    if (ret == ESP_OK) {
        if (frame.type == HTTPD_WS_TYPE_TEXT) {
            buf[frame.len] = '\0';
            ret = handle_ws_text(req, (const char *)buf);
        } else if (frame.type == HTTPD_WS_TYPE_BINARY) {
            ret = handle_ws_binary(req, buf, frame.len);
        } else if (frame.type == HTTPD_WS_TYPE_CLOSE) {
            int ws_fd = httpd_req_to_sockfd(req);
            ESP_LOGI(TAG, "Audio WebSocket closed fd=%d", ws_fd);
            audio_ws_clear_if_current(ws_fd);
        }
    } else {
        ESP_LOGE(TAG, "Receive audio WS frame failed: %s", esp_err_to_name(ret));
    }
    free(buf);
    return ret;
}

static void app_web_close_fn(httpd_handle_t hd, int sockfd)
{
    (void)hd;
    int current_fd = audio_ws_get_fd();
    if (sockfd == current_fd) {
        ESP_LOGI(TAG, "HTTPD closing current audio WebSocket fd=%d", sockfd);
    } else {
        ESP_LOGD(TAG, "HTTPD closing HTTP fd=%d current_audio_ws_fd=%d", sockfd, current_fd);
    }
    audio_ws_clear_if_current(sockfd);
    camera_ws_clear_if_current(sockfd);
    close(sockfd);
}

static void put_le16(uint8_t *buf, uint16_t value)
{
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
}

static void put_le32(uint8_t *buf, uint32_t value)
{
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
}

static esp_err_t camera_ws_send_packet(int fd, const uint8_t *data, size_t len)
{
    if (!s_server || fd < 0 || !data || len == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!camera_ws_is_current(fd)) {
        return ESP_ERR_INVALID_STATE;
    }
    httpd_ws_client_info_t info = httpd_ws_get_fd_info(s_server, fd);
    if (info != HTTPD_WS_CLIENT_WEBSOCKET) {
        ESP_LOGW(TAG, "Camera WS send skipped: fd=%d info=%s len=%u", fd, ws_info_name(info), (unsigned)len);
        camera_ws_clear_if_current(fd);
        return ESP_ERR_INVALID_STATE;
    }

    httpd_ws_frame_t frame = {
        .type = HTTPD_WS_TYPE_BINARY,
        .payload = (uint8_t *)data,
        .len = len,
    };
    esp_err_t ret = httpd_ws_send_frame_async(s_server, fd, &frame);
    if (ret == ESP_OK) {
        return ESP_OK;
    }

    ESP_LOGW(TAG, "Camera WS send failed fd=%d: %s", fd, esp_err_to_name(ret));
    camera_ws_clear_if_current(fd);
    if (s_server) {
        httpd_sess_trigger_close(s_server, fd);
    }
    return ret;
}

static int camera_ws_get_fd(void)
{
    return ws_get_fd(&s_camera_ws_fd);
}

static esp_err_t camera_ws_send_text(int fd, const char *text)
{
    if (!s_server || fd < 0 || !text) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!camera_ws_is_current(fd)) {
        return ESP_ERR_INVALID_STATE;
    }
    httpd_ws_client_info_t info = httpd_ws_get_fd_info(s_server, fd);
    if (info != HTTPD_WS_CLIENT_WEBSOCKET) {
        ESP_LOGW(TAG, "Camera WS text send skipped: fd=%d info=%s", fd, ws_info_name(info));
        camera_ws_clear_if_current(fd);
        return ESP_ERR_INVALID_STATE;
    }

    httpd_ws_frame_t frame = {
        .type = HTTPD_WS_TYPE_TEXT,
        .payload = (uint8_t *)text,
        .len = strlen(text),
    };
    esp_err_t ret = httpd_ws_send_frame_async(s_server, fd, &frame);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Camera WS text send failed fd=%d: %s", fd, esp_err_to_name(ret));
        camera_ws_clear_if_current(fd);
    }
    return ret;
}

static void camera_ws_stream_task(void *arg)
{
    int fd = (int)(intptr_t)arg;
    int64_t last_frame_us = 0;
    int64_t window_start_us = 0;
    uint32_t window_frame_count = 0;
    bool stream_ready_logged = false;
    uint32_t frame_count = 0;
    ESP_LOGI(TAG, "Camera WS stream task started fd=%d", fd);
    while (camera_ws_is_current(fd)) {
        app_uvc_format_t format = app_uvc_get_format();
        bool streaming = app_uvc_is_streaming();
        if (format != APP_UVC_FORMAT_MJPEG || !streaming) {
            stream_ready_logged = false;
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        if (!stream_ready_logged) {
            ESP_LOGI(TAG, "Camera WS stream ready fd=%d", fd);
            stream_ready_logged = true;
        }

        uvc_host_frame_t *frame = app_uvc_get_frame(pdMS_TO_TICKS(APP_WEB_CAMERA_FRAME_WAIT_MS));
        if (!frame) {
            ESP_LOGW(TAG, "Get camera frame timeout fd=%d streaming=%d current=%d", fd, app_uvc_is_streaming(), camera_ws_is_current(fd));
            continue;
        }

        size_t packet_len = APP_WEB_CAMERA_HEADER_SIZE + frame->data_len;
        uint8_t *packet = malloc(packet_len);
        if (!packet) {
            ESP_LOGE(TAG, "Allocate camera WS packet failed len=%u", (unsigned)packet_len);
            app_uvc_return_frame(frame);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        packet[0] = 'C';
        packet[1] = 1;
        packet[2] = APP_WEB_CAMERA_FORMAT_MJPEG;
        packet[3] = 0;
        put_le32(packet + 4, frame->data_len);
        put_le32(packet + 8, (uint32_t)(esp_timer_get_time() / 1000));
        put_le16(packet + 12, frame->vs_format.h_res);
        put_le16(packet + 14, frame->vs_format.v_res);
        memcpy(packet + APP_WEB_CAMERA_HEADER_SIZE, frame->data, frame->data_len);
        size_t frame_len = frame->data_len;
        esp_err_t frame_ret = app_uvc_return_frame(frame);
        if (frame_ret != ESP_OK) {
            ESP_LOGW(TAG, "Return camera WS frame failed: %s", esp_err_to_name(frame_ret));
        }

        esp_err_t ret = camera_ws_send_packet(fd, packet, packet_len);
        free(packet);
        if (ret != ESP_OK) {
            ESP_LOGI(TAG, "Camera WS stream stopped fd=%d after %"PRIu32" frames: %s", fd, frame_count, esp_err_to_name(ret));
            break;
        }

        frame_count++;
        int64_t now_us = esp_timer_get_time();
        float inst_fps = last_frame_us > 0 ? 1000000.0f / (float)(now_us - last_frame_us) : 0.0f;
        last_frame_us = now_us;
        if (window_start_us == 0) {
            window_start_us = now_us;
            window_frame_count = 0;
        }
        window_frame_count++;
        if (frame_count == 1 || frame_count % 30 == 0) {
            int64_t window_us = now_us - window_start_us;
            float avg_fps = window_us > 0 ? (float)window_frame_count * 1000000.0f / (float)window_us : 0.0f;
            ESP_LOGI(TAG, "Camera WS frame sent fd=%d count=%"PRIu32" len=%u avg_fps=%.2f inst_fps=%.2f window_frames=%"PRIu32,
                     fd, frame_count, (unsigned)frame_len, avg_fps, inst_fps, window_frame_count);
            window_start_us = now_us;
            window_frame_count = 0;
        }
    }
    ESP_LOGI(TAG, "Camera WS stream task exit fd=%d frames=%"PRIu32, fd, frame_count);
    camera_ws_clear_if_current(fd);
    vTaskDelete(NULL);
}

static esp_err_t camera_ws_start_stream(int fd)
{
    if (fd < 0) {
        return ESP_ERR_INVALID_ARG;
    }
    if (camera_ws_is_current(fd)) {
        return ESP_OK;
    }

    camera_ws_open(fd);
    if (xTaskCreate(camera_ws_stream_task, "camera_ws", 4096, (void *)(intptr_t)fd, 5, NULL) != pdPASS) {
        ESP_LOGE(TAG, "Create camera WebSocket stream task failed");
        camera_ws_clear_if_current(fd);
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

static esp_err_t handle_camera_ws_text(const char *msg)
{
    bool b = false;
    esp_err_t ret = ESP_OK;
    cJSON *root = cJSON_Parse(msg);
    if (!root) {
        ESP_LOGW(TAG, "Parse camera WS JSON failed");
        app_web_send_error("invalid_json", "Invalid camera JSON command");
        return ESP_ERR_INVALID_ARG;
    }

    const char *type = cjson_get_type(root);
    if (!type) {
        ESP_LOGW(TAG, "Camera WS command missing type");
        ret = ESP_ERR_INVALID_ARG;
    } else if (strcmp(type, "set_camera_enabled") == 0 && cjson_get_bool(root, "enabled", &b)) {
        ESP_LOGI(TAG, "Camera WS command set_enabled=%d", b);
        ret = app_uvc_set_enabled(b);
    } else if (strcmp(type, "set_crop") == 0) {
        float left = 0, top = 0, width = 0, height = 0;
        if (!cjson_get_float(root, "left", &left) || !cjson_get_float(root, "top", &top) ||
            !cjson_get_float(root, "width", &width) || !cjson_get_float(root, "height", &height)) {
            ret = ESP_ERR_INVALID_ARG;
        } else {
            // Keeps the ESP32-side crop -> 40x40 patch cut (app_ai_patch.c)
            // aligned with the red crop box the user sees on the web page.
            ret = app_ai_patch_set_crop_percent(left, top, width, height);
        }
    } else if (strcmp(type, "set_camera_format") == 0) {
        const char *format = cjson_get_string(root, "format");
        uint32_t resolution = 0;
        if (!format || !cjson_get_u32(root, "resolution", &resolution)) {
            ret = ESP_ERR_INVALID_ARG;
        } else if (strcmp(format, "mjpeg") == 0) {
            ret = app_uvc_set_format(APP_UVC_FORMAT_MJPEG, (uint8_t)resolution);
        } else if (strcmp(format, "h264") == 0) {
            ret = app_uvc_set_format(APP_UVC_FORMAT_H264, (uint8_t)resolution);
        } else {
            ESP_LOGW(TAG, "Unsupported camera WS format command: %s", format);
            ret = ESP_ERR_NOT_SUPPORTED;
        }
    } else {
        ESP_LOGW(TAG, "Unsupported camera WS command: %s", type ? type : "unknown");
        ret = ESP_ERR_NOT_SUPPORTED;
    }

    if (ret != ESP_OK) {
        app_web_send_error("camera_command_failed", esp_err_to_name(ret));
    }
    cJSON_Delete(root);
    return ret;
}

static esp_err_t camera_ws_handler(httpd_req_t *req)
{
    int ws_fd = httpd_req_to_sockfd(req);
    if (req->method == HTTP_GET) {
        return ESP_OK;
    }

    httpd_ws_frame_t frame = {0};
    esp_err_t ret = httpd_ws_recv_frame(req, &frame, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Get camera WS frame length failed: %s", esp_err_to_name(ret));
        return ret;
    }
    if (frame.len > APP_WEB_CAMERA_WS_RX_BUF_SIZE) {
        ESP_LOGW(TAG, "Camera WS control frame too large: %u", (unsigned)frame.len);
        return ESP_ERR_INVALID_SIZE;
    }
    uint8_t payload[APP_WEB_CAMERA_WS_RX_BUF_SIZE + 1] = {0};
    if (frame.len > 0) {
        frame.payload = payload;
        ret = httpd_ws_recv_frame(req, &frame, frame.len);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Receive camera WS frame payload failed: %s", esp_err_to_name(ret));
            return ret;
        }
        payload[frame.len] = '\0';
    }
    if (frame.type == HTTPD_WS_TYPE_CLOSE) {
        ESP_LOGI(TAG, "Camera WebSocket closed fd=%d", ws_fd);
        camera_ws_clear_if_current(ws_fd);
    } else {
        // Consume the browser control frame before starting the producer to keep the TCP stream aligned.
        if (frame.type == HTTPD_WS_TYPE_TEXT && frame.len > 0) {
            ret = handle_camera_ws_text((const char *)payload);
            if (ret != ESP_OK) {
                return ret;
            }
        }
        ret = camera_ws_start_stream(ws_fd);
    }
    return ret;
}

esp_err_t app_web_start(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.uri_match_fn = httpd_uri_match_wildcard;
    config.max_uri_handlers = 24;
    config.close_fn = app_web_close_fn;
    /* FIX "httpd_accept_conn: error in accept (23)" = ENFILE (hết socket
     * toàn hệ thống). Với CONFIG_LWIP_MAX_SOCKETS=10 mà phải chia sẻ cho
     * captive DNS, Wi-Fi AP/DHCP, WebSocket camera + audio..., httpd cần
     * giới hạn số socket nó tự giữ và PHẢI bật lru_purge_enable để tự đóng
     * kết nối cũ/im lặng lâu, nếu không khi hết socket 1 lần thì accept()
     * sẽ lỗi (23) VĨNH VIỄN từ đó về sau (đúng như log bạn thấy).
     * -> Nhớ tăng thêm CONFIG_LWIP_MAX_SOCKETS trong sdkconfig (xem ghi chú
     * bên dưới hàm này) để có dư chỗ cho toàn hệ thống. */
    config.max_open_sockets = 4;
    config.lru_purge_enable = true;
    config.backlog_conn = 2;
    // Keep enough worker stack for the existing HTTP/audio/camera control
    // handlers. Multi-photo P4/P3 processing itself runs in its own tasks.
    config.stack_size = 12288;
    config.task_priority = 4;

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "HTTP server start failed: %s", esp_err_to_name(ret));
        return ret;
    }

    const httpd_uri_t uris[] = {
        {.uri = "/", .method = HTTP_GET, .handler = index_handler},
        {.uri = "/assets/style.css", .method = HTTP_GET, .handler = style_handler},
        {.uri = "/assets/app.js", .method = HTTP_GET, .handler = app_js_handler},
        {.uri = "/assets/mic-player-worklet.js", .method = HTTP_GET, .handler = mic_player_worklet_js_handler},
        {.uri = "/api/devices", .method = HTTP_GET, .handler = devices_handler},
        {.uri = "/api/ai_patch/capture", .method = HTTP_POST, .handler = ai_patch_capture_handler},
        {.uri = "/api/ai_patch/status", .method = HTTP_GET, .handler = ai_patch_status_handler},
        {.uri = "/api/ai_patch/patch", .method = HTTP_GET, .handler = ai_patch_patch_handler},
        {.uri = "/api/ai_patch/photo", .method = HTTP_GET, .handler = ai_patch_photo_handler},
        {.uri = "/api/ai_infer/features", .method = HTTP_GET, .handler = ai_infer_features_handler},
        {.uri = "/api/ai_session/status", .method = HTTP_GET, .handler = ai_session_status_handler},
        {.uri = "/api/ai_session/features", .method = HTTP_GET, .handler = ai_session_features_handler},
        {.uri = "/api/ai_coreset/status", .method = HTTP_GET, .handler = ai_coreset_status_handler},
        {.uri = "/api/ai_coreset/bank", .method = HTTP_GET, .handler = ai_coreset_bank_handler},
        {.uri = "/camera_ws", .method = HTTP_GET, .handler = camera_ws_handler, .is_websocket = true},
        {.uri = "/ws", .method = HTTP_GET, .handler = audio_ws_handler, .is_websocket = true},
        {.uri = "/connecttest.txt", .method = HTTP_GET, .handler = captive_probe_handler},
        {.uri = "/redirect", .method = HTTP_GET, .handler = captive_probe_handler},
        {.uri = "/generate_204", .method = HTTP_GET, .handler = captive_probe_handler},
        {.uri = "/hotspot-detect.html", .method = HTTP_GET, .handler = captive_probe_handler},
        {.uri = "/ncsi.txt", .method = HTTP_GET, .handler = captive_probe_handler},
    };
    for (size_t i = 0; i < sizeof(uris) / sizeof(uris[0]); i++) {
        ret = httpd_register_uri_handler(s_server, &uris[i]);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Register URI %s failed: %s", uris[i].uri, esp_err_to_name(ret));
            return ret;
        }
        if (strcmp(uris[i].uri, "/camera_ws") == 0) {
            ESP_LOGW(TAG, "Registered camera WebSocket URI");
        }
    }
    ret = httpd_register_err_handler(s_server, HTTPD_404_NOT_FOUND, captive_404_handler);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Register captive 404 handler failed: %s", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "HTTP server started");
    return ESP_OK;
}

bool app_web_has_client(void)
{
    return audio_ws_get_fd() >= 0;
}

esp_err_t app_web_broadcast_state(void)
{
    if (!s_server) {
        ESP_LOGW(TAG, "Skip state broadcast: server is not started");
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t ret = queue_audio_ws_work(APP_AUDIO_WS_WORK_STATE, NULL, 0, false);
    return ret;
}

esp_err_t app_web_send_error(const char *code, const char *message)
{
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Create error JSON failed");
        return ESP_ERR_NO_MEM;
    }
    cJSON_AddStringToObject(root, "type", "error");
    cJSON_AddStringToObject(root, "code", code ? code : "unknown");
    cJSON_AddStringToObject(root, "message", message ? message : "unknown error");
    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json) {
        ESP_LOGE(TAG, "Print error JSON failed");
        return ESP_ERR_NO_MEM;
    }
    esp_err_t ret = queue_audio_ws_work(APP_AUDIO_WS_WORK_TEXT, (const uint8_t *)json, strlen(json), false);
    cJSON_free(json);
    return ret;
}

esp_err_t app_web_send_spk_refill_budget_bytes(uint32_t bytes)
{
    if (bytes == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    char json[64];
    snprintf(json, sizeof(json), "{\"type\":\"spk_refill_budget\",\"bytes\":%"PRIu32"}", bytes);
    return queue_audio_ws_work(APP_AUDIO_WS_WORK_TEXT, (const uint8_t *)json, strlen(json), true);
}

esp_err_t app_web_send_mic_pcm(const uint8_t *data, size_t len)
{
    // Prefix raw PCM with a compact frame header so the browser can route it.
    if (!data || len == 0 || len > UINT16_MAX) {
        return ESP_ERR_INVALID_ARG;
    }
    return queue_audio_ws_mic_pcm(data, len);
}

esp_err_t app_web_notify_ai_session(const char *event_name)
{
    if (!s_server) {
        return ESP_ERR_INVALID_STATE;
    }
    int fd = camera_ws_get_fd();
    if (fd < 0) {
        return ESP_ERR_INVALID_STATE;
    }
    char json[96];
    snprintf(json, sizeof(json), "{\"type\":\"ai_session\",\"event\":\"%s\"}",
             event_name ? event_name : "update");
    return camera_ws_send_text(fd, json);
}

esp_err_t app_web_notify_capture_button(void)
{
    // Called by main.c only *after* the ESP32-side capture (grab frame ->
    // decode -> crop -> cut patches, see app_ai_patch_capture()) triggered
    // by the physical GPIO4 (P4) button has already finished. This just
    // tells whichever browser currently has the camera WebSocket open that
    // fresh results are ready, so it can pull the captured photo
    // (/api/ai_patch/photo) and patches (/api/ai_patch/patch) - both of
    // which are guaranteed to come from that same just-finished capture.
    if (!s_server) {
        ESP_LOGW(TAG, "Skip capture-button notify: HTTP server is not started");
        return ESP_ERR_INVALID_STATE;
    }
    int fd = camera_ws_get_fd();
    if (fd < 0) {
        ESP_LOGW(TAG, "Capture button pressed, but no browser has the camera page open");
        return ESP_ERR_INVALID_STATE;
    }
    esp_err_t ret = camera_ws_send_text(fd, "{\"type\":\"capture_photo\",\"source\":\"button\"}");
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Notify capture button to browser failed: %s", esp_err_to_name(ret));
    }
    return ret;
}