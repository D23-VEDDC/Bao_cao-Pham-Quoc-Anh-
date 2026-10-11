#include "app_web_server.h"

#include <stdio.h>
#include <string.h>
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "app_ai_patch.h"
#include "app_uvc_manager.h"
#include "vedc_controller.h"

static const char *TAG = "app_web";
static httpd_handle_t s_server;
static httpd_handle_t s_stream_server;

#define STREAM_PORT 81
#define STREAM_BOUNDARY "vedcframe"

/* Giới hạn fps preview. Camera đẩy ~30 fps VGA MJPEG (~10+ Mbps) làm nghẽn Wi-Fi
 * AP của ESP32 => /api/status bị timeout (web báo "Mất kết nối") và hình chớp.
 * Preview chỉ để giám sát nên ~12 fps là đủ; AI vẫn chụp full-res riêng. */
#define STREAM_MIN_FRAME_US      (80 * 1000)
/* Camera báo "streaming" nhưng không ra frame quá lâu => restart stream UVC. */
#define STREAM_STALL_US          (6 * 1000 * 1000)
#define STREAM_RESTART_GAP_US    (20 * 1000 * 1000)
/* Số frame hỏng liên tiếp tối đa bị loại bỏ trước khi chấp nhận (tránh đen màn hình). */
#define STREAM_MAX_BAD_FRAMES    10

extern const uint8_t web_index_html_start[] asm("_binary_index_html_start");
extern const uint8_t web_index_html_end[] asm("_binary_index_html_end");
extern const uint8_t web_app_js_start[] asm("_binary_app_js_start");
extern const uint8_t web_app_js_end[] asm("_binary_app_js_end");
extern const uint8_t web_style_css_start[] asm("_binary_style_css_start");
extern const uint8_t web_style_css_end[] asm("_binary_style_css_end");

static esp_err_t send_asset(httpd_req_t *req, const uint8_t *start, const uint8_t *end, const char *content_type)
{
    httpd_resp_set_type(req, content_type);
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, (const char *)start, (ssize_t)(end - start));
}

static esp_err_t index_handler(httpd_req_t *req)
{
    return send_asset(req, web_index_html_start, web_index_html_end, "text/html; charset=utf-8");
}

static esp_err_t js_handler(httpd_req_t *req)
{
    return send_asset(req, web_app_js_start, web_app_js_end, "application/javascript; charset=utf-8");
}

static esp_err_t css_handler(httpd_req_t *req)
{
    return send_asset(req, web_style_css_start, web_style_css_end, "text/css; charset=utf-8");
}

/* Returns the exact JPEG frame that was used for the most recent ESP32 AI
 * capture. There is intentionally no capture trigger in the web app: STM32
 * owns the sequence and this endpoint is read-only. */
static esp_err_t last_image_handler(httpd_req_t *req)
{
    uint8_t *jpeg = NULL;
    size_t jpeg_len = 0;
    esp_err_t ret = app_ai_patch_copy_last_jpeg(&jpeg, &jpeg_len);
    if (ret != ESP_OK) {
        return httpd_resp_send_err(req,
                                   ret == ESP_ERR_NOT_FOUND ? HTTPD_404_NOT_FOUND : HTTPD_500_INTERNAL_SERVER_ERROR,
                                   esp_err_to_name(ret));
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");
    ret = httpd_resp_send(req, (const char *)jpeg, (ssize_t)jpeg_len);
    heap_caps_free(jpeg);
    return ret;
}

#define STATUS_JSON_BUF_SIZE 16384

static esp_err_t status_handler(httpd_req_t *req)
{
    /* The controller snapshot is the single source of truth for the UI. It is
     * assembled from the same protocol state machine that answers STM32.
     *
     * FIX: buffer 16 KB KHÔNG được để trên stack (httpd stack chỉ 8-12 KB) ->
     * trước đây gây stack overflow, task httpd crash/reboot nên trình duyệt
     * báo "Không kết nối được ESP32-S3 Web Console". Cấp phát trên heap. */
    char *response = heap_caps_malloc(STATUS_JSON_BUF_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!response) {
        response = heap_caps_malloc(STATUS_JSON_BUF_SIZE, MALLOC_CAP_8BIT);
    }
    if (!response) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no memory");
    }

    esp_err_t ret = vedc_controller_get_status_json(response, STATUS_JSON_BUF_SIZE);
    if (ret != ESP_OK) {
        heap_caps_free(response);
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, esp_err_to_name(ret));
    }
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");
    ret = httpd_resp_send(req, response, (ssize_t)strlen(response));
    heap_caps_free(response);
    return ret;
}

/* Lấy frame MỚI NHẤT. Queue UVC giữ tối đa 2 frame và driver bỏ frame mới khi
 * queue đầy => người đọc chậm sẽ luôn nhận frame CŨ (đây là nguồn trễ ~vài giây).
 * Ở đây đọc hết queue, trả các frame cũ về pool và chỉ giữ frame cuối. */
static uvc_host_frame_t *get_latest_frame(TickType_t first_wait)
{
    uvc_host_frame_t *latest = app_uvc_get_frame(first_wait);
    if (!latest) return NULL;
    uvc_host_frame_t *newer;
    while ((newer = app_uvc_get_frame(0)) != NULL) {
        app_uvc_return_frame(latest);
        latest = newer;
    }
    return latest;
}

/* Snapshot 1 ảnh JPEG (giữ tương thích, dùng khi cần). */
static esp_err_t stream_handler(httpd_req_t *req)
{
    esp_err_t ret = app_uvc_set_enabled(true);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "camera stream unavailable");
    }

    uvc_host_frame_t *frame = NULL;
    for (int i = 0; i < 15 && !frame; ++i) {
        if (!app_uvc_is_streaming()) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        frame = get_latest_frame(pdMS_TO_TICKS(1000));
        break;
    }
    if (!frame) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "camera frame unavailable");
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");
    ret = httpd_resp_send(req, (const char *)frame->data, (ssize_t)frame->data_len);
    esp_err_t return_ret = app_uvc_return_frame(frame);
    if (return_ret != ESP_OK) {
        ESP_LOGW(TAG, "Return streamed UVC frame failed: %s", esp_err_to_name(return_ret));
    }
    return ret;
}

/* Frame JPEG hợp lệ? Frame bị cắt/đứt do mất gói USB isoc là nguồn chính gây
 * nhấp nháy: trình duyệt vẽ nửa ảnh/ảnh xám rồi lại ảnh đúng. */
static bool jpeg_frame_looks_valid(const uint8_t *d, size_t len)
{
    if (!d || len < 4) return false;
    if (d[0] != 0xFF || d[1] != 0xD8) return false;            /* SOI */
    /* Một số camera đệm thêm byte sau EOI -> tìm EOI trong 2 KB cuối thay vì 32 byte. */
    size_t tail = len > 2048 ? len - 2048 : 0;
    for (size_t i = len - 1; i > tail; --i) {
        if (d[i - 1] == 0xFF && d[i] == 0xD9) return true;
    }
    return false;
}

/* MJPEG liên tục (multipart/x-mixed-replace) chạy trên server riêng cổng 81.
 *
 * LƯU Ý: httpd của ESP-IDF chỉ có MỘT task => handler này chiếm cả server cổng 81.
 * Nếu client biến mất âm thầm (Wi-Fi chập chờn, tab bị treo) mà server không
 * phát hiện, mọi kết nối /stream mới phải chờ handler cũ chết => khung camera
 * đen hàng chục giây. Vì vậy: bật TCP keepalive + send timeout ngắn, và luôn
 * gửi keepalive khi chưa có frame. */
static esp_err_t mjpeg_stream_handler(httpd_req_t *req)
{
    esp_err_t ret = app_uvc_set_enabled(true);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        return httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "camera stream unavailable");
    }

    httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" STREAM_BOUNDARY);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, max-age=0");

    int sock = httpd_req_to_sockfd(req);
    int one = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Phát hiện client chết sau ~5 s (2 + 1*3) thay vì chờ TCP timeout hàng phút. */
    int ka_idle = 2, ka_intvl = 1, ka_cnt = 3;
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &ka_idle, sizeof(ka_idle));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &ka_intvl, sizeof(ka_intvl));
    setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &ka_cnt, sizeof(ka_cnt));

    /* Buffer riêng để COPY frame ra rồi trả frame về pool UVC ngay.
     * Trước đây frame bị giữ suốt lúc gửi qua Wi-Fi (có thể nhiều giây khi nghẽn)
     * => pool UVC cạn, driver tràn buffer, capture AI cũng bị đói frame. */
    uint8_t *copy = NULL;
    size_t copy_cap = 0;

    int idle_ms = 0;
    int bad_frames = 0;
    uint32_t st_frames = 0, st_bytes = 0, st_dropped = 0;
    int64_t st_t0 = esp_timer_get_time();
    int64_t last_sent_us = 0;
    int64_t last_frame_us = st_t0;
    int64_t last_restart_us = 0;

    while (true) {
        /* stream socket chỉ GỬI nên lru_purge luôn coi nó là "ít dùng nhất" -> chống bị đóng */
        httpd_sess_update_lru_counter(s_stream_server, sock);

        if (!app_uvc_is_streaming()) {
            app_uvc_set_enabled(true);
            vTaskDelay(pdMS_TO_TICKS(100));
            idle_ms += 100;
            last_frame_us = esp_timer_get_time();       /* chưa stream => chưa tính là treo */
            if (idle_ms >= 1000) {
                if (httpd_resp_send_chunk(req, "\r\n", 2) != ESP_OK) break;
                idle_ms = 0;
            }
            continue;
        }

        /* Throttle: ngủ cho đủ khoảng cách giữa 2 frame rồi mới lấy frame MỚI NHẤT. */
        int64_t now = esp_timer_get_time();
        int64_t wait_us = STREAM_MIN_FRAME_US - (now - last_sent_us);
        if (last_sent_us && wait_us > 1000) {
            vTaskDelay(pdMS_TO_TICKS((uint32_t)(wait_us / 1000)) + 1);
        }

        uvc_host_frame_t *frame = get_latest_frame(pdMS_TO_TICKS(300));
        now = esp_timer_get_time();
        if (!frame) {
            idle_ms += 300;
            if (idle_ms >= 1000) {
                if (httpd_resp_send_chunk(req, "\r\n", 2) != ESP_OK) break;
                idle_ms = 0;
            }
            /* Watchdog: UVC báo đang stream nhưng không có frame => restart camera. */
            if (now - last_frame_us > STREAM_STALL_US && now - last_restart_us > STREAM_RESTART_GAP_US) {
                ESP_LOGW(TAG, "stream stalled %lld ms, restarting UVC", (long long)((now - last_frame_us) / 1000));
                app_uvc_restart_stream();
                last_restart_us = now;
                last_frame_us = now;
            }
            continue;
        }
        last_frame_us = now;
        idle_ms = 0;

        size_t len = frame->data_len;
        if (!jpeg_frame_looks_valid(frame->data, len) && bad_frames < STREAM_MAX_BAD_FRAMES) {
            bad_frames++;
            st_dropped++;
            app_uvc_return_frame(frame);
            continue;
        }
        bad_frames = 0;

        const uint8_t *src = frame->data;
        if (len + 64 > copy_cap) {
            size_t want = len + 8192;
            uint8_t *n = heap_caps_realloc(copy, want, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!n) n = heap_caps_realloc(copy, want, MALLOC_CAP_8BIT);
            if (n) { copy = n; copy_cap = want; }
        }
        if (copy && len <= copy_cap) {
            memcpy(copy, frame->data, len);
            app_uvc_return_frame(frame);                /* trả pool ngay, trước khi gửi */
            frame = NULL;
            src = copy;
        }

        char part[96];
        int hlen = snprintf(part, sizeof(part),
                            "\r\n--" STREAM_BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                            (unsigned)len);
        ret = httpd_resp_send_chunk(req, part, hlen);
        if (ret == ESP_OK) {
            ret = httpd_resp_send_chunk(req, (const char *)src, (ssize_t)len);
        }
        if (frame) app_uvc_return_frame(frame);         /* nhánh fallback không copy được */
        if (ret != ESP_OK) break;                       /* client đóng kết nối */

        last_sent_us = esp_timer_get_time();
        st_frames++; st_bytes += (uint32_t)len;
        if (last_sent_us - st_t0 >= 3000000) {
            ESP_LOGI(TAG, "stream: %.1f fps, avg %u KB/frame, dropped(bad) %u",
                     st_frames * 1e6f / (float)(last_sent_us - st_t0),
                     (unsigned)(st_frames ? st_bytes / st_frames / 1024 : 0), (unsigned)st_dropped);
            st_frames = 0; st_bytes = 0; st_dropped = 0; st_t0 = last_sent_us;
        }
    }
    heap_caps_free(copy);
    httpd_resp_send_chunk(req, NULL, 0);
    return ret;
}

esp_err_t app_web_server_start(void)
{
    if (s_server) {
        return ESP_OK;
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 8;
    config.max_open_sockets = 5;
    config.stack_size = 12288;
    /* UVC/USB host chạy priority cao nhất trên core 0 -> không ghim httpd vào core 0 */
    config.core_id = tskNO_AFFINITY;
    /* /api/status rất nhẹ nhưng trước đây phải tranh CPU với command_worker (prio 9,
     * core 0) đang decode JPEG + chạy TFLite => web bị đứng lúc chụp. Cho httpd
     * chính ưu tiên cao hơn worker (9) và UART event task (10). */
    config.task_priority = 11;
    config.lru_purge_enable = true;   /* tự đóng socket cũ khi trình duyệt mở nhiều kết nối */
    config.recv_wait_timeout = 5;
    config.send_wait_timeout = 5;

    esp_err_t ret = httpd_start(&s_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "HTTP server start failed: %s", esp_err_to_name(ret));
        return ret;
    }

    const httpd_uri_t routes[] = {
        { .uri = "/", .method = HTTP_GET, .handler = index_handler },
        { .uri = "/app.js", .method = HTTP_GET, .handler = js_handler },
        { .uri = "/style.css", .method = HTTP_GET, .handler = css_handler },
        { .uri = "/api/status", .method = HTTP_GET, .handler = status_handler },
        { .uri = "/api/last-image", .method = HTTP_GET, .handler = last_image_handler },
        /* Backward-compatible alias; still read-only. */
        { .uri = "/api/capture/image", .method = HTTP_GET, .handler = last_image_handler },
        { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler },
    };

    for (size_t i = 0; i < sizeof(routes) / sizeof(routes[0]); ++i) {
        ret = httpd_register_uri_handler(s_server, &routes[i]);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Register URI %s failed: %s", routes[i].uri, esp_err_to_name(ret));
            httpd_stop(s_server);
            s_server = NULL;
            return ret;
        }
    }

    /* Server riêng cho MJPEG stream (cổng 81). */
    httpd_config_t scfg = HTTPD_DEFAULT_CONFIG();
    scfg.server_port = STREAM_PORT;
    scfg.ctrl_port = scfg.ctrl_port + 1;
    scfg.max_uri_handlers = 2;
    scfg.max_open_sockets = 4;          /* chừa chỗ cho kết nối preconnect/retry của trình duyệt */
    scfg.stack_size = 8192;
    scfg.core_id = tskNO_AFFINITY;
    scfg.task_priority = 8;             /* cao hơn mặc định 5 để không bị đói CPU lúc AI chạy */
    scfg.lru_purge_enable = true;
    scfg.recv_wait_timeout = 5;
    scfg.send_wait_timeout = 4;         /* client chết -> handler thoát trong ~4 s, nhường server (1 task) cho kết nối mới */
    ret = httpd_start(&s_stream_server, &scfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Stream server start failed: %s", esp_err_to_name(ret));
        s_stream_server = NULL;
    } else {
        const httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = mjpeg_stream_handler };
        httpd_register_uri_handler(s_stream_server, &stream_uri);
        ESP_LOGI(TAG, "MJPEG stream ready at http://192.168.4.1:%d/stream", STREAM_PORT);
    }

    ESP_LOGI(TAG, "Observation-only Web UI ready at http://192.168.4.1");
    return ESP_OK;
}