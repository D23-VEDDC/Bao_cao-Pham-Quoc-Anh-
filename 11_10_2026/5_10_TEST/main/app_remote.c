#include "app_remote.h"

#include <stdio.h>
#include <string.h>
#include "esp_http_client.h"
#include "esp_log.h"

static const char *TAG = "app_remote";

esp_err_t app_remote_upload_inspection(const char *matrix_json)
{
    if (!matrix_json) {
        return ESP_ERR_INVALID_ARG;
    }

    const char *url = CONFIG_VEDC_SERVER_UPLOAD_URL;
    if (!url[0]) {
        ESP_LOGW(TAG, "Inspection upload skipped: server URL is empty");
        return ESP_ERR_INVALID_STATE;
    }
    const size_t matrix_len = strlen(matrix_json);

    esp_http_client_config_t config = {
        .url = url,
        .timeout_ms = 10000,
        .disable_auto_redirect = true,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        ESP_LOGE(TAG, "HTTP client init failed");
        return ESP_ERR_NO_MEM;
    }

    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_header(client, "Content-Type", "application/json");

    esp_err_t ret = esp_http_client_open(client, matrix_len);
    if (ret == ESP_OK) ret = esp_http_client_write(client, matrix_json, matrix_len);
    if (ret == ESP_OK) {
        int content_length = esp_http_client_fetch_headers(client);
        if (content_length < 0) {
            ret = ESP_FAIL;
        } else {
            int status = esp_http_client_get_status_code(client);
            if (status < 200 || status >= 300) {
                ESP_LOGW(TAG, "Upload server returned HTTP %d", status);
                ret = ESP_FAIL;
            } else {
                ESP_LOGI(TAG, "Inspection matrix uploaded: %u bytes HTTP %d",
                         (unsigned)matrix_len, status);
            }
        }
    }
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return ret;
}
