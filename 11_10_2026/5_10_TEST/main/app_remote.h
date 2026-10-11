#pragma once

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t app_remote_upload_inspection(const char *matrix_json);

#ifdef __cplusplus
}
#endif
