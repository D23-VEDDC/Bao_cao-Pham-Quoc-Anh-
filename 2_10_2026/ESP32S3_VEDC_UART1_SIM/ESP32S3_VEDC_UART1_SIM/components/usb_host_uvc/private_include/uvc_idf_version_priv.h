/*
 * SPDX-FileCopyrightText: 2024-2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sdkconfig.h"
#include "esp_idf_version.h"

// @todo fix the hard-coded number here: Should be taken from HAL FIFO config in future versions of esp-idf
#if (CONFIG_IDF_TARGET_ESP32P4 || CONFIG_IDF_TARGET_LINUX)
#define MAX_MPS_IN 4096
#else
// NOTE (patched): on ESP32-S2/S3 the DWC USB-OTG HCD hard-caps periodic IN
// endpoints to 408 bytes at runtime (see hcd_dwc.c mps_limits.in_mps),
// regardless of the "Hardware FIFO size biasing" Kconfig choice
// (upstream esp-idf issue #14941 - the bias options don't actually change
// this at runtime). The original 596 value here assumes a limit that the
// hardware doesn't actually give us, so any endpoint with wMaxPacketSize
// between 409 and 596 gets picked here but then rejected later with
// "EP MPS (xxx) exceeds supported limit (408)". Lowering this to 408
// makes uvc_desc_get_streaming_intf_and_ep() pick a smaller (working)
// alternate setting instead.
#define MAX_MPS_IN 408
#endif

// Definition of USB_EP_DESC_GET_MULT for IDF versions that don't have it.
// It was introduced in IDF v5.3 and backported to v5.2.1 and v5.1.4
#if !(ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 1, 4) && ESP_IDF_VERSION != ESP_IDF_VERSION_VAL(5, 2, 0))
#define USB_EP_DESC_GET_MULT(desc_ptr) (((desc_ptr)->wMaxPacketSize & 0x1800) >> 11)
#endif
