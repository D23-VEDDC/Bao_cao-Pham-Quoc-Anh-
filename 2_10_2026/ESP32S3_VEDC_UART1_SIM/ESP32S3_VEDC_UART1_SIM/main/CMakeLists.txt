idf_component_register(
    SRCS
        "main.c"
        "vedc_controller.c"
        "app_protocol.c"
        "app_ai_coreset.c"
        "app_ai_infer.cpp"
        "app_ai_patch.c"
        "app_ai_session.c"
        "app_uac_manager.c"
        "app_usb_host.c"
        "app_uvc_manager.c"
        "app_web.c"
        "app_wifi.c"
    INCLUDE_DIRS "."
    EMBED_FILES
        "feature_extractor_32d_full_int8.tflite"
        "web/index.html"
        "web/style.css"
        "web/app.js"
        "web/mic-player-worklet.js"
    REQUIRES
        nvs_flash
        esp-tflite-micro
        esp-nn
        driver
        esp_timer
        esp_http_server
        esp_wifi
        esp_netif
        cjson
        esp-dsp
        captive_dns
        usb_host_uvc
        usb_host_uac
)

# Tối ưu hóa hiệu năng biên dịch O3 và tính toán số thực nhanh cho AI
# (không -UESP_NN nữa, để kernel esp_nn tăng tốc phần cứng được dùng)
add_compile_options(-O3 -ffast-math)