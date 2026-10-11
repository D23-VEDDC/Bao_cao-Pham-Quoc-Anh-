idf_component_register(
    SRCS
        "main.c"
        "vedc_controller.c"
        "app_protocol.c"
        "app_ai_coreset.c"
        "app_ai_infer.cpp"
        "app_ai_patch.c"
        "app_usb_host.c"
        "app_uvc_manager.c"
        "app_remote.c"
        "app_wifi.c"
        "app_web_server.c"
    INCLUDE_DIRS "."
    EMBED_FILES
        "feature_extractor_32d_full_int8.tflite"
        "../web/index.html"
        "../web/app.js"
        "../web/style.css"
        "../web/ptit-logo.jpg"
    REQUIRES
        nvs_flash
        esp-tflite-micro
        esp-nn
        driver
        esp_timer
        esp_wifi
        esp_netif
        esp_http_client
        esp_http_server
        cjson
        esp-dsp
        usb_host_uvc
        captive_dns
)

# Build application-side code at O3. ESP-NN is enabled by the local
# TFLM component CMake; only entries in VEDC_REF_KERNELS are reference kernels.
add_compile_options(-O3 -ffast-math)
