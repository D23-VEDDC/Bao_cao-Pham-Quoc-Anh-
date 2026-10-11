# ESP-NN kernel bisect

The normal project build keeps ESP-NN enabled. Only the kernels listed by
`VEDC_REF_KERNELS` are compiled with the TFLM reference implementation.

In `components/espressif__esp-tflite-micro/CMakeLists.txt`, edit:

```cmake
set(VEDC_REF_KERNELS "")
```

Examples:

```cmake
set(VEDC_REF_KERNELS "conv depthwise_conv fully_connected pooling add mul softmax")
set(VEDC_REF_KERNELS "conv depthwise_conv fully_connected")
set(VEDC_REF_KERNELS "conv")
```

For every change on Windows/ESP-IDF:

```text
idf.py fullclean
idf.py build
idf.py -p COMx flash monitor
```

Interpretation:

* PASS with all 7 reference: the deterministic TFLM path is correct.
* Split the list in half. If the result still fails, the culprit is in the
  half you removed; if it passes, it is in the half that was removed.
* After identifying the kernel, narrow it to individual model shapes.
  Keep the fast ESP-NN path for all safe shapes and use a local fallback for
  only the known-bad shape, as already done in `conv.cc`.

Do not enable `CONFIG_VEDC_FORCE_TFLM_REFERENCE=y` while doing the selective
bisect: that option disables ESP-NN for the whole component and masks the
per-kernel CMake switch.
