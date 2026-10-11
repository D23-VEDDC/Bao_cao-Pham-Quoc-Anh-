# Kiểm tra độ đúng vector sau khi flash

## 1. Kiểm tra nhanh trên ESP32

Sau P4/P3, log cần có dạng:

```text
[AI ARENA] location=INTERNAL ...
[AI PERF] invokes=... avg=... us min=... us max=... us ...
[AI DONE] ... status=ok elapsed=... s
```

Nếu xuất hiện:

```text
[ESP-NN FIX] 1x1 Conv channels not multiple of 16 ...
[ESP-NN FIX] 2x2 pointwise Conv -> TFLM reference
```

thì chỉ những shape Conv đó dùng reference; phần còn lại vẫn dùng ESP-NN.

## 2. So sánh với PC

Cài Python package:

```text
pip install numpy requests pillow tensorflow
```

hoặc dùng `tflite-runtime` thay cho `tensorflow` nếu môi trường của bạn có wheel phù hợp.

Ví dụ với một ảnh/pipeline cuối cùng trên ESP32:

```text
python tools/validate_vectors.py --device http://192.168.4.1 --model main/feature_extractor_32d_full_int8.tflite --max-patches 8
```

Nếu dùng phiên P3 nhiều ảnh, chọn một ảnh:

```text
python tools/validate_vectors.py --device http://192.168.4.1 --model main/feature_extractor_32d_full_int8.tflite --max-patches 8 --session-image 0
```

Script lấy đúng patch 40x40 từ ESP32, chạy cùng model int8 trên PC, dequantize + L2-normalize và so sánh cosine. Mốc thực hành là cosine >= 0.99 cho toàn bộ các patch đã chọn.

## 3. Kiểm tra thứ hạng

Script cũng dùng patch 0 làm query và so sánh thứ tự nearest-neighbor giữa vector ESP32 và PC. Nếu cosine tốt nhưng ranking lệch nhiều, cần kiểm tra preprocessing / output normalization tiếp.
