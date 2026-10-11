# VEDC – ESP32-S3 Vision App đồng bộ STM32F103

Bản này thay thế Web UI camera cũ bằng **Observation-only Vision Console**. Browser không còn quyền trigger capture/train/check; **STM32F103 là master của trình tự**, ESP32-S3 chỉ nhận protocol, chụp/AI/đoạn tính toán và trả kết quả.

## 1. Giao thức STM32 ↔ ESP32-S3 đã đồng bộ

Protocol giữ nguyên frame:

- `AA | LEN | CMD | DATA | CRC8(poly 0x07) | 55`
- UART1: 115200 8N1, ESP32-S3 RX=GPIO18, TX=GPIO17.

Các transaction được hỗ trợ đúng flow STM32:

| STM32 → ESP32-S3 | ESP32-S3 → STM32 | Ý nghĩa |
|---|---|---|
| `0x10 LEARN_START` | `0xFF ACK` | bắt đầu Learn |
| `0x11 CAPTURE_SAMPLE[idx]` | `0x91 CAPTURE_ACK` | chụp 5 mẫu ngoại quan |
| `0x12 POWER_SAMPLE[idx,Imax,Iavg]` | `0x92 POWER_ACK` | nhận 5 mẫu dòng |
| `0x13 TRAIN_START` | `0x93 TRAIN_DONE[26]` | build Memory Bank |
| `0x20 CHECK_START` | `0xFF ACK` | bắt đầu Check |
| `0x21 VISUAL_CHECK[board]` | `0xA1 VISUAL_RESULT[11]` | AI ngoại quan |
| `0x22 POWER_CHECK[board,Imax,Iavg]` | `0xA2 POWER_RESULT[19]` | kiểm tra dòng |
| `0x23 CHECK_NEXT[board]` | `0xFF ACK` | sang PCB kế tiếp |

### Retry/idempotent

STM32 có retry khi mất ACK/RESULT. ESP32-S3 đã cache transaction để retry **không gây chụp/Train/Check lần hai và không tăng board index hai lần**:

- CAPTURE cùng `idx` → trả lại ACK cũ.
- POWER_SAMPLE cùng `idx` → trả lại ACK.
- TRAIN_START sau khi train xong → trả lại `TRAIN_DONE` cũ.
- VISUAL_CHECK cùng board → trả lại `VISUAL_RESULT` cũ.
- POWER_CHECK cùng board + payload → trả lại `POWER_RESULT` cũ.
- CHECK_NEXT cùng board vừa xử lý → chỉ ACK, không increment lần nữa.

## 2. Web App mới

Truy cập `http://192.168.4.1`.

UI có 3 tầng quan sát:

1. **Pipeline LEARN → TRAIN → CHECK** với stage hiện tại.
2. **Live camera** ở khung trái.
3. **Ảnh inspection** ở khung phải, là đúng JPEG của frame đã dùng để cắt patch/AI.

Không còn nút `Chụp & phân tích`.

### Khi VISUAL FAIL

ESP32-S3 lưu:

- `capture_id`
- `cols / rows`
- `crop_width / crop_height`
- `threshold`
- score của từng patch 40×40
- số patch lỗi
- `max_dmin`

Browser lấy đúng JPEG cuối và dựng overlay theo cùng geometry center-crop 40×40. Các patch có `score > threshold` được:

- khoanh khung đỏ trên ảnh gốc ở panel bên phải;
- tô đỏ nhẹ vùng patch;
- liệt kê tối đa 8 patch lỗi có score cao nhất ở panel `Patch lỗi nổi bật`.

Do overlay được tính từ **chính crop/patch geometry của frame đã AI**, không có lần chụp thứ hai chỉ để hiển thị.

## 3. Web API

- `GET /api/status` – snapshot read-only của controller.
- `GET /api/last-image` – JPEG cuối do STM32-triggered capture tạo ra.
- `GET /stream` – live preview.
- `GET /api/capture/image` – alias tương thích cũ, cũng chỉ read-only.

`POST /api/capture` đã được bỏ khỏi Web UI và route server.

## 4. Runtime startup

UART1 của STM32 vẫn được mở sớm để không mất command lúc boot. Controller chỉ thực thi CAPTURE/TRAIN/CHECK sau khi UVC + AI patch + TFLite + Coreset đã init xong.

## 5. Build

Project này dùng ESP-IDF. Thư mục `build/` và `.git/` không được đóng gói vào bản source sạch để tránh mang theo artifact/build cũ; mở project bằng ESP-IDF và build lại.
