# Bao_cao-Pham-Quoc-Anh-

I. Công việc nhóm thực hiện 
1. Phần AI

II. Công việc chi tiết 
1. Phần AI

a. Hướng đi của nhóm về AI
+) Thu thập và tổng hợp dư liệu trên mạng theo nguồn (Git và roboflow) phù hợp cho mô hình mobilenetV2 
+) Tiền xử lý và cắt ảnh thành từng patch 224x224 để tăng số lượng và nổi bật đặc trưng của ảnh
+) Bổ sung thêm giả lập tạo lỗi nhân tạo (Vì ảnh theo nguồn từ kaggle là ảnh tổng hợp ko có lỗi)
+) Xây dựng mô hình Feature Extractor dựa trên MobileNetV2 nhằm học vector đặc trưng 32D
+) Áp dụng phương pháp Triplet Learning để tối ưu không gian đặc trưng, giúp tăng khả năng phân biệt giữa PCB đạt và PCB lỗi.

b. Tiến độ công việc AI hiện tại

- Thu thập và xây dựng dữ liệu
+) Xây dựng bộ dữ liệu gồm hai lớp Normal và Defect, mỗi lớp khoảng 3.000 ảnh.
- Tiền xử lý dữ liệu
+) Cắt vùng mạch bằng thuật toán Canny Edge.
+) Chia ảnh thành các patch 224×224.
+) Sinh thêm các mẫu lỗi nhân tạo để tăng tính đa dạng dữ liệu.
+) Chia dữ liệu thành các tập Train, Val và Test.

- Kiểm tra dữ liệu
+) Kiểm tra trực quan các mẫu ảnh sau tiền xử lý.
+) Xác nhận dữ liệu đạt kích thước chuẩn 224×224×3 và sẵn sàng cho quá trình huấn luyện.

- Xây dựng mô hình
+) Thiết kế mô hình trích xuất đặc trưng dựa trên MobileNetV2 (alpha = 0.5).
+) Xây dựng các tầng Dense theo cấu trúc phân cấp 640 → 256 → 64 → 32.
+) Chuẩn hóa vector đầu ra bằng L2 Normalization.

- Huấn luyện mô hình
+) Xây dựng sử dụng Triplet Loss.
+) Áp dụng tăng cường dữ liệu nhằm tăng khả năng tổng quát hóa.
+) Huấn luyện theo hai giai đoạn:
    Huấn luyện các tầng đặc trưng.
    Fine-tuning các lớp cuối của MobileNetV2.

- Đánh giá mô hình
+) Đánh giá bằng thuật toán KNN sử dụng Cosine Similarity.
+) Phân tích các chỉ số Accuracy, Precision, Recall, F1-score và Confusion Matrix.

c. Việc làm tiếp theo sau trích xuất 