# Kiểm chứng ngoại quan bằng Memory Bank AI

Firmware giữ luồng gốc: model TFLite tạo vector 32 chiều cho mỗi patch 40×40,
LEARN chọn vector đại diện bằng greedy k-center, CHECK so từng vector với
Memory Bank toàn cục. Ảnh test không cần trùng vị trí với ảnh LEARN.

Điểm CHECK dùng láng giềng gần nhất cùng một phần khoảng cách tới láng giềng
thứ ba để tránh một vector trùng ngẫu nhiên che lỗi. Ngưỡng LEARN được hiệu
chỉnh bằng cách lần lượt bỏ một ảnh sạch ra và so nó với bốn ảnh còn lại.

1. Nạp firmware, reset LEARN và chụp 5 ảnh PCB sạch với mức xê dịch nhẹ thường
   gặp khi vận hành. Giữ ánh sáng và độ cao camera như lúc CHECK. Theo dõi log
   `cross-image max`. Nếu TRAIN báo một ảnh là outlier, reset LEARN và chụp
   lại 5 mẫu sạch; firmware không đưa bộ mẫu khác thường vào Memory Bank.
2. CHECK ít nhất 10 ảnh PCB sạch ở các vị trí khác nhau. Lưu `/api/status` và
   JPEG `/api/last-image` sau mỗi lượt.
3. CHECK vật lạ ở nhiều vị trí và kích thước. Đối chiếu `check.scores`,
   `check.threshold`, số patch lỗi và khung đỏ với đúng JPEG vừa chụp.

Nếu model cho vector gần như không đổi khi có vật lạ, cần đánh giá hoặc huấn
luyện lại chính feature extractor; chỉnh ngưỡng không thể tạo thông tin mà
vector AI không biểu diễn. Chưa có ảnh PCB mẫu trong kho mã, nên build thành
công không chứng minh độ chính xác phát hiện trên thiết bị.
