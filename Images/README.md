# Hệ thống Dashboard Phân Tích (Power BI)

Thư mục này chứa hình ảnh xuất ra từ các Dashboard Power BI được thiết kế để theo dõi, đánh giá và giải thích hoạt động của toàn bộ Hệ thống Nhận diện Cử chỉ Tay. Các Dashboard trải dài từ khâu hiểu dữ liệu, đánh giá độ chính xác tĩnh cho đến phân tích hành vi động/hiệu năng hệ thống.

Dưới đây là chi tiết 3 trang Dashboard chính:

### 1. Tổng quan & Tiền xử lý (Overview and Preprocessing)
**Tập tin minh hoạ:** `Overview and Preprocessing.png`

Trang Dashboard này đóng vai trò như một bức tranh toàn cảnh về dữ liệu đầu vào và các đặc trưng nhận diện:
- **Phân bổ Dữ liệu (Class Balance):** Cho thấy số lượng mẫu của các lớp (One, Two, Three, Victory, Open_Close).
- **Phân tích Đặc trưng (Radar Chart / Feature Importance):** Giải thích (Interpretability) cách mô hình XGBoost phân loại cử chỉ tĩnh. Biểu đồ Radar ("Dấu vân tay cử chỉ") dựa trên khoảng cách Euclidean từ cổ tay (p0) đến các đầu ngón tay làm nổi bật sự khác biệt giữa các hành động.
- **Tiền xử lý (Preprocessing Quality):** Trực quan hóa tọa độ (Landmarks) sau khi đã được chuẩn hóa (Normalize) về tâm, chứng minh thuật toán loại bỏ hoàn toàn nhiễu từ vị trí đứng hoặc kích cỡ tay trong khung hình.

### 2. Đánh giá Cử chỉ Tĩnh (Static Gesture Action Recognition)
**Tập tin minh hoạ:** `Static Gesture Action Recognition.png`

Trang này phân tích sâu hiệu suất của mô hình học máy (XGBoost) trên tập Testing Split (20% dữ liệu ẩn để tránh rò rỉ):
- **Confusion Matrix (Ma trận nhầm lẫn):** Minh họa độ chính xác (Precision) và độ nhạy (Recall) cho từng cử chỉ độc lập. Báo cáo tỷ lệ nhận diện nhầm lẫn (False Positives/False Negatives).
- **Accuracy Metrics (KPIs):** Theo dõi F1-Score, Overall Accuracy đạt được. Mọi đánh giá đều xuất phát nghiêm ngặt từ tập Test, đảm bảo tính Generalization của Model khi triển khai thực tế.
- **Radar Chart So sánh:** Đặt các "dấu vân tay" của dự đoán đúng cạnh dự đoán sai (nếu có) để phân tích nguyên nhân tại sao thuật toán lại bị nhầm lẫn ở gốc rễ hình học không gian.

### 3. Đánh giá Hành động Động & Hiệu năng (Dynamic Gesture and System Performance)
**Tập tin minh hoạ:** `Dynamic Gesture and System Performance.png`

Đây là linh hồn của hệ thống Real-Time (Đo kiểm Kịch bản 3 Giai đoạn thực tế):
- **Phân tích Cử chỉ Động (GRU Model):** Đánh giá độ tin cậy của mạng RNN/GRU trong việc trích xuất chuỗi 50 khung hình cho hành động `Clap` (vỗ tay) và `Shake` (lắc tay).
- **System Health Telemetry (FPS Jitter Test):** Biểu đồ thể hiện tốc độ khung hình (Actual FPS) được ghi nhận trực tiếp từ vòng lặp OpenCV (`main.py`). Gồm 3 đoạn thực tế test: (1) Lý tưởng, (2) Thay đổi khoảng cách, và (3) Xử lý rung rắc/mờ (Motion Blur). Đường biên trung bình giúp khẳng định hệ thống đạt chuẩn Real-Time (~30 FPS).
- **Vùng hoạt động an toàn (Operating Envelope):** Biểu đồ tương quan kép giữa *Diện tích Bounding Box* và *Độ tự tin Mô hình*. Thể hiện quá trình lùi dần ra xa khỏi hệ thống (Bounding Box giảm) nhưng độ chính xác (Model Confidence) vẫn bám trụ bền bỉ trên mức 70%.
