# Hệ thống Dashboard Phân Tích (Power BI)

Thư mục này chứa hình ảnh xuất ra từ các Dashboard Power BI được thiết kế để theo dõi, đánh giá và giải thích hoạt động của toàn bộ Hệ thống Nhận diện Cử chỉ Tay. Các Dashboard trải dài từ khâu hiểu dữ liệu, đánh giá độ chính xác tĩnh cho đến phân tích hành vi động/hiệu năng hệ thống.

Dưới đây là chi tiết 3 trang Dashboard chính:

### 1. Tổng quan & Tiền xử lý (Overview and Preprocessing)

![Overview and Preprocessing](/Images/Overview%20and%20Preprocessing.png)

Trang Dashboard này đóng vai trò như một bức tranh toàn cảnh về dữ liệu đầu vào và các đặc trưng nhận diện:
- **Phân bổ Dữ liệu (Class Balance):** Cho thấy số lượng mẫu của các lớp (One, Two, Three, Victory, Open_Close).
- **Phân tích Đặc trưng (Radar Chart / Feature Importance):** Giải thích (Interpretability) cách mô hình XGBoost phân loại cử chỉ tĩnh. Biểu đồ Radar ("Dấu vân tay cử chỉ") dựa trên khoảng cách Euclidean từ cổ tay (p0) đến các đầu ngón tay làm nổi bật sự khác biệt giữa các hành động.
- **Tiền xử lý (Landmarks Transformation & Performance):** Biểu đồ Scatter minh họa trực quan quá trình dời tâm và chuẩn hóa tỷ lệ (scale) các điểm tọa độ bàn tay. Kế tiếp là biểu đồ cột nhằm so sánh hiệu suất độ chính xác (Accuracy vs Macro_F1) của mô hình trước và sau khi áp dụng Data Preprocessing.

**💡 Chẩn đoán & Insight chi tiết:**
- **Tính Cân bằng & Toàn vẹn Dữ liệu (Data Completeness):** Biểu đồ Class Balance cho thấy việc thu thập dữ liệu rải đều ở các lớp. Việc không có nhãn thiểu số (Minority Class) giúp mô hình XGBoost không bị rơi vào thái cực "thiên vị" (Bias/Overfitting) nghiêng về một cử chỉ riêng biệt.
- **Tính tất yếu của việc Scale & Dời Tâm (Translation & Scale Invariance):** Thông qua biểu đồ Scatter, ta thấy rõ tọa độ gốc ban đầu dính "nhiễu không gian" (Bị trôi đi theo vị trí tay đặt xa gần trên camera). Khi các điểm tọa độ được scale đồng bộ và dời tâm gốc tọa độ (After Preprocessing), biểu đồ Cột bên cạnh đã chứng minh tác dụng kinh ngạc của nó: Giúp Accuracy mạnh mẽ vượt từ 0.84 (Raw Data) lên 0.91 (Preprocessed Data). Khâu này ép AI loại bỏ "vị trí đứng" và chỉ còn tập trung vào đúng "hình thái tay học", là cốt lõi của tính chính xác trên toàn hệ thống Real-time.

### 2. Đánh giá Cử chỉ Tĩnh (Static Gesture Action Recognition)

![Static Gesture Action Recognition](/Images/Static%20Gesture%20Action%20Recognition.png)

Trang này phân tích sâu hiệu suất của mô hình học máy (XGBoost) trên tập Testing Split (20% dữ liệu ẩn để tránh rò rỉ):
- **Confusion Matrix (Ma trận nhầm lẫn):** Minh họa độ chính xác (Precision) và độ nhạy (Recall) cho từng cử chỉ độc lập. Báo cáo tỷ lệ nhận diện nhầm lẫn (False Positives/False Negatives).
- **Accuracy Metrics (KPIs):** Theo dõi F1-Score, Overall Accuracy đạt được. Mọi đánh giá đều xuất phát nghiêm ngặt từ tập Test, đảm bảo tính Generalization của Model khi triển khai thực tế.
- **Radar Chart So sánh:** Đặt các "dấu vân tay" của dự đoán đúng cạnh dự đoán sai (nếu có) để phân tích nguyên nhân tại sao thuật toán lại bị nhầm lẫn ở gốc rễ hình học không gian.

**💡 Chẩn đoán & Insight chi tiết:**
- **Không rò rỉ dữ liệu (No Data Leakage):** Việc tính toán toàn bộ KPIs trên phân vùng 20% Testing hứa hẹn tỷ lệ Accuracy/F1-Score là hiệu suất "thực chiến" khi vận hành trên webcam thực tế, chứ không phải sự học vẹt (overfitting) của mô hình.
- **Tối ưu hóa Báo động giả (False Positives Management):** Khi xem xét Confusion Matrix (Ma trận nhầm lẫn), chúng ta có thể chỉ mặt những cặp cử chỉ dễ bị thuật toán nhầm lẫn với nhau nhất. Insight này đặc biệt quan trọng, vì nó biện minh cho thiết kế "Hold-to-confirm" (Cần ít nhất 12/15 frame cố định) trong code `main.py` để chủ động gọt dũa những nhầm lẫn tức thời.
- **X-Quang Lỗi Hình Học (Root-Cause Analysis):** Khi Radar Chart đặt một mẫu "Nhận diện Trật" cạnh một mẫu "Nhận diện Trúng", nó mang tầm vóc của một công cụ dò lỗi nguyên nhân gốc (Root-cause). Lấy ví dụ, nếu Radar Chart chỉ ra đặc trưng $p_{12}$ (đầu ngón giữa) của bản sai bị thấp hơn bản đúng, ta lập tức biết nguyên nhân sai là do **người dùng gập chưa hết ngón tay** thay vì đổ lỗi cho thuật toán XGBoost tính toán kém.

### 3. Đánh giá Hành động Động & Hiệu năng (Dynamic Gesture and System Performance)

![Dynamic Gesture and System Performance](/Images/Dynamic%20Gesture%20and%20System%20Performance.png)

Đây là linh hồn của hệ thống Real-Time (Đo kiểm Kịch bản 3 Giai đoạn thực tế):
- **Phân tích Cử chỉ Động (GRU Model):** Đánh giá độ tin cậy của mạng RNN/GRU trong việc trích xuất chuỗi 50 khung hình cho hành động `Clap` (vỗ tay) và `Shake` (lắc tay).
- **System Health Telemetry (FPS Jitter Test):** Biểu đồ thể hiện tốc độ khung hình (Actual FPS) được ghi nhận trực tiếp từ vòng lặp OpenCV (`main.py`). Gồm 3 đoạn thực tế test: (1) Lý tưởng, (2) Thay đổi khoảng cách, và (3) Xử lý rung rắc/mờ (Motion Blur). Đường biên trung bình giúp khẳng định hệ thống đạt chuẩn Real-Time (~30 FPS).
- **Vùng hoạt động an toàn (Operating Envelope):** Biểu đồ tương quan kép giữa *Diện tích Bounding Box* và *Độ tự tin Mô hình*. Thể hiện quá trình lùi dần ra xa khỏi hệ thống (Bounding Box giảm) nhưng độ chính xác (Model Confidence) vẫn bám trụ bền bỉ trên mức 70%.

**💡 Chẩn đoán & Insight chi tiết:**
- **Giải phẫu Nút thắt cổ chai (Bottleneck Analysis):** Dòng FPS Jitter Test cho thấy hệ thống chạy phẳng và mượt mờ ở mức ~32 FPS (Giai đoạn lý tưởng). Các "gai" đột biến (spikes) vọt lên 150-180 FPS ở giai đoạn rung lắc mạnh không phải là lỗi. Nó minh chứng cho hiện tượng OpenCV "xả buffer đệm" sau khi AI tốn mili-giây xử lý khó khăn các khung hình bị mờ (motion blur). Đường Average neo cứng ở 31.90 FPS khẳng định hệ thống đạt chuẩn Real-Time không có bất kỳ vấn đề nghẽn bộ nhớ (memory leak) nào.
- **Xác định Ngưỡng vận hành từ xa (Distance Boundary):** Đồ thị kép Operating Envelope khen ngợi độ bao quát (Generalization) của AI. Kể cả khi user lùi xa, kích cỡ Box sụt mất một nửa (Vùng Xanh lún xuống), thì Model Confidence (Đường Vàng) chỉ tụt nhẹ từ 1.0 xuống 0.7 - 0.8 chứ không bao giờ chạm đáy, đảm bảo thao tác điều khiển thiết bị vẫn diễn ra thông suốt.
- **Chứng minh Cơ chế "Khóa Nhiễu":** Ở giai đoạn rung tay cực nhanh (Jitter), đồ thị Vàng gãy đoạn rớt liên tục về 0. Tuy nhiên, thông qua các khoá an toàn bảo vệ như *Buffer Voting* (Với GRU phải 20/30 khung hình đồng thuận) và *Hold-to-confirm* (1 giây chờ), hệ thống thành công dập tắt tất cả các báo động giả tồi tệ nhất, biến một luồng webcam thô (jittery) thành những dòng lệnh (trigger) tĩnh tại, chắc chắn cho môi trường Production.
