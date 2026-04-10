# Hand Gesture Recognition

Hệ thống nhận diện cử chỉ tay thời gian thực sử dụng kết hợp computer vision và deep learning để phân loại các cử chỉ tay (cả tĩnh và động).

## Tổng Quan Kiến Trúc

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │───▶│    YOLO     │───▶│  Multi-Person
│   Input     │    │  (Human)    │    │   Tracking
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
          ┌─────────────────┐                              ┌─────────────────┐
          │  MediaPipe      │                              │   Prediction    │
          │  Hand Landmarks │                              │    Results      │
          └────────┬────────┘                              └─────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌─────────────────┐    ┌─────────────────┐
│  XGBoost        │    │     GRU         │
│  (Static)       │    │   (Dynamic)     │
│  Gestures       │    │   Motion        │
└─────────────────┘    └─────────────────┘
```

## Các Thành Phần Chính

### 1. Human Detection (`human_detection/`)
- **Model**: YOLOv8x (Ultralytics)
- **Chức năng**: Phát hiện và theo dõi nhiều người trong khung hình
- **Output**: Bounding box + Track ID cho từng người

### 2. Hand Extraction (`motion_detection/handlers/kp_extractor.py`)
- **Model**: MediaPipe Hand Landmarker
- **Chức năng**: Trích xuất 21 landmarks của bàn tay (tọa độ x, y, z)
- **Hỗ trợ**: Tối đa 4 tay trong khung hình

### 3. Static Gesture Recognition (XGBoost)
- **Model**: `gesture_model.pkl`
- **Input**: 42 features (21 điểm × 2 trục x, y)
- **Cử chỉ hỗ trợ**:
  - `1-one`: Ngón trỏ giơ lên
  - `2-two`: Hai ngón giơ lên
  - `3-three`: Ba ngón giơ lên
  - `5-rotate`: 5 ngón tay 
  - `7-victory`: Ngón Victory (V)
  - `4-open_close`: Open/Close

### 4. Dynamic Motion Recognition (GRU)
- **Model**: `motion_model.pth` (PyTorch)
- **Input**: 50 frames × 252 features (tọa độ + velocity)
- **Hành động hỗ trợ**:
  - `Clap`: Vỗ tay
  - `Shake`: Lắc tay

## Cấu Trúc Thư Mục

```
Hand_Gesture_Recognition/
├── requirements.txt
├── .gitignore
├── Images/                    # Chứa hình ảnh và Dashboard Power BI
│   ├── README.md              # Giải thích các Dashboard (System Health, Gestures...)
│   └── *.png                  # Ảnh export từ Power BI
├── collect_dynamic_data/      # Script và thu thập dữ liệu hành động động
├── data/                      # Thư mục chứa dữ liệu ảnh/hành động dùng để huấn luyện
├── human_detection/
│   └── human_detector.py      # YOLO person detection
└── motion_detection/
    ├── main.py                # Main application (có tích hợp Telemetry tracking)
    ├── handlers/
    │   ├── kp_extractor.py    # MediaPipe wrapper
    │   ├── gru.py             # GRU model definition
    │   ├── audio_handler.py   # Audio processing
    │   └── detect_human.py    # Human detection wrapper
    ├── utils/
    │   ├── motion_utils.py    # Motion preprocessing
    │   ├── hand_helpers.py    # Hand landmark processing
    │   ├── visualizer.py      # Drawing utilities
    │   ├── logger.py          # Logging utilities
    │   └── preprocess.py      # Data preprocessing
    ├── scripts/
    │   ├── train_model.py         # Huấn luyện XGBoost & xuất Power BI metrics
    │   ├── train_gru.py           # Huấn luyện GRU & xuất Power BI metrics
    │   ├── export_radar_chart.py  # Trích xuất đặc trưng hình học bàn tay (Radar Chart)
    │   ├── prepare_data.py        # Tiền xử lý dữ liệu
    │   └── create_csv.py          # Tạo file CSV ban đầu
    ├── models/
    │   ├── yolov8n.pt / .pt       # YOLO models
    │   ├── gesture_recognizer.task  # MediaPipe
    │   ├── hand_landmarker.task     # MediaPipe
    │   ├── gesture_model.pkl       # XGBoost model
    │   ├── motion_model.pth        # GRU model
    │   └── hand_gestures.csv       # Training data
    └── test/
        ├── test_camera.py
        ├── test_hand.py
        ├── test_yolo.py
        └── test_mic.py
```

## Cài Đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt PyTorch với CUDA (nếu có GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Sử Dụng

```bash
# Chạy hệ thống nhận diện cử chỉ
python motion_detection/main.py

# Huấn luyện lại mô hình XGBoost (static gestures)
python motion_detection/scripts/train_model.py

# Huấn luyện lại mô hình GRU (dynamic motions)
python motion_detection/scripts/train_gru.py
```

Điều khiển:
- `q`: Thoát

## Các Tính Năng Chính

1. **Multi-Person Tracking**: Theo dõi và xử lý đồng thời nhiều người
2. **Real-time Processing**: Xử lý thời gian thực với FPS cao
3. **Noise Filtering**: 
   - Buffer voting (30 frames) cho GRU
   - Hold-to-confirm (1 giây) cho static gestures
   - Confidence threshold (85%)
4. **Smart Cooldown**: Tránh triggers trùng lặp với cooldown 1.5 giây

## Yêu Cầu Hệ Thống

- Python 3.8+
- OpenCV
- MediaPipe
- PyTorch (CUDA cho GPU)
- XGBoost
- Ultralytics (YOLO)

## License

MIT License
