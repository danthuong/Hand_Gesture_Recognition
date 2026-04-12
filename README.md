# Hand Gesture Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)](https://google.github.io/mediapipe/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Static%20Gestures-green.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GRU%20Dynamic-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv8-Person%20Detection-yellow.svg)](https://ultralytics.com/)

> Real-time hand gesture recognition system combining computer vision and deep learning to classify hand gestures (both static and dynamic).

## 📊 Project Overview

| Metric | Value |
|--------|-------|
| Static Gestures | 6 classes |
| Dynamic Motions | 2 classes |
| Hand Landmarks | 21 keypoints |
| Max Hands Tracked | 4 simultaneous |
| Detection Model | YOLOv8x |
| Static Classifier | XGBoost |
| Dynamic Classifier | GRU |

**Source:** Custom dataset collected using MediaPipe Hand Landmarker

---

## 📚 Project Components

| Phần | Nội dung | Người thực hiện | Status |
|------|----------|----------------|--------|
| 🧠 **1. Introduction** | Project overview & motivation | Team | ✅ Done |
| 📖 **2. Related Work** | Literature review | Team | ✅ Done |
| 🧹 **3. Data Preprocessing** | EDA & feature engineering | Team | ✅ Done |
| 🤖 **4. ML Models** | XGBoost & GRU architecture | Team | ✅ Done |
| 🧪 **5. Experiment** | Experimental setup | Team | ✅ Done |
| 📊 **6. Results** | Performance analysis | Team | ✅ Done |
| 🎯 **7. Conclusion** | Summary & future work | Team | ✅ Done |
| 📚 **8. References** | Academic references | Team | ✅ Done |

---

## 🗂️ Project Structure

```
Hand_Gesture_Recognition/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Images/                      # Dashboard images & plots
│   ├── 01_gesture_distribution.png
│   ├── 02_feature_boxplot_p8y.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_tsne_projection.png
│   ├── 05_mean_profile.png
│   └── *.png
├── reports/                     # LaTeX report files
│   ├── main.tex
│   └── Sections/
│       ├── 0-Title.tex
│       ├── 1-Intro.tex
│       ├── 2-RelatedWork.tex
│       ├── 3-DataPreprocessing.tex
│       ├── 4-MLModels.tex
│       ├── 5-Experiment.tex
│       ├── 6-Results.tex
│       ├── 7-Conclusion.tex
│       └── 8-References.tex
├── ML (1)/                      # LaTeX template reference
├── collect_dynamic_data/        # Dynamic data collection scripts
├── data/                        # Training data
├── human_detection/
│   └── human_detector.py        # YOLO person detection
└── motion_detection/
    ├── main.py                  # Main application
    ├── handlers/
    │   ├── kp_extractor.py      # MediaPipe wrapper
    │   ├── gru.py               # GRU model
    │   ├── audio_handler.py     # Audio processing
    │   └── detect_human.py      # Human detection wrapper
    ├── utils/
    │   ├── motion_utils.py
    │   ├── hand_helpers.py
    │   ├── visualizer.py
    │   ├── logger.py
    │   └── preprocess.py
    ├── scripts/
    │   ├── train_model.py       # XGBoost training
    │   ├── train_gru.py         # GRU training
    │   ├── prepare_data.py
    │   └── create_csv.py
    ├── models/
    │   ├── yolov8n.pt
    │   ├── gesture_recognizer.task
    │   ├── hand_landmarker.task
    │   ├── gesture_model.pkl
    │   ├── motion_model.pth
    │   └── hand_gestures.csv
    └── test/
        ├── test_camera.py
        ├── test_hand.py
        ├── test_yolo.py
        └── test_mic.py
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/Hand_Gesture_Recognition.git
cd Hand_Gesture_Recognition

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run the System

```bash
# Start hand gesture recognition
python motion_detection/main.py

# Train XGBoost model (static gestures)
python motion_detection/scripts/train_model.py

# Train GRU model (dynamic motions)
python motion_detection/scripts/train_gru.py
```

**Controls:**
- `q`: Quit application

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV, MediaPipe |
| Object Detection | YOLOv8 (Ultralytics) |
| Static Classification | XGBoost |
| Dynamic Classification | PyTorch GRU |
| Report | LaTeX (Overleaf) |
| Visualization | Plotly, Seaborn |

---

## 🎯 Supported Gestures

### Static Gestures (XGBoost)
| Gesture | Description |
|---------|-------------|
| `1-one` | Index finger up |
| `2-two` | Two fingers up |
| `3-three` | Three fingers up |
| `5-rotate` | Five fingers open |
| `7-victory` | Victory sign (V) |
| `4-open_close` | Open/Close hand |

### Dynamic Motions (GRU)
| Motion | Description |
|--------|-------------|
| `Clap` | Clapping motion |
| `Shake` | Hand shaking |

---

## 🔑 Key Features

1. **Multi-Person Tracking**: Simultaneous tracking of multiple people
2. **Real-time Processing**: High FPS real-time processing
3. **Noise Filtering**:
   - Buffer voting (30 frames) for GRU
   - Hold-to-confirm (1 second) for static gestures
   - Confidence threshold (85%)
4. **Smart Cooldown**: 1.5s cooldown to prevent duplicate triggers

---

## 📜 License

MIT License

---

## 👥 Contributors

| STT | Họ & tên | MSSV | Nhiệm vụ đảm nhiệm | % Hoàn thành |
|-----|----------|------|---------------------|--------------|
| 1 | Đào Quang Dương | 2310579 | - Lập trình và huấn luyện mô hình nhận diện (XGBoost, mạng GRU).<br>- Tích hợp luồng camera xử lý thời gian thực (Real-time).<br>- Tinh chỉnh siêu tham số (Hyperparameter Tuning). | 100% |
| 2 | Võ Thanh Đạt | 2310717 | - Xử lý số liệu, xuất các ma trận nhầm lẫn và metrics đánh giá.<br>- Thiết kế và xây dựng Dashboard trực quan hóa trên Power BI.<br>- Hỗ trợ phân tích lỗi (Error Analysis) từ các biểu đồ thực nghiệm. | 100% |
| 3 | Hà Bảo Nhi | 2312496 | - Khai phá dữ liệu (EDA) và Tiền xử lý (Preprocessing) chuyên sâu.<br>- Tham gia lập trình thuật toán và trích xuất đặc trưng tay.<br>- Soạn thảo báo cáo, tổng hợp và phân tích kết quả thực nghiệm. | 100% |

**Course:** Data Mining

---

## 🔗 References

1. [MediaPipe Hand Landmarker](https://google.github.io/mediapipe/solutions/hands)
2. [YOLOv8 Documentation](https://docs.ultralytics.com/)
3. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
4. [GRU: Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1412.3555)

---

*Last Updated: April 2026*
