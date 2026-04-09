import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys

class HandExtractor:
    def __init__(self, mp_model='gesture_recognizer.task', max_hands=6):
        print("[MODULE_HAND_POSE] Đang khởi tạo MediaPipe (LIVE_STREAM)...")
        self.latest_hands = [] 
        self.latest_gestures = []
        
        # BIẾN LƯU FRAME VÀ FPS CỦA MEDIAPIPE
        self.latest_frame = None 
        self.mp_prev_time = time.time()
        self.mp_fps = 0
        
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=mp_model),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=max_hands,
            result_callback=self._result_callback       
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

    def _result_callback(self, result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        extracted_hands = []
        gestures = []
        
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                kp_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
                extracted_hands.append(kp_array)
                
        if result.gestures:
            for gesture_list in result.gestures:
                if len(gesture_list) > 0:
                    gestures.append(gesture_list[0].category_name)
        
        # 1. Trích xuất frame chuẩn từ MediaPipe (Nó trả về RGB, ta phải đổi lại thành BGR để cv2.imshow)
        frame_rgb = output_image.numpy_view()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 2. Tính toán FPS thực tế của MediaPipe
        current_time = time.time()
        fps = 1 / (current_time - self.mp_prev_time) if (current_time - self.mp_prev_time) > 0 else 0
        self.mp_prev_time = current_time

        # 3. Cập nhật dữ liệu
        self.latest_hands = extracted_hands
        self.latest_gestures = gestures
        self.latest_frame = frame_bgr
        self.mp_fps = int(fps)

    def process_frame_async(self, frame, timestamp_ms):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.recognizer.recognize_async(mp_image, timestamp_ms)

    def get_latest_results(self):
        # Hàm này giờ trả về cả frame và fps
        return self.latest_hands, self.latest_gestures, self.latest_frame, self.mp_fps

    def cleanup(self):
        self.recognizer.close()
        print("[MODULE_HAND_POSE] Đã giải phóng tài nguyên MediaPipe.")

# ==========================================
# CẤU HÌNH VẼ KHUNG XƯƠNG
# ==========================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         
    (0, 5), (5, 6), (6, 7), (7, 8),         
    (5, 9), (9, 10), (10, 11), (11, 12),    
    (9, 13), (13, 14), (14, 15), (15, 16),  
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

def draw_hand_skeleton_from_array(frame, kp_array, w, h):
    """Vẽ khung xương tay từ mảng 1D 63 phần tử"""
    points = kp_array.reshape(21, 3)
    pixel_points = []
    for p in points:
        cx, cy = int(p[0] * w), int(p[1] * h)
        pixel_points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) 

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection[0], connection[1]
        if start_idx < len(pixel_points) and end_idx < len(pixel_points):
            pt1 = pixel_points[start_idx]
            pt2 = pixel_points[end_idx]
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


# ==========================================

# --- CẤU HÌNH THU THẬP ---
SAVE_PATH = "None_dataset"
NUM_SAMPLES = 200
DURATION = 2.0
REST_TIME = 2.0
MISSING_VALUE = -1.0
    
MP_MODEL_PATH = "gesture_recognizer.task"
# -------------------------

def main():
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    try:
        extractor = HandExtractor(mp_model=MP_MODEL_PATH, max_hands=2)
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        print("Vui lòng kiểm tra lại đường dẫn file .task của MediaPipe!")
        return

    cap = cv2.VideoCapture(1)
    
    print(f"[DATA_COLLECTOR] Chuẩn bị thu thập {NUM_SAMPLES} mẫu...")
    time.sleep(5)

    for sample_idx in range(105, NUM_SAMPLES):
        data_points = []
        print(f"\n>>> ĐANG QUAY MẪU {sample_idx + 1}/{NUM_SAMPLES}...")
        
        start_time = time.time()
        
        # --- PHASE 1: QUAY 2 GIÂY ---
        while (time.time() - start_time) < DURATION:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1) 
            timestamp_ms = int(time.time() * 1000)

            extractor.process_frame_async(frame, timestamp_ms)
            
            hands_data, gestures, mp_frame, fps = extractor.get_latest_results()
            
            frame_coords = np.full(126, MISSING_VALUE)
            
            display_frame = mp_frame.copy() if mp_frame is not None else frame.copy()
            h, w = display_frame.shape[:2]

            if hands_data:
                num_hands = min(len(hands_data), 2) 
                for i in range(num_hands):
                    start_idx = i * 63
                    end_idx = start_idx + 63
                    # Gán tọa độ vào mảng tổng
                    frame_coords[start_idx:end_idx] = hands_data[i]
                    
                    # Vẽ khung xương lên display_frame
                    draw_hand_skeleton_from_array(display_frame, hands_data[i], w, h)
            
            # Lưu dòng dữ liệu của frame hiện tại
            data_points.append(frame_coords)
            
            # Hiển thị UI
            cv2.putText(display_frame, f"REC: SAMPLE {sample_idx+1}/{NUM_SAMPLES}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Hands: {len(hands_data)} | FPS: {fps}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow("Data Collector", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                extractor.cleanup()
                cap.release()
                cv2.destroyAllWindows()
                return

        # Lưu file CSV
        file_name = os.path.join(SAVE_PATH, f"sample_{sample_idx+1}.csv")
        np.savetxt(file_name, np.array(data_points), delimiter=",")
        print(f"Đã lưu {file_name} (Shape: {np.array(data_points).shape})")

        # --- PHASE 2: NGHỈ 2 GIÂY ---
        rest_start = time.time()
        while (time.time() - rest_start) < REST_TIME:
            success, frame = cap.read()
            if not success: break
            
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "RESTING... PREPARE NEXT", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Data Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                extractor.cleanup()
                cap.release()
                cv2.destroyAllWindows()
                return

    extractor.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[DATA_COLLECTOR] HOÀN THÀNH THU THẬP DATASET!")

if __name__ == "__main__":
    main()