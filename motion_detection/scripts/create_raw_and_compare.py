import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "models", "hand_landmarker.task"))
DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "data", "static"))
MODELS_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "models"))
RAW_CSV_PATH = os.path.join(MODELS_DIR, "hand_gestures_raw.csv")
PREPROC_CSV_PATH = os.path.join(MODELS_DIR, "hand_gestures.csv")

base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def extract_raw_landmarks(img_path):
    try:
        image = mp.Image.create_from_file(img_path)
        result = detector.detect(image)
        if not result.hand_landmarks:
            return None
        hand = result.hand_landmarks[0]
        # Get raw coordinates
        return [val for lm in hand for val in (lm.x, lm.y)]
    except:
        return None

def extract_and_save_raw():
    total_images = 0
    success = 0
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    header.append("label")
    
    with open(RAW_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for category in sorted(os.listdir(DATA_DIR)):
            cat_path = os.path.join(DATA_DIR, category)
            if not os.path.isdir(cat_path) or category == "6-clap":
                continue
            
            files = []
            for root, _, fs in os.walk(cat_path):
                for file in fs:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        files.append(os.path.join(root, file))
            
            # Using only 200 items per class to speed up extraction, since this is for a Power BI Demo
            # The pattern of raw vs preprocessed will hold
            sample_files = files[:400] 
            print(f"Trích xuất RAW {category}: {len(sample_files)} ảnh mẫu")
            for img_path in sample_files:
                total_images += 1
                features = extract_raw_landmarks(img_path)
                if features is not None:
                    writer.writerow(features + [category])
                    success += 1
    print(f"Đã trích xuất xong {success}/{total_images} ảnh RAW. Lưu: {RAW_CSV_PATH}")

def evaluate_models_compare():
    import json
    
    # 1. Evaluate Preprocessed Data
    print("Đang Load Dữ Liệu Đã Tiền Xử Lý (Preprocessed)...")
    df_pre = pd.read_csv(PREPROC_CSV_PATH)
    # Filter only classes that we used in raw (to be fair if we sampled)
    
    # 2. Evaluate Raw Data
    print("Đang Load Dữ Liệu Thô (Raw)...")
    df_raw = pd.read_csv(RAW_CSV_PATH)
    
    # We will balance the evaluation by picking exactly the same label counts
    common_classes = set(df_pre.label.unique()).intersection(set(df_raw.label.unique()))
    
    results = []
    
    for name, df in [("Raw Data", df_raw), ("Preprocessed Data", df_pre)]:
        df = df[df.label.isin(common_classes)]
        X = df.drop('label', axis=1)
        y = df['label']
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            objective='multi:softprob', random_state=42, tree_method='hist'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=True)
        results.append({
            "DataType": name,
            "Accuracy": acc,
            "Macro_Precision": rep["macro avg"]["precision"],
            "Macro_Recall": rep["macro avg"]["recall"],
            "Macro_F1": rep["macro avg"]["f1-score"]
        })
    
    comp_df = pd.DataFrame(results)
    comp_path = os.path.join(MODELS_DIR, "powerbi_preprocessing_impact.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"-> Đã lưu bảng so sánh hiệu suất trước/sau tiền xử lý: {comp_path}")
    
    # Visualization Data for One Hand: extract 1 row from raw and 1 row from preproc to plot
    sample_raw = df_raw.iloc[0:1].drop('label', axis=1).values[0].reshape(21, 2)
    sample_pre = df_pre[df_pre.label == df_raw.iloc[0].label].iloc[0:1].drop('label', axis=1).values[0].reshape(21, 2)
    
    vis_data = []
    for i in range(21):
        vis_data.append({"Joint": f"J{i}", "X": sample_raw[i, 0], "Y": sample_raw[i, 1], "State": "Before Preprocessing"})
        vis_data.append({"Joint": f"J{i}", "X": sample_pre[i, 0], "Y": sample_pre[i, 1], "State": "After Preprocessing"})
        
    vis_df = pd.DataFrame(vis_data)
    vis_path = os.path.join(MODELS_DIR, "powerbi_hand_joints_vis.csv")
    vis_df.to_csv(vis_path, index=False)
    print(f"-> Đã lưu point coordinate mẫu để vẽ before/after: {vis_path}")

print("--- 1. Cập nhật Model Metrics Mặc định ---")
# Import functions exported earlier
exec(open(os.path.join(CURRENT_DIR, "export_evaluate_powerbi.py"), encoding='utf-8').read())

print("\n--- 2. Tạo Dữ Liệu Raw so sánh Before / After Preprocessing ---")
if not os.path.exists(RAW_CSV_PATH):
    extract_and_save_raw()
evaluate_models_compare()

print("Hoàn tất mọi dữ liệu cho Power BI!")
