import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

CSV_PATH = os.path.join(MODELS_DIR, 'hand_gestures.csv')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'gesture_model.pkl')

# --- 1. Đọc dữ liệu ---
if not os.path.exists(CSV_PATH):
    print(f"Không tìm thấy file {CSV_PATH}")
    exit()

df = pd.read_csv(CSV_PATH)
X = df.drop('label', axis=1) 
y = df['label']              

# --- 2. Mã hóa nhãn (Chuyển chữ thành số) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 3. Chia tập dữ liệu ---
# Lưu ý: truyền y_encoded vào đây thay vì y
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- 4. Huấn luyện mô hình XGBoost ---
print("Đang huấn luyện mô hình XGBoost...")
model = XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.1,
    objective='multi:softprob', 
    random_state=42,
    tree_method='hist' # Bỏ comment nếu máy có card NVIDIA và đã cài đặt hỗ trợ, thay gpu_hist bang hist vi may khong co ho tro GPU XGBoost tuong thich
)

# Bây giờ y_train chính là các nhãn đã được mã hóa số
model.fit(X_train, y_train)

# --- 5. Kiểm tra độ chính xác ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác đạt được: {acc * 100:.2f}%")

# --- 6. Xuất mô hình VÀ bộ giải mã nhãn ---
# Chúng ta lưu dưới dạng tuple (model, label_encoder)
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump((model, label_encoder), f)

print(f"Đã lưu 'bộ não' và 'bộ giải mã' vào: {MODEL_SAVE_PATH}")


# --- 7. Xuất dữ liệu cho Power BI ---
print("\nĐang xuất dữ liệu cho Power BI...")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# A. Bảng Data Balance
balance_df = y.value_counts().reset_index()
balance_df.columns = ['Label', 'Count']
powerbi_balance_path = os.path.join(MODELS_DIR, 'powerbi_data_balance.csv')
balance_df.to_csv(powerbi_balance_path, index=False)
print(f"-> Đã xuất Data Balance: {powerbi_balance_path}")

# B. Bảng dự đoán (Predictions)
# Giải mã lại nhãn dạng chuỗi từ dự đoán dạng số
true_labels_text = label_encoder.inverse_transform(y_test)
pred_labels_text = label_encoder.inverse_transform(y_pred)

# Lấy xác suất dự đoán lớn nhất (độ tự tin)
y_pred_proba = model.predict_proba(X_test)
confidence = np.max(y_pred_proba, axis=1)

predictions_df = pd.DataFrame({
    'True_Label': true_labels_text,
    'Predicted_Label': pred_labels_text,
    'Is_Correct': true_labels_text == pred_labels_text,
    'Confidence': confidence
})

powerbi_predictions_path = os.path.join(MODELS_DIR, 'powerbi_model_predictions.csv')
predictions_df.to_csv(powerbi_predictions_path, index=False)
print(f"-> Đã xuất bảng dự đoán: {powerbi_predictions_path}")

# C. Bảng Metrics (Accuracy, Precision, Recall, F1)
report = classification_report(y_test, y_pred, output_dict=True)
metrics_data = []
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        label_text = label_encoder.inverse_transform([int(label)])[0] if str(label).isdigit() else label
        metrics_data.append({
            'Class': label_text,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1-score'],
            'Support': metrics['support']
        })
metrics_data.append({
    'Class': 'Macro Average',
    'Precision': report['macro avg']['precision'],
    'Recall': report['macro avg']['recall'],
    'F1_Score': report['macro avg']['f1-score'],
    'Support': report['macro avg']['support']
})
metrics_df = pd.DataFrame(metrics_data)
powerbi_metrics_path = os.path.join(MODELS_DIR, 'powerbi_model_metrics.csv')
metrics_df.to_csv(powerbi_metrics_path, index=False)
print(f"-> Đã xuất bảng Metrics: {powerbi_metrics_path}")

# D. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
cm_df.index.name = 'True_Label'
cm_df.reset_index(inplace=True)
cm_df_melted = cm_df.melt(id_vars='True_Label', var_name='Predicted_Label', value_name='Count')
powerbi_cm_path = os.path.join(MODELS_DIR, 'powerbi_confusion_matrix.csv')
cm_df_melted.to_csv(powerbi_cm_path, index=False)
print(f"-> Đã xuất Confusion Matrix: {powerbi_cm_path}")

# E. Bảng tầm quan trọng của đặc trưng (Feature Importances)
feature_importances = model.feature_importances_
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

powerbi_features_path = os.path.join(MODELS_DIR, 'powerbi_feature_importances.csv')
features_df.to_csv(powerbi_features_path, index=False)
print(f"-> Đã xuất bảng độ quan trọng đặc trưng: {powerbi_features_path}")
