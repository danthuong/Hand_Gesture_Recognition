import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random

# ==========================================
# SETTING SEED CHO REPRODUCIBILITY
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from handlers.gru import MotionGRU

base_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(os.path.dirname(base_dir), "models", "motion_model.pth")

# ==========================================
# DATASET VÀ DATA AUGMENTATION 
# ==========================================
class HandDataset(Dataset):
    def __init__(self, x_path, y_path, augment=True):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = self.x[idx].copy()
        label = self.y[idx]

        if self.augment:
            # 1. Jittering (Thêm nhiễu nhẹ) - Chỉ cộng vào tọa độ hợp lệ
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.005, data.shape)
                valid_mask = data != -1.0
                data[valid_mask] += noise[valid_mask]

            # 2. Scaling 
            if np.random.random() > 0.5:
                scale_factor = np.random.uniform(0.85, 1.15)
                valid_mask = data != -1.0
                data[valid_mask] *= scale_factor

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==========================================
# TRAINING SETTINGS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dữ liệu
full_dataset = HandDataset("X_data.npy", "y_labels.npy", augment=True)
train_size = int(0.8 * len(full_dataset)) # 20% Validation
test_size = len(full_dataset) - train_size
train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

model = MotionGRU().to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Tự động giảm Learning Rate nếu bị local minima
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# ==========================================
# TRAINING LOOP
# ==========================================
EPOCHS = 50
best_val_loss = float('inf')

print("\nTRAINING...")
for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Tính độ chính xác
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total
    
    # Cập nhật Scheduler
    scheduler.step(avg_val_loss)

    # In log mỗi 5 epoch
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    # --- LƯU MODEL TỐT NHẤT ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        if (epoch + 1) > 10: # Không in rác ở mấy epoch đầu
            print(f"  -> Saved best model (Val Loss {best_val_loss:.4f})")

print(f"\nDONE TRAINING, BEST MODEL AT {SAVE_PATH}.")

# ==========================================
# EXPORT DATA FOR POWER BI
# ==========================================
print("\nĐang xuất dữ liệu cho Power BI (Dựa trên tập Test vừa chia)...")
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = os.path.dirname(SAVE_PATH)
LABEL_MAP_INV = {0: 'none', 1: 'shake', 2: 'clap'}

# A. Data Balance (Tính trên toàn tập gốc để biễu diễn thực tế số lượng mẫu)
y_all = np.load("y_labels.npy")
unique_labels, counts = np.unique(y_all, return_counts=True)
balance_data = [{'Label': LABEL_MAP_INV[l], 'Count': c} for l, c in zip(unique_labels, counts)]
pd.DataFrame(balance_data).to_csv(os.path.join(MODELS_DIR, 'powerbi_gru_data_balance.csv'), index=False)

# Chạy inference trên tập TEST vừa chia
model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
model.eval()

y_test_true = []
y_test_pred = []

with torch.no_grad():
    # Sử dụng nguyên xi cái test_loader (đại diện cho tập Test) lúc nãy
    # Đã có shuffle=False rồi nên duyệt theo thứ tự
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        y_test_true.extend(labels.numpy())
        y_test_pred.extend(preds.cpu().numpy())

y_true_text = [LABEL_MAP_INV[l] for l in y_test_true]
y_pred_text = [LABEL_MAP_INV[l] for l in y_test_pred]

# B. Metrics
report = classification_report(y_true_text, y_pred_text, output_dict=True, zero_division=0)
metrics_data = []
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        metrics_data.append({
            'Class': label,
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
pd.DataFrame(metrics_data).to_csv(os.path.join(MODELS_DIR, 'powerbi_gru_model_metrics.csv'), index=False)

# C. Confusion Matrix
labels_list = list(LABEL_MAP_INV.values())
cm = confusion_matrix(y_true_text, y_pred_text, labels=labels_list)
cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
cm_df.index.name = 'True_Label'
cm_df.reset_index(inplace=True)

cm_df_melted = cm_df.melt(id_vars='True_Label', var_name='Predicted_Label', value_name='Count')
cm_df_melted.to_csv(os.path.join(MODELS_DIR, 'powerbi_gru_confusion_matrix.csv'), index=False)

print("-> Hoàn tất xuất GRU Metrics & Confusion Matrix vào thư mục models/")