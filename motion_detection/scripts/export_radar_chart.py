import os
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

CSV_PATH = os.path.join(MODELS_DIR, 'hand_gestures.csv')
OUTPUT_PATH = os.path.join(MODELS_DIR, 'powerbi_radar_fingers.csv')

print('Đọc dữ liệu gốc...')
df = pd.read_csv(CSV_PATH)

# Tính khoảng cách Euclidean từ Cổ tay (p0) đến 5 đầu ngón tay
def calc_distance(row, p_end):
    dx = row[f'p{p_end}_x'] - row['p0_x']
    dy = row[f'p{p_end}_y'] - row['p0_y']
    return np.sqrt(dx**2 + dy**2)

print('Tính toán khoảng cách hình học...')
# p4: Ngón cái, p8: Ngón trỏ, p12: Ngón giữa, p16: Ngón áp út, p20: Ngón út
df['Finger_Thumb'] = df.apply(lambda r: calc_distance(r, 4), axis=1)
df['Finger_Index'] = df.apply(lambda r: calc_distance(r, 8), axis=1)
df['Finger_Middle'] = df.apply(lambda r: calc_distance(r, 12), axis=1)
df['Finger_Ring'] = df.apply(lambda r: calc_distance(r, 16), axis=1)
df['Finger_Pinky'] = df.apply(lambda r: calc_distance(r, 20), axis=1)

# Gom nhóm theo nhãn và tính trung bình khoảng cách
avg_distances = df.groupby('label')[['Finger_Thumb', 'Finger_Index', 'Finger_Middle', 'Finger_Ring', 'Finger_Pinky']].mean().reset_index()

# Melt dataframe (Unpivot) để dễ nạp vào Radar Chart trong Power BI
melted_df = avg_distances.melt(id_vars='label', var_name='Finger_Name', value_name='Avg_Distance')

# Làm đẹp tên ngón tay
finger_map = {
    'Finger_Thumb': '1_Ngón Cái',
    'Finger_Index': '2_Ngón Trỏ',
    'Finger_Middle': '3_Ngón Giữa',
    'Finger_Ring': '4_Ngón Áp Út',
    'Finger_Pinky': '5_Ngón Út'
}
melted_df['Finger_Name'] = melted_df['Finger_Name'].map(finger_map)

# Lưu file kết quả
melted_df.to_csv(OUTPUT_PATH, index=False)
print(f'-> Đã lưu dữ liệu cho Radar Chart: {OUTPUT_PATH}')
