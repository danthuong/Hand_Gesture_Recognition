import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Hàm vẽ và lưu Confusion Matrix
def plot_composite_evaluation(y_true, y_pred, y_proba, classes, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # Tạo khung ảnh 1 dòng 2 cột
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)

    # --- 1. Vẽ Confusion Matrix (Bên trái) ---
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14)
    axes[0].set_ylabel('Thực tế (True Label)')
    axes[0].set_xlabel('Dự đoán (Predicted Label)')

    # --- 2. Vẽ ROC Curve (Bên phải) bằng chiến lược One-vs-Rest ---
    n_classes = len(classes)
    Y_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Vẽ đường cong cho từng class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.4f})')

    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve (One-vs-Rest)', fontsize=14)
    axes[1].legend(loc="lower right")

    # Lưu và hiển thị
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📸 Đã lưu biểu đồ kết hợp vào: {save_path}")
    plt.show()