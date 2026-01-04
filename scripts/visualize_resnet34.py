import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 绝对路径设置
# ----------------------------
BASE_DIR = "/Users/Jerry/Desktop/ISE/251104hw"
TEST_PATH = os.path.join(BASE_DIR, "data/SO32_preproc/test_resized")
MODEL_PATH = os.path.join(BASE_DIR, "models/resnet34_aug_epoch10.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualization")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 数据预处理
# ----------------------------
test_tf = T.Compose([T.ToTensor()])
test_dataset = torchvision.datasets.ImageFolder(TEST_PATH, transform=test_tf)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------------------
# 加载模型
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet34(weights=None)
model.fc = nn.Linear(512, len(test_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# 测试集评估
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ----------------------------
# 分类报告
# ----------------------------
class_names = test_dataset.classes
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=True)

print("Classification report saved to classification_report.csv")

# ----------------------------
# 混淆矩阵
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
df_cm.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))
print("Confusion matrix saved to confusion_matrix.csv")

# ----------------------------
# 可视化混淆矩阵 (Matplotlib + Seaborn)
# ----------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(df_cm, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ----------------------------
# 可视化混淆矩阵 (Plotly 交互式)
# ----------------------------
fig = px.imshow(df_cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="True", color="Count"),
                x=class_names, y=class_names)
fig.update_layout(title="Confusion Matrix (Interactive)")
fig.write_html(os.path.join(OUTPUT_DIR, "confusion_matrix.html"))

print("Interactive confusion matrix saved to confusion_matrix.html")

# ----------------------------
# 训练曲线示例 (可选，如果你有训练loss/accuracy记录)
# ----------------------------
# 假设你有 CSV 保存了训练loss和val accuracy
train_log_csv = os.path.join(BASE_DIR, "runs/train_log.csv")  # 可在训练脚本里生成
if os.path.exists(train_log_csv):
    df_log = pd.read_csv(train_log_csv)
    plt.figure()
    plt.plot(df_log['epoch'], df_log['train_loss'], label="Train Loss")
    plt.plot(df_log['epoch'], df_log['val_accuracy'], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
    plt.close()
    print("Training curve saved to training_curve.png")
else:
    print("No training log CSV found. Skip training curve plot.")
