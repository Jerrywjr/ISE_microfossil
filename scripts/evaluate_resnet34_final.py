import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

# ----------------------------
# 绝对路径设置
# ----------------------------
BASE_DIR = "/Users/Jerry/Desktop/ISE/251104hw"
TEST_PATH = os.path.join(BASE_DIR, "data/SO32_preproc/test_resized")
MODEL_PATH = os.path.join(BASE_DIR, "models/baseline_vit_back.pt")

# ----------------------------
# 数据预处理
# ----------------------------
test_tf = T.Compose([
    T.ToTensor()
])

# ----------------------------
# 加载数据集
# ----------------------------
if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(f"测试集路径不存在: {TEST_PATH}")

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

# 计算整体 Accuracy
acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {acc:.2f}%")

# 每类准确率 & 混淆矩阵
class_names = test_dataset.classes
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\nClassification Report:\n")
print(report)

# 可选：保存混淆矩阵到 CSV
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(BASE_DIR, "resnet34_test_confusion_matrix.csv"))
print(f"\n混淆矩阵已保存到 {os.path.join(BASE_DIR, 'resnet34_test_confusion_matrix.csv')}")
