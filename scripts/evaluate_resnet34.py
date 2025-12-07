import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn

# ----------------------------
# 路径（用你的绝对路径）
# ----------------------------
test_path = "/Users/Jerry/Desktop/ISE/251104hw/data/SO32_preproc/test_resized"
model_path = "/Users/Jerry/Desktop/ISE/251104hw/models/resnet34_aug_epoch10.pt"

# ----------------------------
# 数据预处理
# ----------------------------
test_tf = T.Compose([
    T.ToTensor()
])

test_dataset = torchvision.datasets.ImageFolder(test_path, transform=test_tf)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------------------
# 加载模型
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet34(weights=None)
model.fc = nn.Linear(512, len(test_dataset.classes))  # 替换分类头
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# 测试集评估
# ----------------------------
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"Test Accuracy: {acc:.2f}%")
