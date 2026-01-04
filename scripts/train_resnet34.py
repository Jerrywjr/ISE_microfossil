import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

# ----------------------------
# 绝对路径设置
# ----------------------------
BASE_DIR = "/Users/Jerry/Desktop/ISE/251104hw"
TRAIN_PATH = os.path.join(BASE_DIR, "data/SO32_preproc/train_resized")
VAL_PATH   = os.path.join(BASE_DIR, "data/SO32_preproc/val_resized")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "runs")

# 确保保存目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------
# TensorBoard
# ----------------------------
writer = SummaryWriter(LOG_DIR)

# ----------------------------
# 数据增强
# ----------------------------
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ToTensor()
])

test_tf = T.Compose([
    T.ToTensor()
])

# ----------------------------
# 数据集
# ----------------------------
if not os.path.isdir(TRAIN_PATH):
    raise FileNotFoundError(f"训练集路径不存在: {TRAIN_PATH}")
if not os.path.isdir(VAL_PATH):
    raise FileNotFoundError(f"验证集路径不存在: {VAL_PATH}")

train_dataset = torchvision.datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
val_dataset   = torchvision.datasets.ImageFolder(VAL_PATH, transform=test_tf)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------------------------
# 模型
# ----------------------------
model = torchvision.models.resnet34(weights=None, num_classes=len(train_dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------------
# 损失函数 & 优化器
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# 训练循环
# ----------------------------
EPOCHS = 10  # 你可以根据需要修改
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {epoch_loss:.4f}")
    writer.add_scalar("Loss/train", epoch_loss, epoch)

    # ----------------------------
    # 每个 epoch 自动保存模型
    # ----------------------------
    model_path = os.path.join(MODEL_DIR, f"resnet34_aug_epoch{epoch}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# ----------------------------
# 验证集评估
# ----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
writer.add_scalar("Accuracy/val", accuracy, EPOCHS)

writer.close()
