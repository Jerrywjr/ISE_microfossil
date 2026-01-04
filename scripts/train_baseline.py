import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import os

# ----------------------------
# 数据增强
# ----------------------------
train_tf = T.Compose([
    T.RandomHorizontalFlip(),  # 随机水平翻转
    T.RandomVerticalFlip(),    # 随机垂直翻转
    T.RandomRotation(20),      # 随机旋转 ±20°
    T.ToTensor()               # 转为 Tensor
])

test_tf = T.Compose([
    T.ToTensor()               # 测试集只转为 Tensor
])

# ----------------------------
# 数据集路径
# ----------------------------
train_dataset = torchvision.datasets.ImageFolder("data/SO32_preproc/train_resized", transform=train_tf)
test_dataset = torchvision.datasets.ImageFolder("data/SO32_preproc/val_resized", transform=test_tf)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------------------
# 模型
# ----------------------------
model = torchvision.models.vit_b_16(weights=None, num_classes=len(train_dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------------
# 损失与优化器
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# 训练循环
# ----------------------------
epochs = 2  # 可以根据需要改
for epoch in range(epochs):
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
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}")
    
    # 每个 epoch 自动保存模型
    os.makedirs("models", exist_ok=True)
    save_path = f"models/baseline_vit_aug_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ----------------------------
# 测试集评估
# ----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")
