# ================================
# @File         : 03_构建神经网络.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : 利用pytorch构建神经网络模型
# ================================

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

# 获取用于模型训练的设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 加载训练数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 构建神经网络
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(), 
            nn.Linear(512, 512), nn.ReLU(), 
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        output = self.linear(x)
        
        return output
    
# 模型训练
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()                # 根据反向传播得到的梯度调整参数
        optimizer.zero_grad()           # 重置模型梯度，防止重复计算

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试模型
def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 测试时使用no_grad()可以提高计算效率
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

if __name__ == '__main__':
    model = Network().to(device='cpu')
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"--------------Epoch {epoch}--------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

# 保存模型
torch.save(model, 'model.pth')
# 加载模型使用torch.load()方法
