# ================================
# @File         : 01_基础CNN实现.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : 利用pytorch构建CNN卷积网络模型
#                 数据集位置：../01_Pytorch/data
# ================================

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
    root="../01_Pytorch/data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="../01_Pytorch/data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 构建CNN神经网络
class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # (1, 28, 28) -> (16, 28, 28) -> (16, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              # 输入通道数量
                out_channels=16,            # 输出通道数量
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 卷积核移动步长
                padding=2                   # 填充大小
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        # (16, 14, 14) -> (32, 14, 14) -> (32, 7, 7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(32*7*7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    model = ConvNetwork().to(device)
    learning_rate = 1e-3
    batch_size = 64
    epochs = 3

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"--------------Epoch {epoch}--------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    torch.save(model, 'conv_model.pth')
