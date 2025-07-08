# ================================
# @File         : 03_AlexNet实现.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : 利用pytorch构建AlexNet卷积网络模型
#                 是首个真正意义上的深度卷积神经网络
#                 在视觉识别领域有较强的优势
#                 数据集位置：../01_Pytorch/data
# ================================

import torch
import numpy as np
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

# 检查MPS可用性
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # (1, 227, 227) -> (96, 55, 55)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=96, 
                kernel_size=11, 
                stride=4
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(3, 2)
        )

        # (96, 55, 55) -> (256, 27, 27)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# 数据预处理, 图像大小改为227*227*3
net_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=0.5, std=0.5)
])

# 加载训练数据
training_data = datasets.FashionMNIST(
    root="../01_Pytorch/data",
    train=True,
    download=True,
    transform=net_transform
)

test_data = datasets.FashionMNIST(
    root="../01_Pytorch/data",
    train=False,
    download=True,
    transform=net_transform
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 将数据移动到设备上
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':
    net = AlexNet()
    # 将模型移动到设备上
    net = net.to(device)
    learning_rate = 1e-3
    batch_size = 64
    epochs = 3

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # from matplotlib import pyplot as plt
    # data_sample = next(iter(train_dataloader))
    # plt.imshow(np.squeeze(data_sample[0][0]), cmap="gray")
    # plt.show()

    for epoch in range(epochs):
        print(f"--------------Epoch {epoch}--------------")
        train(train_dataloader, net, loss_fn, optimizer)