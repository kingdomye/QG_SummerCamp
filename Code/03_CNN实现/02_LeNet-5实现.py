# ================================
# @File         : 02_LeNet-5实现.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : 利用pytorch构建Lenet-5卷积网络模型
#               Lenet-5模型在手写体识别上具有很大优势
#               MNIST数据集在下载时出现问题，暂时无法测试
# ================================

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms


'''
模型网络结构：
输入C1 -> 卷积S2 -> 池化C3 -> 卷积S4 -> 池化C5 -> 卷积F6（全连接） -> 全连接 -> 输出
'''
class LeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 卷积层C1 池化层S2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # 卷积层C3 池化层S4
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # 全连接层C5 F6
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120), 
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84), 
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


# 加载MNIST手写数字数据集
# train_data = torchvision.datasets.MNIST(
#     root='./data', 
#     train=True, 
#     transform=transforms.ToTensor(), 
#     download=True
# )

# test_data = torchvision.datasets.MNIST(
#     root='./data', 
#     train=False, 
#     transform=transforms.ToTensor(), 
#     download=True
# )

if __name__ == '__main__':
    net = LeNet()
    print(net)
