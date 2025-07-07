# ================================
# @File         : 02_torch数据集.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : torch Dataset 和 DataLoader的基本用法
# ================================

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Dataset实现随机显示一个样本图像
'''
从torch的内置数据集中下载FashionMNIST数据集
Fashion-MNIST 是一个包含 Zalando 商品图像的数据集，
由 60,000 个训练样本和 10,000 个测试样本组成。
每个样本包含一个 28×28 的灰度图像和来自 10 个类别之一的关联标签。
'''

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

# 数据集对应的标签
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

random_sample_index = torch.randint(len(training_data), size=(1, 1)).item()
random_sample = training_data[random_sample_index]
img, label = random_sample
# plt.title(labels_map[label])
# plt.imshow(img.squeeze())
# plt.show()


# 利用DataLoader遍历数据集
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_img, train_label = next(iter(train_dataloader))
plt.imshow(train_img[60].squeeze())
plt.show()
