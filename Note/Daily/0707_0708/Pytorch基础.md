# Pytorch基础

## torch tensor

张量（Tensor）是一种特殊的数据结构，它与数组和矩阵非常相似。在深度学习框架 PyTorch 中，张量是核心的数据表示形式，我们使用张量来编码模型的输入和输出，同时也用它来存储模型的参数。下面将介绍几种常见的创建 PyTorch 张量的方法，若你想了解更多方法，可以查询 PyTorch 官网的 API 文档。

### 直接从数据创建张量
这种方法非常直观，我们可以将 Python 列表或嵌套列表直接转换为 PyTorch 张量。以下代码展示了如何从一个包含日期和时间信息的二维列表创建张量：
```python
import torch

# 定义一个二维列表数据
data = [[2025, 7, 7], [9, 6, 45]]
# 使用 torch.tensor 函数将数据转换为张量
x_data = torch.tensor(data)
# 打印生成的张量
print(x_data)
```
在上述代码中，我们首先导入了 `torch` 库，然后定义了一个二维列表 `data`，接着使用 `torch.tensor` 函数将其转换为张量 `x_data`，最后打印出这个张量。

### 从 numpy 数组创建张量
由于在数据处理中，我们经常会使用 NumPy 库进行数据的处理和操作，因此将 NumPy 数组转换为 PyTorch 张量是很常见的需求。以下代码展示了如何将 NumPy 数组转换为 PyTorch 张量：
```python
import numpy as np
import torch

# 定义一个二维列表数据
data = [[2025, 7, 7], [9, 6, 45]]
# 将列表数据转换为 NumPy 数组
np_array = np.array(data)
# 使用 torch.tensor 函数将 NumPy 数组转换为张量
x_np = torch.tensor(np_array)
# 打印生成的张量
print(x_np)
```
在这段代码中，我们先导入了 `numpy` 和 `torch` 库，然后将列表 `data` 转换为 NumPy 数组 `np_array`，接着使用 `torch.tensor` 函数将其转换为 PyTorch 张量 `x_np`，最后打印该张量。

### 其他函数创建张量
除了上述两种方法，PyTorch 还提供了一些函数来创建具有特定属性的张量，例如创建全为 1 的张量或随机值的张量。以下代码展示了如何使用这些函数：
```python
import torch

# 定义一个二维列表数据
data = [[2025, 7, 7], [9, 6, 45]]
# 将数据转换为张量
x_data = torch.tensor(data)

# 创建一个与 x_data 形状相同且元素全为 1 的张量
x_ones = torch.ones_like(x_data)
print(x_ones)

# 创建一个与 x_data 形状相同且元素为随机浮点数的张量
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

# 创建一个形状为 (3, 3) 的随机值张量
rand_tensor = torch.rand((3, 3, ))
# 创建一个形状为 (2, 3) 的全为 1 的张量
ones_tensor = torch.ones((2, 3, ))
print(rand_tensor, ones_tensor)
```
在这段代码中，我们使用了 `torch.ones_like` 函数创建了一个与 `x_data` 形状相同且元素全为 1 的张量 `x_ones`，使用 `torch.rand_like` 函数创建了一个与 `x_data` 形状相同且元素为随机浮点数的张量 `x_rand`，还使用 `torch.rand` 和 `torch.ones` 函数分别创建了指定形状的随机值张量和全为 1 的张量。

### tensor 属性
张量具有各种属性，在开发过程中，我们可以直接访问这些属性来获取张量的相关信息，例如形状和数据类型。以下代码展示了如何访问张量的形状和数据类型属性：
```python
import torch
import numpy as np

# 定义一个二维列表数据
data = [[2025, 7, 7], [9, 6, 45]]
# 将数据转换为张量
x_data = torch.tensor(data)
# 将数据转换为 NumPy 数组，再转换为张量
np_array = np.array(data)
x_np = torch.tensor(np_array)

# 打印张量的形状
print(f"tensor shape: {x_data.shape}")
# 打印张量的数据类型
print(f"tensor dtype: {x_np.dtype}")
```
在上述代码中，我们定义了两个张量 `x_data` 和 `x_np`，然后使用 `shape` 属性获取张量的形状，使用 `dtype` 属性获取张量的数据类型，并将这些信息打印出来。

## Datasets and Dataloader

在处理数据样本时，代码可能会变得杂乱且难以维护。为了提高代码的可读性和模块化，我们希望将数据集代码与模型训练代码解耦。PyTorch 提供了两种数据原语：`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`，它们允许我们使用预加载的数据集以及自己的数据。`Dataset` 用于存储样本及其对应的标签，而 `DataLoader` 则在 `Dataset` 周围封装了一个迭代器，方便我们访问样本。

下面我们将使用 PyTorch 的数据集来显示 FashionMNIST 数据集的图像。首先，我们需要加载数据集：
```python
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 定义标签映射，用于将数字标签转换为对应的类别名称
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

# 加载训练数据集
training_data = datasets.FashionMNIST(
    root="data",  # 数据集存储的根目录
    train=True,   # 是否加载训练集
    download=False,  # 是否下载数据集，如果已经下载则设为 False
    transform=ToTensor()  # 数据预处理，将图像转换为张量
)

# 加载测试数据集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)
```
在上述代码中，我们首先导入了必要的库，然后定义了一个标签映射 `labels_map`，用于将数字标签转换为对应的类别名称。接着，我们使用 `datasets.FashionMNIST` 函数分别加载了训练数据集和测试数据集，并指定了数据集的存储路径、是否为训练集、是否下载以及数据预处理方式。

### 分别使用 datasets 和 dataloader 获取样本并且显示图像

#### 利用 datasets 遍历数据集
我们可以直接通过索引访问 `Dataset` 中的样本，以下代码展示了如何随机选择一个样本并显示其图像：
```python
import torch
import matplotlib.pyplot as plt

# 随机生成一个索引
random_sample_index = torch.randint(len(training_data), size=(1, 1)).item()
# 根据索引获取随机样本
random_sample = training_data[random_sample_index]
# 从样本中分离出图像和标签
img, label = random_sample
# 设置图像的标题为对应的类别名称
plt.title(labels_map[label])
# 显示图像，使用 squeeze 函数去除维度为 1 的维度
plt.imshow(img.squeeze())
# 显示图像
plt.show()
```
在这段代码中，我们使用 `torch.randint` 函数随机生成一个索引，然后根据这个索引从 `training_data` 中获取一个随机样本。接着，我们从样本中分离出图像和标签，使用 `labels_map` 将标签转换为类别名称并设置为图像的标题，最后使用 `plt.imshow` 函数显示图像。

#### 利用 DataLoader 遍历数据集
`DataLoader` 提供了一种更方便的方式来批量加载数据，以下代码展示了如何使用 `DataLoader` 遍历数据集并显示其中的图像：
```python
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 创建训练数据加载器，指定批量大小为 64，并打乱数据顺序
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# 创建测试数据加载器，指定批量大小为 64，并打乱数据顺序
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 获取训练数据加载器的第一批数据
train_img, train_label = next(iter(train_dataloader))
# 显示第 60 张图像，使用 squeeze 函数去除维度为 1 的维度
plt.imshow(train_img[60].squeeze())
# 显示图像
plt.show()
```
在这段代码中，我们使用 `DataLoader` 分别创建了训练数据加载器和测试数据加载器，并指定了批量大小和是否打乱数据顺序。然后，我们使用 `next(iter(train_dataloader))` 获取训练数据加载器的第一批数据，最后选择第 60 张图像并显示出来。

## Build a basic neural network with PyTorch

接下来，我们将使用 PyTorch 构建一个基本的神经网络模型，并在 FashionMNIST 数据集上进行训练和测试。

```python
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
# 检查 MPS （苹果 Metal Performance Shaders）是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
# 检查 CUDA （NVIDIA GPU 加速）是否可用
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

# 创建训练数据加载器，指定批量大小为 64
train_dataloader = DataLoader(training_data, batch_size=64)
# 创建测试数据加载器，指定批量大小为 64
test_dataloader = DataLoader(test_data, batch_size=64)

# 构建神经网络
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义一个 Flatten 层，用于将输入的图像展平为一维向量
        self.flatten = nn.Flatten()
        # 定义一个 Sequential 容器，包含多个线性层和 ReLU 激活函数
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512), nn.ReLU(), 
            nn.Linear(512, 512), nn.ReLU(), 
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        # 将输入的图像展平为一维向量
        x = self.flatten(x)
        # 通过线性层和激活函数进行前向传播
        output = self.linear(x)
        
        return output
    
# 模型训练
def train(dataloader, model, loss_fn, optimizer):
    # 获取数据集的大小
    size = len(dataloader.dataset)
    # 将模型设置为训练模式
    model.train()
    # 遍历数据加载器中的每个批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播，计算模型的预测值
        pred = model(X)
        # 计算损失值
        loss = loss_fn(pred, y)

        # 反向传播，计算梯度
        loss.backward()
        # 根据反向传播得到的梯度调整模型的参数
        optimizer.step()
        # 重置模型的梯度，防止重复计算
        optimizer.zero_grad()

        # 每 100 个批次打印一次损失值和当前进度
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 测试模型
def test(dataloader, model, loss_fn):
    # 将模型设置为评估模式
    model.eval()
    # 获取数据集的大小
    size = len(dataloader.dataset)
    # 获取数据加载器中的批次数量
    num_batches = len(dataloader)
    # 初始化测试损失和正确预测的数量
    test_loss, correct = 0, 0

    # 测试时使用 no_grad() 可以提高计算效率
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for X, y in dataloader:
            # 前向传播，计算模型的预测值
            pred = model(X)
            # 累加测试损失
            test_loss += loss_fn(pred, y).item()
            # 计算正确预测的数量
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
        # 计算平均测试损失
        test_loss /= num_batches
        # 计算准确率
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

if __name__ == '__main__':
    # 创建神经网络模型，并将其移动到指定设备上
    model = Network().to(device='cpu')
    # 定义学习率
    learning_rate = 1e-3
    # 定义批量大小
    batch_size = 64
    # 定义训练的轮数
    epochs = 10

    # 定义损失函数，使用交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器，使用随机梯度下降优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 开始训练模型
    for epoch in range(epochs):
        print(f"--------------Epoch {epoch}--------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    # 保存模型
    torch.save(model, 'model.pth')
    # 加载模型使用 torch.load() 方法

```
在上述代码中，我们首先导入了必要的库，然后根据设备的可用性选择了合适的训练设备。接着，我们加载了 FashionMNIST 数据集，并创建了数据加载器。之后，我们定义了一个简单的神经网络模型 `Network`，包含一个 `Flatten` 层和多个线性层及 `ReLU` 激活函数。我们还定义了训练函数 `train` 和测试函数 `test`，用于训练和评估模型。最后，我们设置了学习率、批量大小和训练轮数，定义了损失函数和优化器，开始训练模型，并在每个 epoch 结束后进行测试。训练完成后，我们使用 `torch.save` 函数保存了模型。