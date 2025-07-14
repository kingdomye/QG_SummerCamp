# 卷积神经网络（CNN）学习笔记

## 一、引言
卷积神经网络（Convolutional Neural Network, CNN）在图像识别领域具有显著优势。本文将基于提供的代码，对三种经典的CNN模型（LeNet - 5、VGG和ResNet）进行学习总结。

## 二、LeNet - 5模型
### 2.1 模型概述
LeNet - 5是最早的卷积神经网络之一，由Yann LeCun等人在1998年提出，在手写体识别任务中表现出色。

### 2.2 网络结构
- **卷积层C1和池化层S2**：输入图像经过卷积操作提取特征，再通过池化层进行下采样，减少数据量。
- **卷积层C3和池化层S4**：进一步提取更高级的特征并下采样。
- **全连接层C5、F6和输出层**：将卷积和池化后的特征映射转换为一维向量，通过全连接层进行分类。

### 2.3 代码实现要点
```python
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
        # 前向传播过程
        pass
```

## 三、VGG模型
### 3.1 模型概述
VGG是由牛津大学视觉几何组（Visual Geometry Group）提出的深度卷积神经网络，在图像识别领域有很强的竞争力。

### 3.2 网络结构
- **特征提取部分**：由多个卷积层和池化层组成，通过不断增加卷积层的数量来提高模型的表达能力。
- **分类部分**：由三个全连接层组成，用于对图像进行分类。

### 3.3 代码实现要点
```python
class VGG(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            # 定义卷积层和池化层
        )
        self.classifier = nn.Sequential(
            # 定义全连接层
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        result = self.classifier(x)
        return result
```

## 四、ResNet模型
### 4.1 模型概述
ResNet（Residual Neural Network）是由微软研究院的何恺明等人在2015年提出的，通过引入残差块解决了深度神经网络中的梯度消失问题。

### 4.2 模型训练流程
1. **数据预处理**：使用`torchvision.transforms`对图像进行缩放、裁剪、归一化等操作。
2. **数据集加载**：使用`torchvision.datasets`加载CIFAR - 10数据集，并使用`torch.utils.data.DataLoader`进行批量处理。
3. **模型加载**：使用`torchvision.models`加载预训练的ResNet - 18模型，并修改最后一层全连接层的输出维度。
4. **定义损失函数和优化器**：使用交叉熵损失函数和Adam优化器。
5. **模型训练**：在训练过程中，根据训练和测试阶段的不同，设置模型的状态（训练或评估），并进行前向传播、损失计算、反向传播和参数更新。

### 4.3 代码实现要点
```python
# 模型预处理
trans = transforms.Compose([...])

# 数据集加载
train_dataset = torchvision.datasets.CIFAR10(...)
train_dataloader = data.DataLoader(train_dataset, ...)

# 加载本地权重文件
net = models.resnet18(weights=None)
net.load_state_dict(state_dict)
net.fc = nn.Linear(512, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# 模型训练
def train_model(net, dataloaders, criterion, optimizer, num_epochs):
    # 训练过程
    pass
```

## 五、总结
通过对LeNet - 5、VGG和ResNet三种模型的学习，我们了解了不同CNN模型的结构和特点。LeNet - 5结构简单，适用于手写体识别；VGG通过增加网络深度提高了模型的表达能力；ResNet通过引入残差块解决了梯度消失问题，使得可以训练更深的网络。在实际应用中，我们可以根据具体的任务和数据集选择合适的模型。