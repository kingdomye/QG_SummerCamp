# ================================
# @File         : 05_ResNet实现.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : 利用pytorch内置ResNet模型实现图像识别
#               残差神经网络能够解决梯度爆炸的问题
#               由于测试的数据集(CIFAR)太大，没有上传至github
# ================================


import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils import data
import torchvision
from torchvision import models, transforms

device = torch.device('cpu')

# 模型预处理
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.2, 0.2, 0.2))
])

# 数据集加载
train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=trans, download=False)
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=trans, download=False)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

batch_size = 128
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

data_loaders = {
    "train": train_dataloader,
    "test": test_dataloader
}

# 加载本地权重文件
net = models.resnet18(weights=None)
state_dict = torch.load('./resnet18-5c106cde.pth')
net.load_state_dict(state_dict)
net.fc = nn.Linear(512, 10)
# print(net)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# 模型训练
def train_model(net, dataloaders, criterion, optimizer, num_epochs):
    net.to(device)
    
    # 网络加速
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        for phase in ["train", "test"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            
            # 损失
            epoch_loss = 0.0
            # 正确答案的数量
            epoch_corrects = 0
            
            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                # 设置梯度计算开启或关闭
                # 只在训练时开启梯度计算
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    # 计算损失
                    loss = criterion(outputs, labels)
                    # 预测标签
                    # 返回每一行的最大值，也就是所属的类别
                    _, preds = torch.max(outputs, 1)
                    
                    # 反向传播
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                    # loss的总和
                    # loss计算的是平均值，所以要乘上batchsize，计算损失的总和
                    epoch_loss += loss.item() * inputs.size(0)
                    # 预测正确的答案的数量
                    epoch_corrects += torch.sum(preds == labels.data)
                    
                    # 每个epoch的loss和正确率
                    epoch_loss = epoch_loss / len(data_loaders[phase].dataset)
                    epoch_acc = epoch_corrects.double() / len(data_loaders[phase].dataset)
                    
                    epoch_loss = torch.tensor(epoch_loss)
                    epoch_acc = torch.tensor(epoch_acc)
                    print(f"epoch: {epoch + 1}/{num_epochs}; {phase} loss:{np.round(epoch_loss.item(), 5)}; acc:{np.round(epoch_acc.item() * 100, 2)}%")
            # writer.add_scalar(f"{phase} loss", epoch_loss, epoch)
            # writer.add_scalar(f"{phase} accuracy", epoch_acc, epoch)
        
            # 保存模型
            # if (epoch % 20 == 0 and phase == 'train'):
            #     save_path = f"./models/vgg16_{epoch}_{epoch_acc}.pth"
            #     torch.save(net.state_dict(), save_path)      

if __name__ == '__main__':
    train_model(net, data_loaders, criterion, optimizer, num_epochs=1)