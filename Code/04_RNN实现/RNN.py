# ================================
# @File         : RNN.py
# @Time         : 2025/07/08
# @Author       : Yingrui Chen
# @description  : 利用pytorch构建RNN网络模型
# ================================


import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SinDataset(Dataset):
    def __init__(self, inputs, target):
        self.inputs = inputs
        self.target = target

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.target[index]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])        # 取最后一个时间步的输出

        return out


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Step: {batch}, Loss: {loss.item()}")
    

if __name__ == '__main__':
    input_size = 1          # 输入数据的编码维度
    hidden_size = 20        # 隐含层的维度
    num_layers = 5          # 隐含层的层数

    batch_size = 1          # 序列的纵向维度
    seq_len = 12            # 输入的序列长度
    epochs = 10
    learning_rate = 0.01

    rnn = RNN(input_size, hidden_size, num_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    
    # RNN正弦函数预测
    t = np.linspace(0, 20*np.pi, 1200)
    data = np.sin(t)

    inputs, target = [], []
    for i in range(len(data) - seq_len):
        inputs.append(data[i: i+seq_len])
        target.append(data[i+seq_len])
    
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32).view(-1, seq_len, 1).to(device)
    target = torch.tensor(np.array(target), dtype=torch.float32).view(-1, 1).to(device)
    
    dataset = SinDataset(inputs, target)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

    for epoch in range(epochs):
        train(dataloader, rnn, loss_fn, optimizer)
