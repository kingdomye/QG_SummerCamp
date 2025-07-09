# ================================
# @File         : LSTM实现(Error Version).py
# @Time         : 2025/07/09
# @Author       : Yingrui Chen
# @description  : 构建基础的LSTM模型
# ================================

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers
        )

        self.fc = nn.Linear(
            hidden_size, 
            output_size
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        seq_len, batch, hidden_size = x.shape
        x = x.view(seq_len*batch, hidden_size)
        x = self.fc(x)
        x = x.view(seq_len, batch, -1)

        return x
    

def train(dataset, model, loss_fn, optimizer, epoches):
    train_X, train_y = dataset[:, 0], dataset[:, 1]
    train_X = train_X.view(-1, 50, input_size).to(device)
    train_y = train_y.view(-1, 50, output_size).to(device)

    # print(train_X)
    # print(train_y)

    for epoch in range(epoches):
        pred = model(train_X)

        loss = loss_fn(pred, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1} Loss {loss}')


if __name__ == '__main__':
    # 模型参数
    input_size = 1              # 输入变量的维度
    hidden_size = 16            # 隐藏层节点个数
    output_size = 1             # 输出数据维度
    num_layers = 1              # 模型层数

    learning_rate = 1e-3
    epoches = 100
    lstm = Lstm(input_size, hidden_size, output_size, num_layers).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # 数据生成
    nums_data = 200
    start = np.random.randint(0, 10)
    x = np.linspace(start, start+4*np.pi, nums_data)
    y = np.sin(x)
    plt.scatter(x, y, cmap='blue')

    dataset = np.zeros((nums_data, 2))
    dataset[:, 0], dataset[:, 1] = x, y
    dataset = torch.tensor(dataset, dtype=torch.float).to(device)

    # 模型训练
    train(dataset, lstm, loss_fn, optimizer, epoches)

    # 预测
    start = np.random.randint(-10, 10)
    x = np.linspace(start, start+4*np.pi, nums_data)
    y = np.sin(x)
    # plt.scatter(x, y, cmap='blue')

    x_tensor = torch.tensor(x, dtype=torch.float).to(device)
    x_tensor = x_tensor.view(-1, 10, input_size)
    lstm.eval()
    with torch.no_grad():
        pred = lstm(x_tensor)
        pred = pred.cpu().detach().numpy().ravel()

    plt.scatter(x, pred, cmap='red')
    plt.show()