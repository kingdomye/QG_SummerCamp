# ================================
# @File         : LSTM实现.py
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
    train_X, train_y = dataset
    train_X = train_X.permute(1, 0, 2).to(device)
    train_y = train_y.permute(1, 0, 2).to(device)

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
    hidden_size = 32            # 隐藏层节点个数
    output_size = 1             # 输出数据维度
    num_layers = 1              # 模型层数

    learning_rate = 1e-3
    epoches = 1000
    lstm = Lstm(input_size, hidden_size, output_size, num_layers).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    # 数据生成
    nums_data = 100
    sequence_len = 10
    start = np.random.randint(0, 10)
    x = np.linspace(start, start+4*np.pi, nums_data)
    y = np.sin(x)
    plt.scatter(x, y, c='blue', label='True Values')

    X, y_data = [], []
    for i in range(nums_data - sequence_len - 1):
        X.append(y[i: i+sequence_len])
        y_data.append(y[i+1: i+sequence_len+1])
    
    X = torch.tensor(np.array(X), dtype=torch.float).unsqueeze(2)
    y_data = torch.tensor(np.array(y_data), dtype=torch.float).unsqueeze(2)
    dataset = (X, y_data)

    # 模型训练
    train(dataset, lstm, loss_fn, optimizer, epoches)

    # 预测
    input_seq = torch.tensor(y[:sequence_len], dtype=torch.float).unsqueeze(1).unsqueeze(1).to(device)
    predictions = []

    lstm.eval()
    with torch.no_grad():
        for _ in range(nums_data - sequence_len):
            pred = lstm(input_seq)
            last_pred = pred[-1].unsqueeze(0)
            predictions.append(last_pred.cpu().item())
            input_seq = torch.cat((input_seq[1:], last_pred), dim=0)

    predictions = [y[i] for i in range(sequence_len)] + predictions

    plt.scatter(x, predictions, c='red', label='Predictions')
    plt.legend()
    plt.show()
