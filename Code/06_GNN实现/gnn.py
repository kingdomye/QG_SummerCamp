# ================================
# @File         : gnn.py
# @Time         : 2025/07/10
# @Author       : Yingrui Chen
# @description  : 简单GNN模型实现节点分类
#                 Cora数据集是PyG内置的节点分类数据集，
#                 代表着学术论文的相关性分类问题（即把
#                 每一篇学术论文都看成是节点），Cora数
#                 据集有2708个节点，1433维特征，边数为
#                 5429。标签是文献的主题，共计7个类别
# ================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset = Planetoid(
    root='./data', 
    name='Cora'
)
data = dataset[0]

class gnn(nn.Module):
    def __init__(self, num_features, num_hidden_layers, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes

        self.conv1 = GCNConv(self.num_features, self.num_hidden_layers)
        self.conv2 = GCNConv(self.num_hidden_layers, self.num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x
    
def train(data, model, loss_fn, optimizer):
    model.train()
    pred = model(data.x, data.edge_index)
    loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

def test(data, model):
    model.eval()
    pred = model(data.x, data.edge_index)[data.test_mask]
    max_index = torch.argmax(pred, dim=1)
    real = data.y[data.test_mask]

    correct = 0
    for i in range(len(max_index)):
        if max_index[i] == real[i]:
            correct += 1

    result = 100 * (correct / len(real))
    return result
    
if __name__ == '__main__':
    num_features = data.num_features
    num_hidden_layers = 32
    num_classes = dataset.num_classes

    gnn = gnn(num_features, num_hidden_layers, num_classes).to(device)
    data = data.to(device)

    epoches = 500
    learning_rate = 1e-3
    loss_fn = F.nll_loss
    optimizer = torch.optim.Adam(gnn.parameters(), lr=learning_rate)

    for epoch in range(200):
        loss = train(data, gnn, loss_fn, optimizer)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch}   Loss{loss}')

    correct_rate = test(data, gnn)
    print('The ACC is:', correct_rate)
