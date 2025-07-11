# ================================
# @File         : GNN图分类.py
# @Time         : 2025/07/11
# @Author       : Yingrui Chen
# @description  : 简单GNN模型实现图分类
#                 (tips: 这个数据集无论如何训练，准确率都不高？)
#                 ENZYMES是一个常用的图分类基准数据集。
#                 它是由600个图组成的，这些图实际上表示
#                 了不同的蛋白酶的结构，这些蛋白酶分为6
#                 个类别（每个类别有100个蛋白酶）
# ================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset = TUDataset(
    root='./data', 
    name='ENZYMES'
)
dataset = dataset.shuffle()
train_dataset = dataset[: 540]
test_dataset = dataset[540: ]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class GCN(nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_max_pool(x, batch)
        x = F.log_softmax(x, dim=1)

        return x
    

def train(model, optimizer, criterion):
    model.train()

    for data in train_loader:

        pred = model(data.x, data.edge_index, data.batch)
        loss = criterion(pred, data.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
def test(loader, model):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    
    acc = correct / len(loader.dataset)

    return acc * 100

if __name__ == '__main__':
    gnn = GCN(hidden_channels=16)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    epoches = 1000

    for epoch in range(epoches):
        train(gnn, optimizer, criterion)
        train_acc = test(train_loader, gnn)
        test_acc = test(test_loader, gnn)

        print(f'| Epoch: {epoch:03d} | Train Acc: {train_acc:.4f}% | Test Acc: {test_acc:.4f}% |')

