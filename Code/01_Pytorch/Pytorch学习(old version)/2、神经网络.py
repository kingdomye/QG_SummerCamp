import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
inputs = torch.randn(1, 1, 32, 32)
# output = net(inputs)
# target = torch.randn(10)
# target = target.view(1, -1)
# criterion = nn.MSELoss()
#
# loss = criterion(output, target)
# print(loss)
#
# net.zero_grad()
# print('conv1.bias.grad before backward:', net.conv1.bias.grad)
#
# loss.backward()
# print('conv1.bias.grad after backward:', net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

net.zero_grad()
print('conv1.bias.grad before backward:', net.conv1.bias.grad)

output = net(inputs)
criterion = nn.MSELoss()
target = torch.randn(10)
target = target.view(1, -1)
loss = criterion(output, target)
loss.backward()
optimizer.step()
print('conv1.bias.grad after backward:', net.conv1.bias.grad)
