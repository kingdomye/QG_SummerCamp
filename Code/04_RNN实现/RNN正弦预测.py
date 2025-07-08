import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )

        self.linear = nn.Linear(
            in_features=hidden_size, 
            out_features=1
        )

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.view(-1, self.hidden_size)
        out = out.unsqueeze(0)
        return out
    
def train(X, y, model, loss_fn, optimizer):
    model.train()
    
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
    

if __name__ == '__main__':
    input_size = 1
    hidden_size = 16
    output_size = 1
    rnn = RNN(input_size, hidden_size).to(device)

    epoches = 1000
    lr = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr)

    for epoch in range(epoches):
        num_time_steps = 50
        start = np.random.randint(-3, 3)
        time_steps = np.linspace(start, start+10, num_time_steps)
        data = np.sin(time_steps).reshape(num_time_steps, 1)
        X = torch.tensor(data[: -1], dtype=torch.float).view(1, num_time_steps-1, 1).to(device)
        y = torch.tensor(data[1: ], dtype=torch.float).view(1, num_time_steps-1, 1).to(device)

        train(X, y, rnn, loss_fn, optimizer)

    start = np.random.randint(-3, 3)
    time_steps = np.linspace(start, start+10, num_time_steps)
    data = np.sin(time_steps).reshape(num_time_steps, 1)
    X = torch.tensor(data[: -1], dtype=torch.float).view(1, num_time_steps-1, 1).to(device)
    y = torch.tensor(data[1: ], dtype=torch.float).view(1, num_time_steps-1, 1).to(device)
    predictions = []
    input = X[:, 0, :]
    for _ in range(X.shape[1]):
        input = input.view(1, 1, 1)
        pred = rnn(input)
        input = pred
        predictions.append(pred.cpu().detach().numpy().ravel()[0])

    x = X.data.numpy().ravel()
    y = y.data.numpy()
    plt.scatter(time_steps[:-1], x.ravel(), s=90)
    plt.plot(time_steps[:-1], x.ravel())

    plt.scatter(time_steps[1:], predictions)
    plt.show()
