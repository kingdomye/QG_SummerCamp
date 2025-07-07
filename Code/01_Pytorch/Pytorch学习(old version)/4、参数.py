import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2
batch_size = 30
data_size = 100
device = torch.device('mps')


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\t【In Model】input size", input.size(), "\toutput size", output.size())

        return output


if __name__ == '__main__':
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)
    model = Model(input_size, output_size)

    print("Let's use", torch.mps.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    model.to(device)

    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("\t【Outside】input size", input.size(), "\toutput size", output.size())
