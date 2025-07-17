import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision import models


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 读取标签并且转化为字典
def read_label():
    label_csv = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
    label_dict = label_csv.set_index('image_id').to_dict()['label']
    return label_dict

# 读取图片路径
def get_images_path(directory):
    images_path = [os.path.join(directory, file) for file in os.listdir(directory)]
    return images_path

# 图片转tensor
def image_to_tensor(image_path):
    image = Image.open(image_path)
    image_tensor = transforms.ToTensor()(image)
    return image_tensor

# 制作torch dataset数据集
def get_train_dataset(directory):
    images_paths = get_images_path(directory=directory)
    label_dict = read_label()

    dataset = []
    for image_path in images_paths:
        image_name = os.path.basename(image_path)
        target = label_dict[image_name]

        dataset.append((image_path, image_name, target))

    return dataset
    
def get_test_dataset(directory):
    images_paths = get_images_path(directory=directory)

    dataset = []
    for image_path in images_paths:
        image_name = os.path.basename(image_path)

        dataset.append((image_path, image_name))

    return dataset

# dataset分批次
def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(dataloader, model, loss_fn, optimizer, epochs, device):
    model.train()
    model.to(device)  # 将模型移至指定设备
    for epoch in range(epochs):
        for batch in dataloader:
            images_paths, image_names, targets = batch
            images = [image_to_tensor(image_path) for image_path in images_paths]
            images = torch.stack(images).to(device)  # 将图像数据移至指定设备
            targets = torch.tensor(targets).to(device)  # 将目标标签移至指定设备

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

def test(dataset, model, device):
    result = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataset:
            image_path, image_names = batch
            image = image_to_tensor(image_path)
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1)
            result.append((image_names, pred.item()))

    return result

model = models.resnet50(weights=None)
num_classes = 5
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

if __name__ == '__main__':
    epochs = 1
    batch_size = 4
    learning_rate = 1e-3

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = get_train_dataset(directory='/kaggle/input/cassava-leaf-disease-classification/train_images')
    train_dataloader = get_dataloader(train_dataset, batch_size)

    test_dataset = get_test_dataset(directory='/kaggle/input/cassava-leaf-disease-classification/test_images')
    res = test(test_dataset, model, device)
    df = pd.DataFrame(res, columns=['image_id', 'label'])
    df.to_csv('submission.csv', index=False)
