import torch
import torch.nn as nn
from torchvision import models

model = models.efficientnet_b4(pretrained=False)
num_classes = 5
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output {output}')
    print(f'Output shape: {output.shape}')