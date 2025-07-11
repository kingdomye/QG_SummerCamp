import torch
import torch.nn as nn
Z = torch.rand(2, 4, 4)
Z = Z.masked_fill(torch.triu(torch.ones_like(Z), diagonal=1) == 1, float('-inf'))
print(Z)