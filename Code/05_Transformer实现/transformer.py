# ================================
# @File         : transformer.py
# @Time         : 2025/07/10
# @Author       : Yingrui Chen
# @description  : 搭建基础的Transformer模型
#                 参考文章https://zhuanlan.zhihu.com/p/338817680
# ================================

import torch
import numpy as np
from torch import nn


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, dim_input):
        super().__init__()

        self.WQ = nn.Linear(dim_input, dim_input)
        self.WK = nn.Linear(dim_input, dim_input)
        self.WV = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        Z = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.shape[-1])
        Z = self.softmax(Z)
        Z = torch.matmul(Z, V)

        return Z
    

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_input, num_heads):
        super().__init__()

        self.dim_input = dim_input
        self.num_heads = num_heads

        self.self_attention = SelfAttention(dim_input)
        self.linear = nn.Linear(dim_input * num_heads, dim_input)

    def forward(self, X):
        Z = torch.cat([self.self_attention(X) for _ in range(self.num_heads)], dim=-1)
        Z = self.linear(Z)

        return Z
    
# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward):
        super().__init__()

        self.muti_head_attention = MultiHeadAttention(dim_input, num_heads)
        self.layer_norm_1 = nn.LayerNorm(dim_input)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_input, dim_feed_forward), 
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_input)
        )
        self.layer_norm_2 = nn.LayerNorm(dim_input)

    def forward(self, X):
        Z = self.muti_head_attention(X)
        X = self.layer_norm_1(X + Z)
        Z = self.feed_forward(X)
        X = self.layer_norm_2(X + Z)

        return X

# Encoder
class Encoder(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward, num_layers):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(dim_input, num_heads, dim_feed_forward) for _ in range(num_layers)
            ]
        )

    def forward(self, X):
        for encoder_block in self.encoder_blocks:
            X = encoder_block(X)

        return X
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward, C):
        super().__init__()

        

    
if __name__ == '__main__':
    # 生成测试案例
    X = torch.randn(3, 4, 2)

    encoder = Encoder(2, 2, 4, 2)
    print(encoder)
    # encoder = encoder.to(device)
    # X = X.to(device)
    # Z = encoder(X)

    # print(Z)
    # print(Z.shape)
