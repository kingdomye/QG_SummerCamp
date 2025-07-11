# ================================
# @File         : transformer.py
# @Time         : 2025/07/10
# @Author       : Yingrui Chen
# @description  : 搭建基础的Transformer模型
#                 参考文章 https://zhuanlan.zhihu.com/p/338817680
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


# SelfAttention
class SelfAttention(nn.Module):
    def __init__(self, dim_input):
        super().__init__()

        self.WQ = nn.Linear(dim_input, dim_input)
        self.WK = nn.Linear(dim_input, dim_input)
        self.WV = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=False):
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)

        Z = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.shape[-1])
        if mask:
            Z = Z.masked_fill(torch.triu(torch.ones_like(Z), diagonal=1) == 1, float('-inf'))
            print(f'Masked Test Z : {Z}')

        Z = self.softmax(Z)
        Z = torch.matmul(Z, V)

        return Z
    

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_input, num_heads, mask=False):
        super().__init__()

        self.dim_input = dim_input
        self.num_heads = num_heads

        self.self_attention = SelfAttention(dim_input)
        self.linear = nn.Linear(dim_input * num_heads, dim_input)

    def forward(self, Q, K, V, mask=False):
        Z = torch.cat([self.self_attention(Q, K, V, mask) for _ in range(self.num_heads)], dim=-1)
        Z = self.linear(Z)

        return Z
    
# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(dim_input, num_heads)
        self.layer_norm_1 = nn.LayerNorm(dim_input)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_input, dim_feed_forward), 
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_input)
        )
        self.layer_norm_2 = nn.LayerNorm(dim_input)

    def forward(self, X):
        Z = self.multi_head_attention(X, X, X, mask=False)
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
                EncoderBlock(
                    dim_input, 
                    num_heads, 
                    dim_feed_forward
                ) for _ in range(num_layers)
            ]
        )

    def forward(self, X):
        for encoder_block in self.encoder_blocks:
            X = encoder_block(X)

        return X
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward):
        super().__init__()

        self.multi_head_attention_1 = MultiHeadAttention(dim_input, num_heads)
        self.layer_norm_1 = nn.LayerNorm(dim_input)
        self.multi_head_attention_2 = MultiHeadAttention(dim_input, num_heads)
        self.layer_norm_2 = nn.LayerNorm(dim_input)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_input, dim_feed_forward), 
            nn.ReLU(),
            nn.Linear(dim_feed_forward, dim_input)
        )
        self.layer_norm_3 = nn.LayerNorm(dim_input)

    def forward(self, X):
        X_copy = X
        Z = self.multi_head_attention_1(X)
        X = self.layer_norm_1(X + Z)
        Z = self.multi_head_attention_2(X, X_copy, X_copy, mask=True)
        X = self.layer_norm_2(X + Z)
        Z = self.feed_forward(X)
        X = self.layer_norm_3(X + Z)

        return X
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward, num_layers):
        super().__init__()

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim_input, 
                    num_heads, 
                    dim_feed_forward
                ) for _ in range(num_layers)
            ]
        )

    def forward(self, X):
        for decoder_block in self.decoder_blocks:
            X = decoder_block(X)

        return X
    
# Transformer
class Transformer(nn.Module):
    def __init__(self, dim_input, num_heads, dim_feed_forward, num_layers):
        super().__init__()

        self.encoder = Encoder(dim_input, num_heads, dim_feed_forward, num_layers)
        self.decoder = Decoder(dim_input, num_heads, dim_feed_forward, num_layers)
        self.linear = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        X = self.linear(X)
        X = self.softmax(X)

        return X

    
if __name__ == '__main__':
    tf = Transformer(dim_input=512, num_heads=8, dim_feed_forward=2048, num_layers=6)
    print(tf)
