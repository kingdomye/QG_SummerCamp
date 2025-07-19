import mindspore
import numpy as np
from mindspore import nn, ops, Tensor


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits


if __name__ == "__main__":
    network = Network()
    X = Tensor(np.ones((1, 28, 28)), mindspore.float32)
    logits = network(X)
    print(network)
    print(logits)
