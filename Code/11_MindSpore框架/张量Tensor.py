# ================================
# @File         : 张量Tensor.py
# @Time         : 2025/07/19
# @Author       : Yingrui Chen
# @description  : MindSpore张量
# ================================
import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import One, Normal


# 直接从数据生成张量
data = [1, 2, 3, 4]
x_data = Tensor(data)
# print(x_data)

# 从numpy数组生成张量
np_array = np.array([1, 2, 3, 4])
x_np = Tensor(np_array)
# print(x_np)

# 使用init初始化构造张量
tensor1 = Tensor(shape=(2, 2), dtype=mindspore.float32, init=One())
tensor2 = Tensor(shape=(2, 2), dtype=mindspore.float32, init=Normal())
print(tensor1, '\n', tensor2)
