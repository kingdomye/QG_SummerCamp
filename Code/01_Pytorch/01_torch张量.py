# ================================
# @File         : 01_torch张量.py
# @Time         : 2025/07/07
# @Author       : Yingrui Chen
# @description  : torch张量的基础用法
# ================================

import torch
import numpy as np

# 1、直接从数据创建张量
data = [[2025, 7, 7], [9, 6, 45]]
x_data = torch.tensor(data)
print(x_data)

# 2、从NumPy数组创建张量
np_array = np.array(data)
x_np = torch.tensor(np_array)
print(x_np)

# 3、其他张量创建函数
x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=float)
print(x_rand)

rand_tensor = torch.rand((3, 3, ))
ones_tensor = torch.ones((2, 3, ))
print(rand_tensor, ones_tensor)


# 4、tensor属性
print(f"tensor shape: {x_data.shape}")
print(f"tensor dtype: {x_np.dtype}")


# 5、tensor操作
# 转置、切片、索引操作均与python列表类似
# tensor运算与numpy数组运算类似
# 方法较多，建议查询torch API
