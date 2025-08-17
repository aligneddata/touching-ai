# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 00:36:02 2025
"""

import torch
import torch.nn as nn
import numpy as np


# #dim == #[s
# size = #,s + 1 except for 1dim

# 1 dim; size 4
n = np.array([1.0, 2.0, 3.0, 4.0])
print(n.shape, n, n.dtype)

m = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Tensor m:\n", m, m.shape, "1 dim; size 4")


# 2 dim; 1dim size = 4, 2dim size = 1
k = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
print(k, k.shape, k.dtype)


# 2 dim: size1 =4; size2 = 2
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x, x.shape, x.dtype)
y = x + 2
print("Tensor y (x + 2):\n", y)

# rand
x = torch.rand(2, 4)  # batch of 2
print("x:", x.shape, x)  # torch.Size([2, 4])

# Linear
fc = nn.Linear(4, 3)  # 4 inputs → 3 outputs
y = fc(x)
print("y:", y.shape, y)  # torch.Size([2, 3])



# layer that reshapes (flattens) the input tensor into a 2D tensor, 
# keeping the batch dimension intact and merging all other dimensions
# into one
flatten = nn.Flatten()

x = torch.rand(2, 3, 4)  # Shape: (batch=2, channels=3, length=4)
y = flatten(x)

print(x.shape)  # torch.Size([2, 3, 4])
print(y.shape)  # torch.Size([2, 12])
