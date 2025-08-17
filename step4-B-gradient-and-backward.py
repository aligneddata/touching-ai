# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 15:17:28 2025
"""

import torch

# 2. Automatic Differentiation (Autograd)
# Create a tensor with requires_grad=True to track computations for gradients
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = a * 3
c = b.sum()
print("\nTensor a:", a)
print("Tensor b (a * 3):", b)
print("Tensor c (b.sum()):", c)

# Compute gradients
c.backward()
print("Gradient of c with respect to a:", a.grad)




# --- another example ---
x = torch.tensor(2.0, requires_grad=True)  # Track this tensor
y = x ** 3                                  # y = x³

y.backward()  # dy/dx = 3x² = 12 at x=2
print(x.grad)  # tensor(12.)



# --- two variables ---
# Create tensors with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward computation
z = x**2 + y**3   # z = 4 + 27 = 31

# Backward pass
z.backward()  # dz/dx = 2x, dz/dy = 3y^2

print(x.grad)  # 4.0
print(y.grad)  # 27.0



