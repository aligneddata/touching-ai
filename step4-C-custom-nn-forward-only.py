# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 00:46:03 2025
"""

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)   # First layer
        x = self.relu(x)  # Activation
        x = self.fc2(x)   # Second layer
        return x

model = MyModel()
x = torch.rand(3, 4)  # or: ones(3, 4), zeros(3, 4)
print("x:", x)
y = model(x)  # Calls forward() internally
print("y:", y)
