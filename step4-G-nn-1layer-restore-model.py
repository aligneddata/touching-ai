# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:19:38 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim


# 3. Simple Neural Network
# Define a simple linear model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1 layer with 1 node - output = input * W + b
        # W (the weight of the linear layer) is a (1, 1) tensor.
        # b (the bias of the linear layer) is a (1, 1) tensor.
        self.linear = nn.Linear(1, 1) # Input dimension == 1; Output dim == 1

    def forward(self, x):
        return self.linear(x)

# Create an instance of the model

model = SimpleNet()
model.load_state_dict(torch.load('simple_model_epoch_50.pth'))
model.eval()  # Set the model to evaluation mode to disable dropout and batch normalization


# Test the trained model
test_input = torch.tensor([[5.0]])
predicted_output = model(test_input)
print(f"\nPredicted output for input 5.0: {predicted_output.item():.4f}")