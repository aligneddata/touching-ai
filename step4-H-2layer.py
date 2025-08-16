# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 17:09:54 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 3. More Complex Neural Network
# Define a two-layer model
class MoreComplexNet(nn.Module):
    def __init__(self):
        super(MoreComplexNet, self).__init__()
        # First linear layer: 2 input features -> 4 hidden features
        self.linear1 = nn.Linear(2, 4)
        # Activation function to introduce non-linearity
        self.relu = nn.ReLU()
        # Second linear layer: 4 hidden features -> 1 output feature
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create an instance of the model
model = MoreComplexNet()

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data for training: now with 2 features per sample
X_train = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])
# Target: y = 2 * x1 + 3 * x2 (where x1 and x2 are the two features)
y_train = torch.tensor([
    [8.0],
    [13.0],
    [18.0],
    [23.0]
])

# Training loop
print("\nTraining a more complex neural network:")
for epoch in range(100):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test the trained model with new data
test_input = torch.tensor([[5.0, 6.0]])
predicted_output = model(test_input)
# Expected output for 5.0, 6.0 is 2*5 + 3*6 = 28
print(f"\nPredicted output for input [5.0, 6.0]: {predicted_output.item():.4f}"
      + "; expected output: 28")