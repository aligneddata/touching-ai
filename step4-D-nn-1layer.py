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

# Define a loss function and an optimizer
criterion = nn.MSELoss() # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent

# Dummy data for training
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]]) # Target: y = 2x

# Training loop
print("\nTraining a simple neural network:")
for epoch in range(100):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update model parameters

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')  
        # loss.item() is to output the value in Python scalar type

# Test the trained model
test_input = torch.tensor([[5.0]])
predicted_output = model(test_input)
print(f"\nPredicted output for input 5.0: {predicted_output.item():.4f}")