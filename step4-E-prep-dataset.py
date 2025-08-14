# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 21:50:07 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import Dataset



# randomly split a data collection
a = [4*i for i in range(10)]

m, n = random_split(a, [6, 4])

print("m",[i for i in m])
print("n",[i for i in n],type(n))



# more sophisticated data
class MyDataset(Dataset):
    def __init__(self, length):
        # Store as 2D float32 tensors for compatibility with DataLoader batching and model input
        self._x_raw = [torch.tensor([i], dtype=torch.float32) for i in range(length)]
        self._y_raw = [torch.tensor([3*i], dtype=torch.float32) for i in range(length)]

    def __len__(self):
        return len(self._x_raw)

    def __getitem__(self, idx):
        x = self._x_raw[idx]
        y = self._y_raw[idx]
        return x, y
    
    def print_data(self):
        for x, y in zip(self._x_raw, self._y_raw):
            print(f"x = {x}, y = {y}")


# Main
myDataset = MyDataset(20)
myDataset.print_data()

train_dataset, test_dataset = random_split(myDataset, [16, 4])
print(f"Data type of train_dataset: {type(train_dataset)}")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)



# Setup model 
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1) # Input dim == 1; Output dim == 1

    def forward(self, x):
        return self.linear(x)

# Create an instance of the model
model = SimpleNet()
criterion = nn.MSELoss() # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.001) # Stochastic Gradient Descent; learning rate matters




# Training loop
print("\nTraining a simple neural network:")
# 4. The training loop: get X and Y from the Subset and perform calculations
num_epochs = 5
for epoch in range(num_epochs):
    # Iterate over the DataLoader for the Subset
    for X_batch, Y_batch in train_dataloader:
        print("On loop: epoch->training batch")
        # X_batch = X_batch.float()  # Ensure input is float32
        # Y_batch = Y_batch.float()  # Ensure target is float32
        # 
        model.train()  # Set the model to training mode
        
        # a. Zero the gradients
        optimizer.zero_grad()
        
        # b. Forward pass: get predictions (logits) from the model
        outputs = model(X_batch).squeeze()
        
        # c. Calculate the loss
        loss = criterion(outputs, Y_batch.squeeze())
        
        # d. Backward pass: calculate gradients
        loss.backward()
        
        # e. Update the model's weights
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}")

    # total_loss = 0.0
    # for X_batch, Y_batch in test_dataloader:
    #     print("On loop: epoch->testing batch")
    #     model.eval()  # Set the model to evaluation mode
    #     outputs = model(X_batch).squeeze()
    #     loss = criterion(outputs, Y_batch.squeeze())
    #     total_loss += loss.item()
    # total_loss /= len(test_dataloader)
    # print(f"Epoch [{epoch+1}/{num_epochs}], Testing Loss: {total_loss:.4f}")    
    

# Test the trained model
test_input = torch.tensor([[5.0]], dtype=torch.float32)  # Ensure float32
predicted_output = model(test_input)

print(f"\nPredicted output for input 5.0: {predicted_output.item():.4f}")
# Same model but run test 