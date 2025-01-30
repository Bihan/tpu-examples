import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP

class MyMLPModel(nn.Module):
    def __init__(self):
        super(MyMLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # Input layer with 784 features (28x28 image)
        self.fc2 = nn.Linear(256, 128)  # Hidden layer with 256 input and 128 output units
        self.fc3 = nn.Linear(128, 10)   # Output layer with 128 input units and 10 output units (classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after hidden layer
        x = self.fc3(x)          # Output layer (no activation, as it's used for logits)
        return x

# Instantiate the model
my_module = MyMLPModel()

# Wrap the model with FSDP for distributed training
model = FSDP(my_module)

# Define the optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# Transfer the model to an XLA device
device = xm.xla_device()
model = model.to(device)

# Example input tensors
x = torch.randn(32, 784).to(device)  # Batch of 32 samples with 784 features each (28*28 pixels flattened)
y = torch.randint(0, 10, (32,)).to(device)  # Batch of 32 labels

# Forward pass
output = model(x)

# Compute loss using CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
loss = criterion(output, y)

# Backward pass
loss.backward()

# Optimizer step
optim.step()
