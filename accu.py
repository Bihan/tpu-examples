import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
from accelerate import Accelerator


# Define a simple custom dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random inputs and outputs
        data = torch.randn(self.input_dim)
        target = torch.randn(self.output_dim)
        return data, target


def main():
    # Initialize dataset and dataloader
    training_data = RandomDataset(num_samples=1024, input_dim=128, output_dim=10)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    accelerator = Accelerator()
    
    model = nn.Linear(128, 10)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.001)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    for epoch in range(10):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                data, target = batch
                output = model(data)
                loss = loss_fn(output, target)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
    print([p.mean().item() for p in model.parameters()])

if __name__ == '__main__':
    main()
