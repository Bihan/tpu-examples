import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from accelerate import Accelerator
import torch.optim as optim

class RandomDataset(Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random inputs and outputs
        torch.manual_seed(42 + idx)
        data = torch.randn(self.input_dim)
        target = torch.randn(self.output_dim)
        return data, target

def mac_test():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    else:
        mps_device = torch.device("mps")
        print(mps_device)
        # Create a Tensor directly on the mps device
        x = torch.ones(5, device=mps_device)
        y = x * 2
        print(y)

def data_loader():
    t = torch.arange(6, dtype=torch.float32)
    print(f'{t}')
    data_loader = DataLoader(t, batch_size=3)
    print('Dataloader Batch Sizes')
    for batch in data_loader:
        print(batch)

def dataset_dataloader():
    training_data = RandomDataset(num_samples=10, input_dim=3, output_dim=1)
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
    for batch in train_dataloader:
        data, target = batch
        print(data)

def main():
    # Initialize dataset and dataloader
    training_data = RandomDataset(num_samples=1024, input_dim=10, output_dim=3)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    accelerator = Accelerator()
    model = nn.Linear(10, 3)

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
    # mac_test()
    main()