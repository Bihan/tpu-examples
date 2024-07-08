import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import multiprocessing as mp
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch_xla.distributed.xla_backend
from torch.utils.data import DataLoader, Dataset

lock = mp.Manager().Lock()

# Define a simple custom dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_dim, output_dim, device):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random inputs and outputs
        data = torch.randn(self.input_dim, device=self.device)
        target = torch.randn(self.output_dim, device=self.device)
        return data, target

def toy_model(index, lock):
    device = xm.xla_device()

    # Initialize dataset and dataloader
    training_data = RandomDataset(num_samples=1024, input_dim=128, output_dim=10, device=device)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    
    # Initialize a basic model
    model = nn.Linear(128, 10).to(device)

    # Optional for TPU v4 and GPU
    xm.broadcast_master_param(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.001)
    for epoch in range(10):
        model.train()
        for batch in train_dataloader:
            # Generate random inputs and outputs on the XLA device
            data, target = batch
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            # Run the pending graph
            xm.mark_step()

    with lock:
        # Print mean parameters so we can confirm they're the same across replicas
        print(index, [p.mean() for p in model.parameters()])

if __name__ == '__main__':
    xmp.spawn(toy_model, args=(lock,))