import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import multiprocessing as mp
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch_xla.distributed.xla_backend

lock = mp.Manager().Lock()

def toy_model(index, lock):
    device = xm.xla_device()
    # Initialize distributed environment to communicate and sync multiple processes.
    # Set backend 'xla' for TPU
    # dist.init_process_group('xla', init_method='xla://')


    # Initialize a basic toy model
    torch.manual_seed(42)
    model = nn.Linear(128, 10).to(device)

    # Optional for TPU v4 and GPU
    xm.broadcast_master_param(model)

    # `gradient_as_bucket_view=True` required for XLA
    # model = DDP(model, gradient_as_bucket_view=True)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.001)
    model.train()
    for i in range(10):
        # Generate random inputs and outputs on the XLA device
        data, target = torch.randn((128, 128), device=device), torch.randn((128, 10), device=device)

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