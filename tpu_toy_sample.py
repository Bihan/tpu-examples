import torch
import torch_xla
import multiprocessing as mp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

print(torch.__version__)
print(torch_xla.__version__)

lock = mp.Manager().Lock()

def print_device(i, lock):
    device = xm.xla_device()
    with lock:
        print('process', i, device)

def add_ones(i, lock):
    x = torch.ones((3,3), device=xm.xla_device())
    y = x+x
    # Forces all devices to evaluate the current graph
    xm.mark_step()
    with lock:
        print(i, y)

def toy_model(index, lock):
    device = xm.xla_device()
    # Initialize distributed environment to communicate and sync multiple processes.
    # Set backend 'xla' for TPU
    dist.init_process_group('xla', init_method='xla://')

    # Print the world size and rank
    # Get the world size and rank
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    with lock:
        print(f'World size: {world_size}, Rank: {rank}')

    torch.manual_seed(42)
    # Simple model with 128 input features and 10 output features
    model = nn.Linear(128, 10).to(device)

    # Ensures that model parameters are broadcasted from the master core to all other cores.
    xm.broadcast_master_param(model)

    # DDP provides data parallelism by synchronizing gradients across each model replica. 
    # The devices to synchronize across are specified by the input process_group, 
    # which is the entire world by default. 
    # Note that DistributedDataParallel does not chunk or otherwise shard the 
    # input across participating GPUs; the user is responsible for defining how 
    # to do so, for example through the use of a DistributedSampler.

    # gradient_as_bucket_view=True improves performance by reducing memory usage 
    # and the overhead associated with creating and managing multiple gradient
    model = DDP(model, gradient_as_bucket_view=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=.001)

    for i in range(10):
        data, target = torch.randn((128, 128), device=device), torch.randn((128, 10), device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Run the pending graph
        xm.mark_step()
    
    with lock:
        print(index, [p.mean() for p in model.parameters()])

if __name__ == '__main__':
    xmp.spawn(toy_model, args=(lock,))
    # xmp.spawn(print_device, args=(lock,))
    # xmp.spawn(add_ones, args=(lock,))