"""
Copying tutorial from

https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

Run on multiple GPU. Ex:
salloc --gpus-per-node=2 --time=00:05:00 -A $GROUP

This will print: each step twice
1) Basic demo -----: on GPU 1 out of 2.
1) Basic demo -----: on GPU 2 out of 2.
2) Running from checkpoint -----: on GPU 1 out of 2.
2) Running from checkpoint -----: on GPU 2 out of 2.
3) Basic demo parallel ------: on GPU 1 out of 2, sub-dev 0 and 1.
3) Basic demo parallel ------: on GPU 1 out of 2, sub-dev 0 and 1.

"""

import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_demo(demo_fn, nb_gpus):
    mp.spawn(demo_fn,
             args=(nb_gpus,),
             nprocs=nb_gpus,
             join=True)


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        print("        Inside ToyModel Forward. x shape: {}".format(x.shape))
        return self.net2(self.relu(self.net1(x)))


class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        print("        Inside ToyMpModel Forward. x shape: {}".format(x.shape))
        print("             - net1 on dev0: {}".format(self.dev0))
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        print("             - Linear on dev1: {}".format(self.dev1))
        x = x.to(self.dev1)
        return self.net2(x)


def demo_basic(rank, nb_gpus):
    print("1) Basic demo ------")
    print(f"  Running basic DDP example on GPU {rank + 1} out of {nb_gpus}.")
    setup(rank, nb_gpus)

    print("  Creating model and moving it to chosen GPU, then to DDP")
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    print("  Preparing optimizer")
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print("  Running one epoch with a random tensor of size 20x10.")
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def demo_checkpoint(rank, nb_gpus):
    print("2) Running from checkpoint ------")
    print(f"  Running basic DDP example on GPU {rank + 1} out of {nb_gpus}.")
    setup(rank, nb_gpus)

    print("  Creating model and moving it to chosen GPU, then to DDP")
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    print("  Saving checkpoint")
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    print("  Loading again from checkpoint")
    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    print("  Preparing optimizer")
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    print("  Running one epoch with a random tensor of size 20x10.")
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.
    print("  Deleting checkpoint")
    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def demo_model_parallel(rank, nb_gpus):
    print("3) Basic demo parallel ------")
    print(f"  Running basic DDP example on GPU {rank + 1} out of {nb_gpus}.")
    setup(rank, nb_gpus)

    dev0 = (rank * 2) % nb_gpus
    dev1 = (rank * 2 + 1) % nb_gpus
    print("  Creating model and moving it to GPU with id rank, then to DDP,"
          "using two sub-dev: {} and {}".format(dev0, dev1))
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def test_distributed_parallel_from_tutorial():
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    run_demo(demo_model_parallel, world_size)


if __name__ == '__main__':
    test_distributed_parallel_from_tutorial()
