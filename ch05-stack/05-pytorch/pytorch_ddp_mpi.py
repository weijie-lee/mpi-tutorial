#!/usr/bin/env python3
"""最小的 DistributedDataParallel 训练步。

NCCL 后端不会从 mpirun 里自动继承 rank。启动信息走 env://：

    mpirun -np 2 python3 pytorch_ddp_mpi.py
    torchrun --nproc_per_node=2 pytorch_ddp_mpi.py

没有 GPU 时退回 gloo + CPU，用来核对梯度同步后各进程权重一致。
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def bootstrap_env():
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


class SimpleDataset(Dataset):
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        x = torch.randn(self.dim, generator=generator)
        y = torch.tensor([1.0 if x.sum() > 0 else 0.0])
        return x, y


def main():
    bootstrap_env()
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if use_cuda:
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        backend = "nccl"
        device = torch.device("cuda", device_index)
    else:
        device_index = None
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[rank {rank}/{world_size}] backend={backend} device={device}")

    model = nn.Linear(10, 1).to(device)
    if use_cuda:
        ddp_model = DDP(model, device_ids=[device_index])
    else:
        ddp_model = DDP(model)

    dataset = SimpleDataset(128, 10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        output = ddp_model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx == 0 and rank == 0:
            print(f"[rank 0] first batch loss = {loss.item():.4f}")

    weight = next(ddp_model.parameters()).detach().reshape(-1)[0]
    reference = weight.clone()
    dist.broadcast(reference, src=0)
    synced = torch.tensor(
        [1 if abs(weight.item() - reference.item()) < 1e-5 else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(synced, op=dist.ReduceOp.MIN)
    if rank != 0:
        print(f"[rank {rank}] weights synchronized" if synced.item() == 1 else f"[rank {rank}] weights diverged")
    if rank == 0:
        print("RESULT: PASS" if synced.item() == 1 else "RESULT: FAIL")
    dist.destroy_process_group()
    return 0 if synced.item() == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
