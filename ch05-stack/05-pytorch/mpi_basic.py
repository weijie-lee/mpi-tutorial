#!/usr/bin/env python3
"""用 torch.distributed 做一次 AllReduce，并核对求和结果。

pip 安装的 PyTorch 通常没有编进 MPI，backend='mpi' 会直接失败。
torchrun / mpirun 真正常用的是 env:// 启动：

    mpirun -np 2 python3 mpi_basic.py
    # 或者
    torchrun --nproc_per_node=2 mpi_basic.py

环境变量 RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT 由 torchrun 设置。
OpenMPI 不会设置它们，脚本会从 OMPI_COMM_WORLD_* 补上。
有 CUDA 时用 nccl，否则用 gloo，这样没有 GPU 也能把实验跑通。
设置 USE_MPI_BACKEND=1 且当前 PyTorch 带 MPI 时才走 backend='mpi'。
"""

import os
import sys

import torch
import torch.distributed as dist


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


def init_distributed():
    bootstrap_env()
    use_mpi = os.environ.get("USE_MPI_BACKEND") == "1" and dist.is_mpi_available()
    if use_mpi:
        dist.init_process_group(backend="mpi")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return "mpi", device

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        dist.init_process_group(backend="nccl", init_method="env://")
        return "nccl", torch.device("cuda", device_index)

    dist.init_process_group(backend="gloo", init_method="env://")
    return "gloo", torch.device("cpu")


def main():
    backend, device = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Hello from rank {rank}/{world_size} backend={backend} device={device}")

    tensor = torch.tensor([float(rank)], dtype=torch.float32, device=device)
    print(f"Rank {rank}: before allreduce tensor = {tensor.item()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = (world_size - 1) * world_size / 2.0
    got = tensor.item()
    print(f"Rank {rank}: after allreduce tensor = {got}, expected {expected}")

    ok = torch.tensor([1 if abs(got - expected) < 1e-4 else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("RESULT: PASS" if ok.item() == 1 else "RESULT: FAIL")
    dist.destroy_process_group()
    return 0 if ok.item() == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
