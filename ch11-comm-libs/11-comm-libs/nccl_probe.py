#!/usr/bin/env python3
"""有 CUDA 和 NCCL 时跑一次集合通信并核对数值。没有设备时 SKIP，退出码 0。"""

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


def main():
    sys.stdout.reconfigure(line_buffering=True)
    bootstrap_env()
    cuda = torch.cuda.is_available()
    nccl = dist.is_nccl_available()
    devices = torch.cuda.device_count() if cuda else 0
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if rank == 0:
        print(f"torch={torch.__version__} is_nccl_available={nccl} cuda={cuda} device_count={devices} world={world}")
    if not cuda or not nccl:
        if rank == 0:
            print("RESULT: SKIP NCCL collectives need a CUDA build of PyTorch and a visible GPU")
        return 0
    if world > devices:
        if rank == 0:
            print(f"RESULT: SKIP world_size {world} > cuda devices {devices}")
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    tensor = torch.tensor([float(rank + 1)], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expect = world * (world + 1) / 2.0
    ok = abs(tensor.item() - expect) < 1e-4

    send_counts = [1] * world
    send = torch.tensor([float(rank * 100 + dst) for dst in range(world)], dtype=torch.float32, device=device)
    recv = torch.empty(world, dtype=torch.float32, device=device)
    dist.all_to_all_single(recv, send, send_counts, send_counts)
    for src in range(world):
        if abs(recv[src].item() - float(src * 100 + rank)) > 1e-4:
            ok = False

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    failed = flag.item() != 1
    if rank == 0:
        version = torch.cuda.nccl.version()
        print(f"nccl_version={version} allreduce={tensor.item():.0f} expected={expect:.0f}")
        print("RESULT: FAIL" if failed else "RESULT: PASS")
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
