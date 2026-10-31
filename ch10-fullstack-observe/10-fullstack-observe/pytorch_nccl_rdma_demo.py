#!/usr/bin/env python3
"""观察一次 AllReduce，并核对数值。

NCCL 环境变量要在 import torch 之前设置。没有 GPU 时走 gloo，
打印 cpu-gloo，避免把 CPU 回环测出来的数字说成 RDMA 带宽。

    mpirun -np 2 python3 pytorch_nccl_rdma_demo.py --tensor_mb 4 --iterations 10
    mpirun -np 2 python3 pytorch_nccl_rdma_demo.py --disable_rdma
"""

import os
import sys


def configure_nccl(argv):
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,ENV")
    if "--disable_rdma" in argv:
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"
    else:
        os.environ.setdefault("NCCL_IB_DISABLE", "0")


configure_nccl(sys.argv)

import argparse
import time

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


def transport_mode():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "cpu-gloo"
    if os.environ.get("NCCL_IB_DISABLE") == "1":
        return "nccl-tcp-forced"
    return "nccl-default"


def main():
    parser = argparse.ArgumentParser(description="AllReduce observation")
    parser.add_argument("--disable_rdma", action="store_true")
    parser.add_argument("--tensor_mb", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    bootstrap_env()
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if use_cuda:
        index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mode = transport_mode()
    if rank == 0:
        print(f"mode={mode} world_size={world_size} device={device}")
        if mode == "cpu-gloo":
            print("No CUDA device. This run does not exercise NCCL or RDMA.")
        elif mode == "nccl-default":
            print("NCCL may use IB, RoCE, or a socket fallback. Read NCCL INFO lines to see which.")

    tensor_size = max(1, int(args.tensor_mb * 1024 * 1024 / 4))
    tensor = torch.empty(tensor_size, dtype=torch.float32, device=device)
    expected = world_size * (world_size + 1) / 2.0

    for _ in range(2):
        tensor.fill_(float(rank + 1))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    correct = abs(tensor[0].item() - expected) < 1e-2

    latencies = []
    for i in range(args.iterations):
        tensor.fill_(float(rank + 1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        if rank == 0 and (i + 1) % 5 == 0:
            print(f"  iteration {i + 1}/{args.iterations} latency {latencies[-1]:.3f} ms")

    if abs(tensor[0].item() - expected) > 1e-2:
        correct = False

    ok = torch.tensor([1 if correct else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        avg = sum(latencies) / len(latencies)
        data_gb = tensor.element_size() * tensor.nelement() / 1e9
        alg = data_gb / (avg / 1000)
        bus_factor = 1.0 if world_size == 1 else 2.0 * (world_size - 1) / world_size
        print(f"value={tensor[0].item():.1f} expected={expected:.1f}")
        print(f"avg_latency_ms={avg:.3f} alg_GBps={alg:.3f} bus_GBps={alg * bus_factor:.3f}")
        print("RESULT: PASS" if ok.item() == 1 else "RESULT: FAIL")
    dist.destroy_process_group()
    return 0 if ok.item() == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
