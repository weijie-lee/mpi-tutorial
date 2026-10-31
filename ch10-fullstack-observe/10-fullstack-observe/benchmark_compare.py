#!/usr/bin/env python3
"""AllReduce 基准。输出 JSON，便于和另一次运行对比。

没有 GPU 时使用 gloo，mode 记为 cpu-gloo，不会把这次结果叫作 RDMA。
有 GPU 且没有设置 NCCL_IB_DISABLE=1 时，mode 记为 nccl-default：NCCL 会自己选择 IB 或 socket。
本进程看不到网卡完成队列，不能仅凭环境变量断言数据走了 RDMA。

    mpirun -np 2 python3 benchmark_compare.py --tensor_mb 4 --iterations 10 --output results.json

带宽单位是 GB/s（十进制 1e9 字节/秒），不是 Gbps。
算法带宽 = 消息大小 / 单次耗时。
总线带宽 = 算法带宽 * 2*(p-1)/p。
"""

import argparse
import json
import os
import sys
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


def detect_mode():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "cpu-gloo"
    if os.environ.get("NCCL_IB_DISABLE", "0") == "1":
        return "nccl-tcp-forced"
    return "nccl-default"


def init_distributed():
    bootstrap_env()
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(index)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", index)
    dist.init_process_group(backend="gloo", init_method="env://")
    return torch.device("cpu")


def percentile(sorted_values, fraction):
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def benchmark(device, iterations, tensor_mb, warmup):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor_size = max(1, int(tensor_mb * 1024 * 1024 / 4))
    tensor = torch.ones(tensor_size, dtype=torch.float32, device=device)

    check = torch.tensor([float(rank + 1)], dtype=torch.float32, device=device)
    dist.all_reduce(check, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2.0
    correct = abs(check.item() - expected) < 1e-3

    if rank == 0:
        print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        tensor.fill_(1.0)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"Running benchmark ({iterations} iterations, {tensor_mb} MB)...")

    latencies = []
    for _ in range(iterations):
        tensor.fill_(1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    if abs(tensor[0].item() - float(world_size)) > 1e-3:
        correct = False

    data_size_gb = tensor.element_size() * tensor.nelement() / 1e9
    bandwidths = [data_size_gb / (lat / 1000) for lat in latencies]
    bus_factor = 1.0 if world_size == 1 else 2.0 * (world_size - 1) / world_size
    sorted_latencies = sorted(latencies)
    avg_alg = sum(bandwidths) / len(bandwidths)

    return {
        "mode": detect_mode(),
        "world_size": world_size,
        "tensor_mb": tensor_mb,
        "iterations": iterations,
        "warmup": warmup,
        "correct": correct,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "median_latency_ms": round(percentile(sorted_latencies, 0.5), 3),
        "p95_latency_ms": round(percentile(sorted_latencies, 0.95), 3),
        "p99_latency_ms": round(percentile(sorted_latencies, 0.99), 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "stddev_latency_ms": round(
            (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5,
            3,
        ),
        "avg_alg_bandwidth_GBps": round(avg_alg, 3),
        "avg_bus_bandwidth_GBps": round(avg_alg * bus_factor, 3),
        # 保留旧字段名。单位仍是 GB/s，不是 gigabit/s。
        "avg_bandwidth_gbps": round(avg_alg, 3),
        "max_bandwidth_gbps": round(max(bandwidths), 3),
        "latencies": [round(x, 3) for x in latencies],
        "bandwidths": [round(x, 3) for x in bandwidths],
    }


def print_results(results):
    print("\n" + "=" * 60)
    print(f"  Benchmark Results: {results['mode']}")
    print("=" * 60)
    print(f"  World Size:         {results['world_size']} processes")
    print(f"  Tensor Size:        {results['tensor_mb']} MB")
    print(f"  Correctness:        {results['correct']}")
    print(f"  Avg Latency:        {results['avg_latency_ms']:.3f} ms")
    print(f"  P95 Latency:        {results['p95_latency_ms']:.3f} ms")
    print(f"  Alg Bandwidth:      {results['avg_alg_bandwidth_GBps']:.3f} GB/s")
    print(f"  Bus Bandwidth:      {results['avg_bus_bandwidth_GBps']:.3f} GB/s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AllReduce benchmark")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tensor_mb", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    device = init_distributed()
    rank = dist.get_rank()
    results = benchmark(device, args.iterations, args.tensor_mb, args.warmup)
    ok = torch.tensor([1 if results["correct"] else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    results["correct"] = bool(ok.item())

    if rank == 0:
        print_results(results)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Results saved to {args.output}")
        print("RESULT: PASS" if results["correct"] else "RESULT: FAIL")
    dist.destroy_process_group()
    return 0 if ok.item() == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
