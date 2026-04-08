#!/usr/bin/env python3
"""
RDMA vs TCP 自动化性能对比基准测试

本脚本运行 AllReduce benchmark 并输出结构化的 JSON 结果，
方便后续分析和可视化。

使用方法：
    # RDMA 模式
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    mpirun -np 2 python benchmark_compare.py --output rdma_results.json

    # TCP 模式
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    mpirun -np 2 python benchmark_compare.py --output tcp_results.json

    # 对比结果
    python -c "
    import json
    rdma = json.load(open('rdma_results.json'))
    tcp = json.load(open('tcp_results.json'))
    print(f'Latency speedup: {tcp[\"avg_latency_ms\"]/rdma[\"avg_latency_ms\"]:.1f}x')
    print(f'Bandwidth speedup: {rdma[\"avg_bandwidth_gbps\"]/tcp[\"avg_bandwidth_gbps\"]:.1f}x')
    "
"""

import os
import time
import json
import torch
import torch.distributed as dist
import argparse


def detect_transport_mode():
    """检测当前使用的传输模式"""
    ib_disable = os.environ.get("NCCL_IB_DISABLE", "0")
    if ib_disable == "1":
        return "tcp"
    return "rdma"


def benchmark(iterations=50, tensor_mb=100, warmup=10):
    """
    运行 AllReduce benchmark 并返回详细结果。

    Args:
        iterations: 测试迭代次数
        tensor_mb: Tensor 大小（MB）
        warmup: 预热迭代次数

    Returns:
        dict: 包含延迟、带宽等性能指标
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    tensor_size = int(tensor_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    tensor = torch.ones(tensor_size, dtype=torch.float32, device=device)

    # Warmup
    if rank == 0:
        print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    if rank == 0:
        print(f"Running benchmark ({iterations} iterations, {tensor_mb} MB)...")

    latencies = []
    for i in range(iterations):
        # 重置 tensor 避免数值溢出
        tensor.fill_(1.0)
        torch.cuda.synchronize()

        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    data_size_gb = tensor.element_size() * tensor.nelement() / (1024**3)
    bandwidths = [data_size_gb / (lat / 1000) for lat in latencies]

    # 排序用于百分位数计算
    sorted_latencies = sorted(latencies)

    return {
        "mode": detect_transport_mode(),
        "world_size": world_size,
        "tensor_mb": tensor_mb,
        "iterations": iterations,
        "warmup": warmup,
        # 延迟统计 (ms)
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "median_latency_ms": round(sorted_latencies[len(sorted_latencies) // 2], 3),
        "p95_latency_ms": round(sorted_latencies[int(len(sorted_latencies) * 0.95)], 3),
        "p99_latency_ms": round(sorted_latencies[int(len(sorted_latencies) * 0.99)], 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "stddev_latency_ms": round(
            (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5,
            3,
        ),
        # 带宽统计 (GB/s)
        "avg_bandwidth_gbps": round(sum(bandwidths) / len(bandwidths), 3),
        "max_bandwidth_gbps": round(max(bandwidths), 3),
        # 原始数据
        "latencies": [round(x, 3) for x in latencies],
        "bandwidths": [round(x, 3) for x in bandwidths],
    }


def print_results(results):
    """格式化打印结果"""
    print("\n" + "=" * 60)
    print(f"  Benchmark Results: {results['mode'].upper()} Mode")
    print("=" * 60)
    print(f"  World Size:         {results['world_size']} processes")
    print(f"  Tensor Size:        {results['tensor_mb']} MB")
    print(f"  Iterations:         {results['iterations']}")
    print(f"  Warmup:             {results['warmup']}")
    print("-" * 60)
    print(f"  Avg Latency:        {results['avg_latency_ms']:.3f} ms")
    print(f"  Median Latency:     {results['median_latency_ms']:.3f} ms")
    print(f"  P95 Latency:        {results['p95_latency_ms']:.3f} ms")
    print(f"  P99 Latency:        {results['p99_latency_ms']:.3f} ms")
    print(f"  Min Latency:        {results['min_latency_ms']:.3f} ms")
    print(f"  Max Latency:        {results['max_latency_ms']:.3f} ms")
    print(f"  Stddev:             {results['stddev_latency_ms']:.3f} ms")
    print("-" * 60)
    print(f"  Avg Bandwidth:      {results['avg_bandwidth_gbps']:.3f} GB/s")
    print(f"  Max Bandwidth:      {results['max_bandwidth_gbps']:.3f} GB/s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="RDMA vs TCP AllReduce Benchmark"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--tensor_mb", type=int, default=100, help="Tensor size in MB"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    # 初始化
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()

    results = benchmark(
        iterations=args.iterations,
        tensor_mb=args.tensor_mb,
        warmup=args.warmup,
    )

    if rank == 0:
        print_results(results)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
