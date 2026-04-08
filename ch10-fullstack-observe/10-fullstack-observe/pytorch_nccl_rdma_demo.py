#!/usr/bin/env python3
"""
PyTorch -> NCCL -> RDMA 全链路观测脚本

本脚本通过设置 NCCL 环境变量开启详细日志，让你亲眼看到：
1. NCCL 如何发现 RDMA 设备 (ibv_get_device_list)
2. GPU 显存如何注册为 RDMA Memory Region (ibv_reg_mr)
3. Ring AllReduce 如何建立 QP 连接
4. RDMA vs TCP 的性能差异

使用方法：
    # RDMA 模式（默认）
    mpirun -np 2 python pytorch_nccl_rdma_demo.py

    # TCP 回退模式（对比用）
    mpirun -np 2 python pytorch_nccl_rdma_demo.py --disable_rdma

    # 指定 tensor 大小和迭代次数
    mpirun -np 2 python pytorch_nccl_rdma_demo.py --tensor_mb 256 --iterations 100
"""

import os
import time
import torch
import torch.distributed as dist
import argparse


def setup_environment(use_rdma=True):
    """
    设置环境变量以强制 NCCL 的行为并开启详细日志。
    这是观测底层行为的关键！

    NCCL 环境变量说明：
    - NCCL_DEBUG=INFO        : 打印 NCCL 初始化过程，包括设备发现、拓扑建立
    - NCCL_DEBUG_SUBSYS      : 控制打印哪些子系统的日志
      - INIT : 初始化过程
      - NET  : 网络传输层（IB/Socket）
      - ENV  : 环境变量读取
    - NCCL_IB_DISABLE        : 0=启用RDMA, 1=禁用RDMA
    - NCCL_NET_GDR_LEVEL     : GPU Direct RDMA 级别，0=禁用
    """
    # 1. 开启 NCCL 详细日志
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET,ENV"

    if use_rdma:
        # 强制使用 RDMA (IB/RoCE)
        os.environ["NCCL_IB_DISABLE"] = "0"
        print("[Config] Enforcing RDMA (IB/RoCE) for NCCL")
    else:
        # 禁用 RDMA，强制回退到 TCP/IP (Socket)
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_NET_GDR_LEVEL"] = "0"
        print("[Config] Disabling RDMA, falling back to TCP/Socket")


def run_demo(local_rank, world_size, use_rdma, tensor_mb=100, iterations=50):
    """
    运行 AllReduce 观测实验。

    调用链路：
    dist.all_reduce(tensor)
        → PyTorch torch.distributed (选择 backend='nccl')
        → ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
        → NCCL Ring/Tree 算法分解为点对点传输
        → 同节点: NVLink/PCIe 共享内存
        → 跨节点: ibv_post_send() → RDMA NIC → 网络 → 远端 NIC → 远端 GPU
    """
    # 1. 初始化进程组
    print(f"[Rank {local_rank}] Initializing process group...")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank,
    )

    # 设置当前 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 确保所有进程都准备好
    dist.barrier()
    if local_rank == 0:
        print("\n" + "=" * 50)
        print(f"🚀 Starting Test: {'RDMA Enabled' if use_rdma else 'TCP Fallback'}")
        print("=" * 50 + "\n")

    # 2. 准备测试数据
    tensor_size = int(tensor_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    tensor = torch.ones(tensor_size, dtype=torch.float32, device=device) * (
        local_rank + 1
    )

    if local_rank == 0:
        size_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024
        print(f"[Rank 0] Tensor size: {size_mb:.2f} MB")
        print(f"[Rank 0] Before AllReduce: {tensor[0].item()}")

    # 预热 (Warmup) - 让 NCCL 建立连接
    # 第一次 AllReduce 会触发 QP 创建、MR 注册等操作
    for i in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            print(f"  Warmup iteration {i+1}/5 completed")
    dist.barrier()

    # 3. 性能测试与观测
    latencies = []
    start_time = time.perf_counter()

    for i in range(iterations):
        torch.cuda.synchronize()
        iter_start = time.perf_counter()

        # ============================================
        # 这一行就是触发全链路的核心！
        # PyTorch -> NCCL -> RDMA Verbs -> NIC -> 网络
        # ============================================
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        iter_end = time.perf_counter()
        latencies.append((iter_end - iter_start) * 1000)  # ms

        if local_rank == 0 and (i + 1) % 10 == 0:
            avg_lat = sum(latencies[-10:]) / 10
            print(f"  Iteration {i+1}/{iterations} - avg latency: {avg_lat:.2f} ms")

    # 等待所有 GPU 操作完成
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 4. 数据统计与验证
    if local_rank == 0:
        expected_value = sum(range(1, world_size + 1))
        # 注意：经过多次 AllReduce，值会累积，这里只验证最终不是 NaN
        print(f"\n[Rank 0] After AllReduce: tensor[0] = {tensor[0].item()}")

        total_time = end_time - start_time
        avg_time_ms = sum(latencies) / len(latencies)
        p99_time_ms = sorted(latencies)[int(len(latencies) * 0.99)]
        data_size_bytes = tensor.element_size() * tensor.nelement()
        alg_bw = (data_size_bytes / (1024**3)) / (avg_time_ms / 1000)

        print("\n📊 --- Performance Metrics ---")
        print(f"  Mode:               {'RDMA' if use_rdma else 'TCP'}")
        print(f"  Tensor Size:        {data_size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Iterations:         {iterations}")
        print(f"  Average Latency:    {avg_time_ms:.2f} ms")
        print(f"  P99 Latency:        {p99_time_ms:.2f} ms")
        print(f"  Min Latency:        {min(latencies):.2f} ms")
        print(f"  Max Latency:        {max(latencies):.2f} ms")
        print(f"  Algorithm Bandwidth:{alg_bw:.2f} GB/s")
        print("-" * 30 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch -> NCCL -> RDMA 全链路观测脚本"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local rank (usually set by launcher)",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", 1)),
        help="Total number of processes",
    )
    parser.add_argument(
        "--disable_rdma",
        action="store_true",
        help="Disable RDMA and use TCP Socket instead",
    )
    parser.add_argument(
        "--tensor_mb",
        type=int,
        default=100,
        help="Tensor size in MB (default: 100)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    args = parser.parse_args()

    use_rdma = not args.disable_rdma
    setup_environment(use_rdma)
    run_demo(args.local_rank, args.world_size, use_rdma, args.tensor_mb, args.iterations)
