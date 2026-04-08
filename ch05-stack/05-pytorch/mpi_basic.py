#!/usr/bin/env python3
"""
examples/05-pytorch/mpi_basic.py
PyTorch 使用 MPI 后端基础示例
演示如何初始化 MPI，获取 rank 和 world size
编译：不需要编译，Python 脚本
运行：mpirun -np 4 python mpi_basic.py
"""

import torch
import torch.distributed as dist

# --------------------------
# 使用 MPI 后端初始化分布式环境
# 如果是用 MPI 启动多进程，PyTorch 会自动从 MPI 获取 rank/world size
# 不需要手动设置 MASTER_ADDR 和 MASTER_PORT
dist.init_process_group(backend='mpi')

# --------------------------
# 获取当前进程 rank 和总共多少进程
# rank: 当前进程编号，从 0 开始
# world_size: 总共多少进程
rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"Hello from rank {rank}/{world_size} using MPI backend")

# --------------------------
# 示例：all-reduce 一个 tensor
tensor = torch.tensor([rank], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"Rank {rank}: before allreduce tensor = {tensor.item()}")

# allreduce 求和，所有进程得到相同结果：sum(rank)
dist.all_reduce(tensor)

print(f"Rank {rank}: after allreduce tensor = {tensor.item()}")
expected = (world_size - 1) * world_size / 2
print(f"Expected sum = {expected}, difference = {tensor.item() - expected}")

# 销毁进程组
dist.destroy_process_group()
