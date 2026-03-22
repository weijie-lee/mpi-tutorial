#!/usr/bin/env python3
"""
examples/05-pytorch/pytorch_ddp_mpi.py
完整 PyTorch DDP 分布式训练示例，使用 MPI 启动 + NCCL 通信
运行（双节点，每节点4GPU）：
    mpirun -np 8 -H node1:4,node2:4 python pytorch_ddp_mpi.py
在 SLURM 上：
    srun -N 2 --ntasks-per-node=4 python pytorch_ddp_mpi.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader


def setup():
    """
    初始化分布式训练环境
    - 使用 NCCL 后端（GPU 通信最优）
    - 每个进程绑定到对应 GPU
    """
    # 初始化进程组，NCCL 后端，MPI 启动会自动获取信息
    dist.init_process_group(backend='nccl')
    
    # 获取当前进程 rank 和世界大小
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 每个进程绑定到对应的 GPU
    # 假设每个节点 GPU 数量是 ngpus，rank % ngpus 就是当前节点要绑定的 GPU id
    ngpus_per_node = torch.cuda.device_count()
    device_id = rank % ngpus_per_node
    torch.cuda.set_device(device_id)
    
    print(f"[rank {rank}/{world_size}] bound to GPU {device_id}")
    return rank


def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()


class SimpleDataset(Dataset):
    """简单的随机数据集示例"""
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 随机输入，随机标签
        x = torch.randn(self.dim)
        y = torch.tensor([1.0 if x.sum() > 0 else 0.0])
        return x, y


def main():
    # 初始化分布式环境
    rank = setup()
    # 获取当前设备
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device(f'cuda:{rank % ngpus_per_node}')

    # --------------------------
    # 创建模型，包装成 DDP
    # 线性层：输入 10 维，输出 1 维
    model = nn.Linear(10, 1).to(device)
    # DDP 包装模型：自动处理梯度同步
    # 每个参数梯度计算完后，DDP 自动做 NCCL AllReduce 同步梯度
    ddp_model = DDP(model, device_ids=[rank % ngpus_per_node])

    # --------------------------
    # 数据加载
    # 每个进程只处理自己分片的数据，DDP 自动汇总梯度
    dataset = SimpleDataset(1000, 10)
    # 用 DistributedSampler 自动分片，每个进程只拿到自己那部分
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # --------------------------
    # 训练一步
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        # 前向
        output = ddp_model(x)
        loss = criterion(output, y)
        
        # 反向
        optimizer.zero_grad()
        loss.backward()  # DDP 在这里自动调用 NCCL AllReduce 同步梯度！
        optimizer.step()

        if batch_idx % 10 == 0 and rank == 0:
            print(f"[rank 0] batch {batch_idx}, loss = {loss.item():.4f}")

    if rank == 0:
        print("Training step done!")

    # --------------------------
    # 验证模型权重
    # 所有进程权重应该一致，因为 DDP 每次迭代都同步梯度
    if dist.get_world_size() > 1:
        # 检查第一个参数
        param = next(ddp_model.parameters())
        # rank 0 广播权重给所有进程验证
        if rank == 0:
            first_weight = param.data[0, 0].clone()
        else:
            first_weight = torch.zeros_like(param.data[0, 0])
        dist.broadcast(first_weight, 0)
        if rank != 0:
            diff = abs(param.data[0, 0] - first_weight)
            if diff < 1e-6:
                print(f"[rank {rank}] ✓ Weights are synchronized")
            else:
                print(f"[rank {rank}] ✗ Weights are NOT synchronized")

    cleanup()


if __name__ == "__main__":
    main()
