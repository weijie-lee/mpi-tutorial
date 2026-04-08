# 第五章：上层栈结合 - NCCL 与 PyTorch

## 本章简介

本章讲解 MPI 在深度学习中的应用，掌握 NCCL 与 PyTorch 分布式训练的结合。

## 知识点

### MPI vs NCCL：定位分工
- **MPI**：通用消息传递接口，跨节点通信
- **NCCL (NVIDIA Collective Communications Library)**：GPU 专用集合通信库
- **定位差异**：NCCL 专注 GPU 间高效通信，MPI 负责进程管理

### NCCL 核心原语
- `ncclAllReduce` - 全归约
- `ncclBroadcast` - 广播
- `ncclAllGather` - 全收集
- `ncclReduceScatter` - 归约后散射
- `ncclSend` / `ncclRecv` - 点对点通信

### PyTorch 分布式数据并行 (DDP)
- **初始化**：使用 MPI 初始化 PyTorch 分布式环境
- **进程组**：每个进程对应一个 GPU
- **梯度同步**：AllReduce 同步梯度
- **混合精度**：FP16/BF16 训练

### 分布式训练模式
- **数据并行**：每个进程处理不同的数据 batch
- **模型并行**：模型不同层分布在不同 GPU
- **流水线并行**：模型按层分阶段并行

## 核心概念

### PyTorch + MPI + NCCL 栈
```
应用层：PyTorch DDP
    ↓
通信层：NCCL（GPU 间通信）
    ↓
进程管理：MPI（多进程启动与管理）
    ↓
网络层：RDMA / TCP
```

### DDP 初始化
```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
```

## 学习目标

1. 理解 MPI 与 NCCL 的定位差异
2. 掌握 NCCL 核心集合通信原语
3. 学会使用 PyTorch DDP 进行多 GPU 训练
4. 理解分布式训练中的数据并行原理

## 示例代码

本章配套示例在 `05-pytorch/` 目录：
- `ddp_init.py` - DDP 初始化
- `distributed_training.py` - 分布式训练示例
- `gradient_sync.py` - 梯度同步机制

## 下一步

学完本章后，进入 [第六章：完整应用](./ch06-applications/README.md) 学习二维 Jacobi 迭代实战。
