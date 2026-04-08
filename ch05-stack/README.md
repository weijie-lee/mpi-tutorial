# 第五章：上层栈结合 - NCCL 与 PyTorch

## 本章简介

本章讲解 MPI 在深度学习中的应用，掌握 NCCL 与 PyTorch 分布式训练的结合。

## 为什么深度学习需要分布式？

### 单卡训练的瓶颈

- **显存有限**：大模型装不下
- **计算量大**：训练时间太长
- **数据海量**：ImageNet 有 1400 万张图

### 分布式训练解决方案

- **数据并行**：多张卡处理不同数据
- **模型并行**：不同卡负责不同层
- **流水线并行**：模型分阶段并行

## MPI vs NCCL

### 定位差异

| 特性 | MPI | NCCL |
|------|-----|------|
| **定位** | 通用并行通信 | GPU 专用集合通信 |
| **适用范围** | CPU + GPU | 仅 GPU |
| **通信优化** | 通用 | 针对 GPU 深度优化 |
| **典型场景** | 超算、科学计算 | 深度学习训练 |

### 协同关系

```
┌─────────────────────────────────────────────┐
│              应用层 (PyTorch)                │
├─────────────────────────────────────────────┤
│         通信层 (NCCL / MPI)                  │
├─────────────────────────────────────────────┤
│         传输层 (RDMA / TCP)                  │
└─────────────────────────────────────────────┘
```

**NCCL 专注于 GPU 间的高效通信**，而 **MPI 负责进程管理和跨节点通信**。

## NCCL 核心原语

### 常用集合通信

| NCCL 原语 | 功能 | 类似 MPI |
|-----------|------|----------|
| `ncclAllReduce` | 全归约 | MPI_Allreduce |
| `ncclBroadcast` | 广播 | MPI_Bcast |
| `ncclAllGather` | 全收集 | MPI_Allgather |
| `ncclReduceScatter` | 归约后散射 | MPI_Reduce_scatter |
| `ncclSend/Recv` | 点对点 | MPI_Send/Recv |

### AllReduce 详解

深度学习中最常用的操作：**每个 GPU 上的梯度求和**：

```
GPU 0: grad_0 = 1.5
GPU 1: grad_1 = 2.3
GPU 2: grad_2 = 0.8
GPU 3: grad_3 = 1.2

AllReduce 后：
GPU 0: sum = 5.8
GPU 1: sum = 5.8
GPU 2: sum = 5.8
GPU 3: sum = 5.8
```

## PyTorch 分布式数据并行 (DDP)

### 什么是 DDP？

**DDP** = **DistributedDataParallel**

PyTorch 官方分布式训练方案，自动处理：
- 梯度同步
- 模型参数同步
- 负载均衡

### DDP 工作原理

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 初始化进程组
dist.init_process_group(backend='nccl')

# 2. 创建模型并移到当前 GPU
model = MyModel().cuda(rank)

# 3. 包装为 DDP
model = DDP(model, device_ids=[local_rank])

# 4. 训练循环
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()  # DDP 自动 AllReduce
    optimizer.step()
```

### 关键环境变量

| 变量 | 说明 |
|------|------|
| `RANK` | 全局进程 ID |
| `WORLD_SIZE` | 总进程数 |
| `LOCAL_RANK` | 当前节点内的进程 ID |
| `MASTER_ADDR` | 主节点地址 |
| `MASTER_PORT` | 主节点端口 |

## MPI 在深度学习中的角色

### 启动分布式训练

```bash
mpirun -np 4 \
    -H server1,server2,server3,server4 \
    -x MASTER_ADDR=server1 \
    python train.py
```

### PyTorch + MPI + NCCL 栈

```
┌─────────────────────────────────────────┐
│  PyTorch DDP (梯度同步、参数更新)         │
├─────────────────────────────────────────┤
│  NCCL (GPU 间 AllReduce/Broadcast)      │
├─────────────────────────────────────────┤
│  MPI (进程管理、跨节点通信)               │
├─────────────────────────────────────────┤
│  RDMA / TCP (实际网络传输)               │
└─────────────────────────────────────────┘
```

## 分布式训练模式

### 数据并行 (Data Parallel)

每个 GPU 有一份完整模型，处理不同数据：

```
GPU 0: Model + Batch 1 → grad_0
GPU 1: Model + Batch 2 → grad_1
GPU 2: Model + Batch 3 → grad_2
GPU 3: Model + Batch 4 → grad_3
         ↓ AllReduce
         更新模型参数
```

**优点**：简单、扩展性好
**缺点**：每卡都要装下完整模型

### 模型并行 (Model Parallel)

不同 GPU 负责模型不同部分：

```
Layer 0-1: GPU 0 → GPU 1: Layer 2-3 → GPU 2: Layer 4-5 → 输出
```

**优点**：可以训练超大模型
**缺点**：通信开销大

## 示例代码

本章配套示例在 `05-pytorch/` 目录：

```bash
cd ch05-stack/05-pytorch

# NCCL 基本示例
python nccl_allreduce.py

# MPI 初始化
python mpi_basic.py

# DDP 示例
python pytorch_ddp_mpi.py
```

- `mpi_basic.py` - MPI 初始化示例
- `nccl_allreduce.cu` - NCCL AllReduce
- `pytorch_ddp_mpi.py` - DDP 训练示例

## 本章测验

- [ ] 理解 MPI 与 NCCL 的定位差异
- [ ] 掌握 NCCL 核心集合通信原语
- [ ] 学会使用 PyTorch DDP
- [ ] 理解数据并行原理

## 下一步

学完本章后，进入 [第六章：完整应用](./ch06-applications/README.md) 学习二维 Jacobi 迭代实战。
