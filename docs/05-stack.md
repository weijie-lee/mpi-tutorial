# 五、上层栈结合：NCCL 与 PyTorch

在深度学习分布式训练中，MPI 经常和 NCCL 配合使用，最终在 PyTorch 中被调用。本章梳理它们之间的分工和配合方式。

## 1. MPI vs NCCL

### NCCL 是什么
**NCCL** = **NVIDIA Collective Communications Library**，是 NVIDIA 推出的GPU集合通信库，专门针对多GPU、多节点场景做极致优化。

### 定位差异对比

| 特性 | MPI | NCCL |
|------|-----|------|
| 范围 | 通用消息传递，支持任意点对点/集合通信，CPU/GPU都能处理 | 专注于GPU集合通信（allreduce、broadcast等），不做点对点 |
| 硬件 | 支持任何GPU/CPU/网络 | NVIDIA GPU 专属 |
| 优化目标 | 通用性、可移植性 | NVIDIA 硬件上极致性能 |
| 进程启动 | 自带 `mpirun` 启动管理多个进程 | 不负责进程启动，需要上层协调 |

### 常见生产架构
现在主流多节点多GPU训练架构是：
```
MPI 负责：进程启动、节点间协调、初始化通信
NCCL 负责：实际的GPU数据集合通信（AllReduce等）
```
MPI做"后勤"，NCCL做"实干"，发挥各自优势。

## 2. MPI + NCCL 协同工作流程

典型的 PyTorch DDP 流程：
1. **`mpirun`** 启动所有进程，每个节点一张/多张GPU，每个GPU对应一个进程
2. 每个进程调用 **`MPI_Init`**，拿到自己的 rank 和 world size
3. 每个进程调用 `cudaSetDevice` 绑定到一张GPU
4. NCCL 调用底层通过 MPI 交换各个GPU的地址信息、建立连接
5. 训练过程中，每次迭代的梯度allreduce都走NCCL，直接GPU → GPU通信，不经过CPU
6. 训练结束，退出

PyTorch 底层帮你把这些都封装好了，用户只需要正确初始化就行。

## 3. PyTorch 中的 MPI 使用

### 初始化方式
PyTorch `torch.distributed` 原生支持MPI后端：
```python
# examples/05-pytorch/mpi_basic.py
import torch
import torch.distributed as dist

# 使用MPI后端初始化
dist.init_process_group(backend='mpi')

# 获取rank和world size
rank = dist.get_rank()
world_size = dist.get_world_size()
```

运行方式：
```bash
mpirun -np 4 python mpi_basic.py
```

### 后端对比选择

| 后端 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| MPI | CPU训练，或多节点协调 | 成熟，支持任意网络，自带进程启动 | GPU通信性能不如NCCL |
| NCCL | GPU训练，多GPU/多节点 | NVIDIA GPU上性能最好 | 只支持NVIDIA，需要上层做进程启动 |
| Gloo | 多节点CPU训练，或NCCL辅助 | 纯CPU实现，好编译 | GPU性能差 |

### MPI + NCCL 混合用法
在多节点场景，你可以用MPI做初始化启动，然后NCCL做通信：
```python
# 实际上现在PyTorch会自动处理，init_process_group用 'nccl' 后端，
# 但是用 mpirun/srun 启动，底层初始化发现用MPI交换地址更方便，自动就用了MPI协同
dist.init_process_group(backend='nccl')
```
你不用显式指定`mpi`后端，只要你是用`mpirun`启动，PyTorch会自动用MPI做协调，NCCL做GPU通信。

## 4. PyTorch DDP 分布式训练示例

完整的多节点多GPU训练示例，使用 MPI 启动 + NCCL 通信：

```python
# examples/05-pytorch/pytorch_ddp_mpi.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

def setup():
    # 初始化进程组，用NCCL后端，MPI启动
    dist.init_process_group(backend='nccl')
    # 获取rank，绑定到对应GPU
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank

def main():
    rank = setup()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    # 创建模型，包装成DDP
    model = nn.Linear(10, 1).to(device)
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])

    # 这里省略数据加载...
    # 训练过程，每次反向传播后DDP自动做allreduce同步梯度
    output = ddp_model(torch.randn(32, 10).to(device))
    loss = output.sum()
    loss.backward()  # DDP在这里自动调用NCCL AllReduce

    if rank == 0:
        print(f"Training step done, loss: {loss.item()}")

    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

运行（双节点，每节点4GPU）：

**方式一：命令行直接指定主机**
```bash
mpirun -np 8 \
    -H node1:4,node2:4 \
    python pytorch_ddp_mpi.py
```

**方式二：使用 hostfile（推荐）**

创建 `hostfile`：
```
node1 slots=4
node2 slots=4
```
`slots` 表示这个节点能跑多少个进程（通常等于GPU数量）

运行：
```bash
mpirun -np 8 --hostfile hostfile python pytorch_ddp_mpi.py
```

在 SLURM 集群上，通常用 `srun` 启动（它会自动调用MPI）：
```bash
srun -N 2 --ntasks-per-node=4 python pytorch_ddp_mpi.py
```

## 常见问题

### Q: 为什么要用 MPI 启动，而不是 torchrun ？

A: 两种方式都可以，各有适用场景：

| 方式 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| `torchrun` | 单节点多GPU | 简单易用，PyTorch自带 | 多节点需要手动同步地址，配置麻烦 |
| `mpirun/srun` | 多节点多GPU（HPC/集群环境） | 集群调度系统原生支持，自动分配rank、做网络探测 | 需要MPI环境，多了一层 |

在有SLURM调度的AI集群上，几乎都是用 `srun` + MPI 方式启动，这是行业惯例。

### Q: MPI + NCCL 多节点训练需要什么网络配置？

需要满足：
1. **所有节点之间**能**互相SSH访问**（不需要密码，配置好密钥）
2. **NCCL 通信端口范围**开放（默认是 `1024-65535`，需要在防火墙放通）
3. 如果用RDMA，确保IB/RoCE网络打通，节点间能RDMA通信
4. 所有节点使用相同的`NCCL_IB_DISABLE`设置：
   - 有IB/RoCE：默认不用改（NCCL自动启用）
   - 只有TCP/IP：`export NCCL_IB_DISABLE=1` 强制走TCP

### Q: 怎么验证我的程序真的在用 MPI + NCCL ？

可以通过环境变量打开NCCL调试日志：

```bash
export NCCL_DEBUG=INFO
mpirun -np 8 python pytorch_ddp_mpi.py
```

日志里会看到类似这样的输出，说明工作正常：
```
NCCL INFO Connected to all process trees...
NCCL INFO Using network IB
NCCL INFO Using cuda IPC
```

如果看到 `Using network IB` 说明在用RDMA/InfiniBand，`Using network Socket` 说明在用TCP/IP。

### Q: 多节点训练发现NCCL初始化卡住怎么办？

常见原因和解决：

1. **防火墙阻挡** → 检查节点间端口是否放通
2. **IB/RDMA 不可达** → 如果没有RDMA，试试 `export NCCL_IB_DISABLE=1`
3. **GPU 不对齐** → 确保每个进程只绑定一个GPU，不要多个进程抢同一张GPU
4. **节点间版本不一致** → 确保所有节点PyTorch/NCCL版本相同

## 总结分工

| 层级 | 组件 | 职责 |
|------|------|------|
| 进程启动 | MPI (`mpirun`/`srun`) | 启动所有节点上的所有进程，分配rank |
| 初始化协调 | MPI | 交换各GPU的网络地址，建立连接 |
| 实际通信 | NCCL | 梯度Allreduce等集合操作，GPU直接通信 |
| 框架层 | PyTorch DDP | 掩盖底层细节，给用户提供简单接口 |

## 示例代码

- [mpi_basic.py](../examples/05-pytorch/mpi_basic.py) - PyTorch MPI 基础初始化
- [pytorch_ddp_mpi.py](../examples/05-pytorch/pytorch_ddp_mpi.py) - 完整 PyTorch DDP 分布式训练示例（MPI + NCCL）

## 下一步

→ 下一章：[完整应用实例：二维Jacobi迭代](06-applications.md)
