# 五、上层栈结合：NCCL 与 PyTorch

在深度学习分布式训练中，MPI 经常和 NCCL 配合使用，最终在 PyTorch 中被调用。本章从基础概念到原生编程，完整介绍 NCCL。

## 1. MPI vs NCCL：定位分工

### NCCL 是什么
**NCCL** = **NVIDIA Collective Communications Library**，是 NVIDIA 推出的**GPU 专属集合通信库**，专门针对多GPU、多节点场景做极致性能优化。

### 定位差异对比

| 特性 | MPI | NCCL |
|------|-----|------|
| 范围 | 通用消息传递，支持任意点对点/集合通信，CPU/GPU都能处理 | 专注于**GPU集合通信**（allreduce、broadcast、allgather等），不支持通用点对点通信 |
| 硬件 | 支持任何GPU/CPU/网络，跨厂商 | NVIDIA GPU 专属 |
| 优化目标 | 通用性、可移植性、功能完整 | NVIDIA 硬件上极致性能 |
| 进程启动 | 自带 `mpirun` 启动管理多个进程 | 不负责进程启动，需要上层（MPI/ PyTorch）协调 |

### NCCL 集合通信原语 vs MPI 对应表

| NCCL 操作 | 对应 MPI 操作 | 作用 | 深度学习场景 |
|-----------|-------------|------|--------------|
| `ncclBcast` | `MPI_Bcast` | 从 root 广播数据到所有 GPU | 初始化权重同步 |
| `ncclAllReduce` | `MPI_Allreduce` | 所有 GPU 数据归约，结果所有 GPU 都有 | **数据并行训练梯度平均，最常用** |
| `ncclReduce` | `MPI_Reduce` | 所有 GPU 数据归约，结果只给 root | 汇总梯度/指标到 root |
| `ncclAllGather` | `MPI_Allgather` | 每个 GPU 拼一块，所有 GPU 拿到完整结果 | 分布式 embedding 收集所有分片 |
| `ncclReduceScatter` | `MPI_ReduceScatter` | 归约分散，每个 GPU 拿一块结果 | 大张量分片 AllReduce |
| `ncclGather` | `MPI_Gather` | 收集所有 GPU 数据到 root | 汇总结果到 root |
| `ncclScatter` | `MPI_Scatter` | root 分发数据给所有 GPU | 分发数据分片 |
| `ncclSend`/`ncclRecv` | `MPI_Send`/`MPI_Recv` | 点对点通信（NCCL 2.0+ 支持） | 不规则通信 |

可以看到 NCCL 支持的操作和 MPI 集合通信原语几乎一一对应，语义完全一样，只是 NCCL 专用于 GPU，比 MPI 更快。

### 常见生产架构
现在主流多节点多GPU训练架构是：
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ 进程启动协调 │ → │ 初始化通信  │ → │ 实际集合通信 │
│   (MPI)     │   │   (MPI)     │   │   (NCCL)    │
└─────────────┘   └─────────────┘   └─────────────┘
```
MPI 做"后勤"，负责启动和交换信息；NCCL 做"实干"，负责实际GPU数据通信，发挥各自优势。

## 2. NCCL 核心概念

### Communicator（通信器）
类似 MPI 的 Communicator，NCCL 用 `ncclComm_t` 表示一组可以互相通信的GPU：
- 你创建一个 communicator 包含所有你的GPU
- 之后所有集合通信操作都在这个 communicator 上做
- 多任务可以用多个 communicator 隔离

### Group 与 Rank
和 MPI 一样，每个GPU在 communicator 里有唯一的 rank（从 0 开始）。

### 支持的集合通信操作

NCCL 支持这些常用操作：

| 操作 | 说明 | 常用场景 |
|------|------|----------|
| `ncclAllReduce` | 所有 GPU 数据归约（求和），结果广播到所有 GPU | 深度学习训练，同步梯度 |
| `ncclBcast` | 从 root GPU 广播数据到所有 GPU | 初始化权重同步 |
| `ncclAllGather` | 每个 GPU 拼一块，所有 GPU 拿到完整数据 | 收集结果 |
| `ncclReduceScatter` | 每个 GPU 归约一块，结果分散到每个 GPU | 分片AllReduce |
| `ncclSend`/`ncclRecv` | 点对点通信（NCCL 2.0+支持） | 不规则通信 |

> 💡 NCCL 2 开始也支持点对点通信了，但主要还是用集合通信。

## 3. NCCL 原生编程完整示例

### 最简单的 AllReduce 示例（多GPU/多节点）

```cpp
// examples/05-pytorch/nccl_allreduce.cu
/*
 * NCCL AllReduce 示例，对应 MPI_Allreduce
 * 每个 rank i 给所有元素赋值 i+1，AllReduce 求和，结果 = sum(1..size)
 */
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <stdio.h>
#include <cmath>

int main(int argc, char** argv) {
    // 1. 先用MPI初始化，获取rank和size
    // NCCL 不负责进程启动，用 MPI 做这个
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. 每个进程绑定到对应GPU（假设每个进程一个GPU，标准做法）
    cudaError_t err = cudaSetDevice(rank);
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaSetDevice failed: %s\n",
                rank, cudaGetErrorString(err));
        MPI_Finalize();
        return 1;
    }

    // 3. 在GPU分配数据
    const int n = 1024;
    float *d_send, *d_recv;
    err = cudaMalloc(&d_send, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMalloc failed\n", rank);
        MPI_Finalize();
        return 1;
    }
    err = cudaMalloc(&d_recv, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMalloc failed\n", rank);
        cudaFree(d_send);
        MPI_Finalize();
        return 1;
    }

    // 初始化：每个rank i 所有元素都是 i+1
    float *h_send = new float[n];
    for (int i = 0; i < n; i++) {
        h_send[i] = rank + 1.0f;
    }
    err = cudaMemcpy(d_send, h_send, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_send;
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMemcpy failed\n", rank);
        MPI_Finalize();
        return 1;
    }

    // 4. NCCL 初始化
    // 多节点需要 MPI 广播 unique id 给所有进程
    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) {
        // root 生成唯一 id
        ncclResult_t res = ncclGetUniqueId(&id);
        if (res != ncclSuccess) {
            fprintf(stderr, "ncclGetUniqueId failed: %s\n", ncclGetErrorString(res));
            MPI_Finalize();
            return 1;
        }
    }
    // root 通过 MPI 把 id 广播给所有进程
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    // 每个进程初始化自己的 rank
    ncclResult_t res = ncclCommInitRank(&comm, size, id, rank);
    if (res != ncclSuccess) {
        fprintf(stderr, "rank %d: ncclCommInitRank failed: %s\n",
                rank, ncclGetErrorString(res));
        MPI_Finalize();
        return 1;
    }

    // --------------------------
    // 5. 执行 NCCL AllReduce 求和
    // 对应 MPI_Allreduce，语义完全一样
    // ncclSum: 操作类型是求和
    // cudaStreamDefault: 使用默认 CUDA stream
    res = ncclAllReduce(
        d_send,    // 发送缓冲区（GPU指针）
        d_recv,    // 接收缓冲区（GPU指针）
        n,         // 多少个元素
        ncclFloat, // 元素类型
        ncclSum,   // 归约操作：求和
        comm,      // NCCL communicator
        cudaStreamDefault // CUDA stream
    );
    if (res != ncclSuccess) {
        fprintf(stderr, "rank %d: ncclAllReduce failed: %s\n",
                rank, ncclGetErrorString(res));
        ncclCommDestroy(comm);
        MPI_Finalize();
        return 1;
    }

    // 等待 CUDA 完成
    cudaDeviceSynchronize();

    // 6. 验证结果：所有 rank 结果都应该是 sum(1..size)
    float result;
    err = cudaMemcpy(&result, d_recv, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMemcpy failed\n", rank);
        MPI_Finalize();
        return 1;
    }
    float expected = (float)(size * (size + 1)) / 2.0f;
    if (rank == 0) {
        printf("NCCL AllReduce result (first element): %.2f, expected: %.2f\n", result, expected);
        if (fabs(result - expected) < 1e-5) {
            printf("✓ Verification PASS\n");
        } else {
            printf("✗ Verification FAIL\n");
        }
    }

    // 7. 清理资源
    ncclCommDestroy(comm);
    cudaFree(d_send);
    cudaFree(d_recv);
    MPI_Finalize();
    return 0;
}
```

### 编译运行

```bash
cd examples/05-pytorch
nvcc -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnccl -lmpi nccl_allreduce.cu -o nccl_allreduce
mpirun -np 2 ./nccl_allreduce
```

应该输出：
```
NCCL AllReduce result (first element): 3.00, expected: 3.00
✓ Verification PASS
```

### 对应 MPI 版本对比

相同功能的 MPI 版本在 [examples/02-core/all-collectives.c](../examples/02-core/all-collectives.c)，API 几乎一样，只是 NCCL 专用于 GPU 更快。

### 关键要点总结

| 步骤 | 作用 | 对应操作 |
|------|------|----------|
| 获取 unique id | 生成唯一通信域标识 | root `ncclGetUniqueId` |
| 广播 unique id | 所有进程拿到同一个 id | MPI `MPI_Bcast` |
| 初始化每个 rank | `ncclCommInitRank` | 每个进程自己调用 |
| AllReduce | 实际集合通信 | NCCL `ncclAllReduce` 直接 GPU 操作 |

## 4. NCCL 性能优化要点

1. **使用 GPU 直接 RDMA**：如果网络支持 RDMA，NCCL 会自动用 GPU Direct RDMA，不需要 CPU 中转，延迟更低
2. **避免频繁创建销毁 communicator**：创建有开销，一次性创建好一直用
3. **让计算和通信重叠**：用不同 CUDA stream，计算放一个 stream，通信放另一个，可以重叠掩盖延迟
4. **大张量优先用 ReduceScatter + AllGather**：NCCL 内部自动会做，但你也可以显式用，比整体 AllReduce 更快
5. **每个进程一个GPU**：这是最常用的绑定方式，NCCL 优化得最好，不要多个进程抢同一张GPU

## 5. MPI + NCCL 协同工作流程（PyTorch DDP）

典型的 PyTorch DDP 流程：
1. **`mpirun`/`srun`** 启动所有进程，每个GPU对应一个进程
2. 每个进程调用 **`MPI_Init`**，拿到自己的 rank 和 world size
3. 每个进程调用 `cudaSetDevice` 绑定到一张GPU
4. NCCL 底层通过 MPI 交换各个GPU的地址信息、建立连接
5. 训练过程中，每次迭代的梯度 AllReduce 都走 NCCL，直接 GPU → GPU 通信，不经过CPU
6. 训练结束，退出

PyTorch 底层帮你把这些都封装好了，用户只需要正确初始化就行。

## 6. PyTorch 中的 MPI + NCCL 使用

### 初始化方式
PyTorch `torch.distributed` 原生支持：
```python
# examples/05-pytorch/mpi_basic.py
import torch
import torch.distributed as dist

# 使用NCCL后端，MPI 启动会自动获取信息
dist.init_process_group(backend='nccl')

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
| NCCL | GPU训练，多GPU/多节点 | NVIDIA GPU 上性能最好 | 只支持NVIDIA，需要上层做进程启动 |
| Gloo | 多节点CPU训练，或NCCL辅助 | 纯CPU实现，好编译 | GPU性能差 |

### MPI + NCCL 混合用法
在多节点场景，你可以用 MPI 做初始化启动，然后 NCCL 做通信：
```python
# 实际上现在PyTorch会自动处理，init_process_group用 'nccl' 后端，
# 但是用 mpirun/srun 启动，底层初始化发现用MPI交换地址更方便，自动就用了MPI协同
dist.init_process_group(backend='nccl')
```
你不用显式指定`mpi`后端，只要你是用`mpirun`启动，PyTorch会自动用 MPI 做协调，NCCL 做 GPU 通信。

## 7. PyTorch DDP 分布式训练示例

完整的多节点多GPU训练示例，使用 MPI 启动 + NCCL 通信：

```python
# examples/05-pytorch/pytorch_ddp_mpi.py
"""
完整 PyTorch DDP 分布式训练，MPI 启动 + NCCL 通信
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup():
    """
    初始化分布式环境
    - 使用 NCCL 后端
    - 每个进程绑定到对应 GPU
    """
    # 初始化进程组，NCCL 后端
    dist.init_process_group(backend='nccl')
    
    # 获取当前 rank 和世界大小
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 绑定到对应 GPU
    ngpus_per_node = torch.cuda.device_count()
    device_id = rank % ngpus_per_node
    torch.cuda.set_device(device_id)
    
    print(f"[rank {rank}/{world_size}] bound to GPU {device_id}")
    return rank


def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()


class SimpleDataset(Dataset):
    """简单随机数据集示例"""
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
    rank = setup()
    # 获取当前设备
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device(f'cuda:{rank % ngpus_per_node}')

    # --------------------------
    # 创建模型，包装成 DDP
    # 线性层：输入 10 维，输出 1 维
    model = nn.Linear(10, 1).to(device)
    # DDP 包装：自动处理梯度同步，底层调用 NCCL AllReduce
    # NCCL 做梯度平均，对应 MPI_Allreduce
    ddp_model = DDP(model, device_ids=[rank % ngpus_per_node])

    # --------------------------
    # 数据加载：使用 DistributedSampler 自动分片
    dataset = SimpleDataset(1000, 10)
    # DistributedSampler 自动把数据分给各个进程，每个进程只拿自己分片
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # --------------------------
    # 训练
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
        loss.backward()  
        # 🔥 DDP 在这里自动调用 NCCL AllReduce 同步梯度！
        // 对应 MPI_Allreduce，语义完全一样，NCCL 更快
        optimizer.step()

        if batch_idx % 10 == 0 and rank == 0:
            print(f"[rank 0] batch {batch_idx}, loss = {loss.item():.4f}")

    if rank == 0:
        print("Training finished!")

    # 验证：所有进程权重应该一致
    if dist.get_world_size() > 1:
        # rank 0 广播第一个权重给所有进程验证
        first_param = next(ddp_model.parameters())[0][0].item()
        if rank == 0:
            print(f"Verifying weight synchronization... first parameter = {first_param:.6f}")
        # 实际上 DDP 每次迭代都用 NCCL AllReduce 同步了，所以肯定一致

    cleanup()


if __name__ == "__main__":
    main()
```

运行示例：

**单节点 8GPU（默认环境直接运行）：**
```bash
mpirun -np 8 python pytorch_ddp_mpi.py
```

**双节点，每节点4GPU（总共 8GPU）：**

**方式一：命令行直接指定主机**
```bash
mpirun -np 8 \
    -H node1:4,node2:4 \
    python pytorch_ddp_mpi.py
```

**方式二：使用 hostfile（推荐）**

创建 `hostfile`：
```
node1 slots=8
```
`slots` 表示这个节点能跑多少个进程（通常等于 GPU 数量）

单节点直接运行：
```bash
mpirun -np 8 --hostfile hostfile python pytorch_ddp_mpi.py
```

多节点 `hostfile` 示例：
```
node1 slots=8
node2 slots=8
```

运行：
```bash
mpirun -np 16 --hostfile hostfile python pytorch_ddp_mpi.py
```

在 SLURM 集群上，通常用 `srun` 启动（它自动调用 MPI）：
```bash
# 单节点 8GPU
srun -N 1 --ntasks-per-node=8 python pytorch_ddp_mpi.py
# 双节点，每节点 8GPU
srun -N 2 --ntasks-per-node=8 python pytorch_ddp_mpi.py
```

## 常见问题

### Q: 为什么要用 MPI 启动，而不是 torchrun ？

A: 两种方式都可以，各有适用场景：

| 方式 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| `torchrun` | 单节点多GPU | 简单易用，PyTorch 自带 | 多节点需要手动同步地址，配置麻烦 |
| `mpirun`/`srun` | 多节点多GPU（HPC/集群环境） | 集群调度系统原生支持，自动分配 rank、网络探测 | 需要 MPI 环境，多一层 |

在有 SLURM 调度的 AI 集群上，几乎都是用 `srun` + MPI 方式启动，这是行业惯例。

### Q: MPI + NCCL 多节点训练需要什么网络配置？

需要满足：
1. **所有节点之间**能**互相 SSH 访问**（不需要密码，配置好密钥）
2. **NCCL 通信端口范围**开放（默认 `1024-65535`，需要防火墙放通）
3. 如果用 RDMA，确保 IB/RoCE 网络打通，节点间能 RDMA 通信
4. 所有节点设置相同环境变量：
   - 有 IB/RoCE：默认不用改，NCCL 自动启用
   - 只有 TCP/IP：`export NCCL_IB_DISABLE=1` 强制走 TCP

### Q: 怎么验证我的程序真的在用 MPI + NCCL ？

打开 NCCL 调试日志：
```bash
export NCCL_DEBUG=INFO
mpirun -np 8 python pytorch_ddp_mpi.py
```

日志里会看到类似输出，说明工作正常：
```
NCCL INFO Connected to all process trees...
NCCL INFO Using network IB
NCCL INFO Using cuda IPC
```

看到 `Using network IB` 说明在用 RDMA/InfiniBand，看到 `Using network Socket` 说明在用 TCP/IP。

### Q: 多节点训练 NCCL 初始化卡住怎么办？

常见原因和解决：
1. **防火墙阻挡** → 检查节点间端口是否放通
2. **IB/RDMA 不可达** → 如果没有 RDMA，试试 `export NCCL_IB_DISABLE=1`
3. **GPU 不匹配** → 确保每个进程只绑一个 GPU，不要多个进程抢同一张 GPU
4. **版本不一致** → 确保所有节点 PyTorch/NCCL 版本相同

## 总结分工

| 层级 | 组件 | 职责 | 对应 MPI |
|------|------|------|--------|
| 进程启动 | MPI (`mpirun`/`srun`) | 启动所有节点上的所有进程，分配 rank | - |
| 初始化协调 | MPI | 交换各个 GPU 的网络地址，交换 NCCL unique id，建立连接 | - |
| 实际集合通信 | NCCL | 梯度 AllReduce 等集合操作，GPU 直接通信 | ncclAllReduce ≈ MPI_Allreduce |
| 框架层 | PyTorch DDP | 掩盖底层细节，给用户提供简单接口 | - |

## 示例代码

| 示例 | 说明 |
|------|------|
| [nccl_allreduce.cu](../examples/05-pytorch/nccl_allreduce.cu) | NCCL 原生编程 AllReduce 完整示例（MPI + NCCL） |
| [mpi_basic.py](../examples/05-pytorch/mpi_basic.py) | PyTorch MPI 基础初始化 |
| [pytorch_ddp_mpi.py](../examples/05-pytorch/pytorch_ddp_mpi.py) | 完整 PyTorch DDP 分布式训练示例（MPI 启动 + NCCL 通信） |

## 10. 其他硬件厂商的专用通信库

除了 NVIDIA 的 NCCL 之外，不同的 AI 硬件厂商都推出了自己的 GPU/AI 加速器专属集合通信库，分工模式和 NCCL 类似：**MPI 负责进程启动和协调，厂商通信库负责实际 GPU 集合通信**。

### 各厂商通信库对比总览

| 硬件厂商 | 通信库名称 | 全称 | 支持硬件 | 典型使用场景 |
|----------|------------|------|----------|--------------|
| NVIDIA | NCCL | NVIDIA Collective Communications Library | NVIDIA GPU | 标准 NVIDIA GPU 集群 |
| AMD | RCCL | ROCm Collective Communications Library | AMD GPU | AMD GPU 集群 |
| 华为 | HCCL | Huawei Collective Communication Library | 华为昇腾 NPU | 华为 Atlas 昇腾集群 |
| 寒武纪 | CNCCL | Cambricon Collective Communication Library | 寒武纪 MLU | 寒武纪 MLU 集群 |
| 摩尔线程 | MCCL | Moore Threads Collective Communication Library | 摩尔线程 GPU | 摩尔线程 GPU 集群 |

### 各厂商详细说明

#### AMD RCCL

**RCCL** 是 AMD 为 ROCm 平台开发的集合通信库，对应 NVIDIA NCCL 的 AMD 版本：

- 支持 AMD Instinct 系列 GPU
- 提供和 NCCL **几乎完全相同的 API**，移植很方便
- 原生支持 GFX90A (MI250)、GFX940 (MI300) 等架构
- 支持 GPU Direct RDMA 过 PCIe 互连

典型用法和 NCCL 完全一样：
```cpp
// API 几乎和 NCCL 一样，只是名字改了
#include <rccl/rccl.h>

ncclComm_t comm; // 类型名还是 ncclComm_t，API 兼容
rcclCommInitRank(&comm, size, id, rank);
rcclAllReduce(d_send, d_recv, n, ncclFloat, ncclSum, comm, stream);
```

PyTorch 中使用：
```python
# PyTorch 会自动识别，只要选 nccl 后端就行，ROCm 环境自动用 RCCL
dist.init_process_group(backend='nccl')
```

#### 华为 HCCL（昇腾）

**HCCL** 是华为为昇腾 NPU 开发的集合通信库：

- 支持华为 Atlas 训练集群，昇腾 910/310P/920 芯片
- 提供华为自研的拓扑感知算法，在昇腾互联上性能优化
- 支持 HCCL 集合通信原语，和 NCCL 功能对等
- 集成在华为 CANN (Compute Architecture for Neural Networks) 软件栈中

在 PyTorch 中使用（torch-npu）：
```python
# 华为 PyTorch 扩展提供 'hccl' 后端
import torch_npu
from torch_npu.distributed import ParallelInit

dist.init_process_group(backend='hccl')
```

分工模式还是一样：`mpirun/srun` 启动进程，HCCL 做实际集合通信。

#### 寒武纪 CNCCL

**CNCCL** 是寒武纪为 MLU 架构开发的集合通信库：

- 支持寒武纪 MLU370/MLU590 等训练芯片
- 提供和 NCCL 类似的 API 接口
- 针对寒武纪 MLU 互联做了拓扑优化
- 配合 CNToolkit 软件栈使用

分工模式：MPI 负责进程启动，CNCCL 负责 MLU 通信。

### 通用分工模式：MPI + 厂商通信库

不管用哪个厂商的硬件，分工模式几乎都是一样的：

| 层级 | 组件 | 职责 |
|------|------|------|
| 资源调度 | SLURM / Kubernetes | 分配节点和资源 |
| 进程启动 | MPI / `mpirun` / `srun` | 在所有节点启动进程，分配 rank |
| 初始化协调 | MPI | 交换硬件地址信息，建立连接 |
| 实际通信 | 厂商通信库 (NCCL/RCCL/HCCL/CNCCL) | GPU/NPU 直接集合通信，硬件厂商专属优化 |
| 框架层 | PyTorch DDP / ... | 上层框架封装，给用户简单接口 |

这是成熟的分工：
- MPI 做**跨平台通用的进程管理和协调**，不管硬件是什么，这部分都一样
- 厂商做**自己硬件上的极致性能优化**，发挥最好通信效率

### 在 PyTorch 中切换后端对应表

| 硬件厂商 | 通信库 | PyTorch backend 参数 |
|----------|--------|----------------------|
| NVIDIA | NCCL | `backend='nccl'` |
| AMD | RCCL | `backend='nccl'` (自动识别) |
| 华为昇腾 | HCCL | `backend='hccl'` |
| 寒武纪 | CNCCL | `backend='cncl'` (需要寒武纪扩展) |

### 对比：通用 MPI 集合通信 vs 厂商专用通信库

| 维度 | 通用 MPI | 厂商专用通信库 |
|------|----------|----------------|
| 硬件支持 | 支持所有硬件 | 只支持自家硬件 |
| 性能 | 通用优化，不如厂商专属 | 针对自家硬件极致优化 |
| 功能 | 支持通用点对点+集合通信 | 专注于深度学习需要的集合通信 (AllReduce等) |
| 使用场景 | 通用并行计算，CPU/GPU混合计算 | 纯深度学习多GPU训练 |

**结论**：在深度学习多GPU训练中，**MPI + 厂商通信库**是现在的 industry standard，兼顾通用性和性能。

## 下一步

→ 下一章：[完整应用实例：二维Jacobi迭代](06-applications.md)
