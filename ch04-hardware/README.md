# 第四章：硬件结合 - GPU 与 RDMA 支持

## 本章简介

本章讲解现代 HPC 环境中 MPI 与 GPU、RDMA 技术的深度结合，是高性能计算的关键内容。

## GPU 支持 (CUDA-aware MPI)

### 传统模式：GPU → CPU → 网络 → CPU → GPU

```
GPU A: 计算 → 复制到 CPU 内存 → 网卡发送 
                    ↓
网卡接收 → 复制到 CPU 内存 → GPU B: 计算
```

问题：多次 CPU 内存复制，延迟高，CPU 成为瓶颈

### CUDA-aware MPI 模式：GPU 直接通信

```
GPU A ───────────────────────────────────── GPU B
   │                                            │
   └────────── RDMA / 网络直接传输 ─────────────┘
```

数据直接从 GPU 显存发送到网络，**绕过 CPU**！

### 环境检查

```bash
# 查看 OpenMPI 是否支持 CUDA
ompi_info --all | grep -i cuda

# 查看 CUDA 设备
nvidia-smi
```

### CUDA-aware MPI 示例

```c
#include <mpi.h>
#include <cuda_runtime.h>

int main() {
    MPI_Init(&argc, &argv);
    
    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * 100);
    
    // 直接发送 GPU 指针！
    MPI_Send(d_data, 100, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    
    cudaFree(d_data);
    MPI_Finalize();
}
```

## RDMA 基础

### 什么是 RDMA？

**RDMA** = **Remote Direct Memory Access**（远程直接内存访问）

传统网络：数据 → 内核缓冲区 → 用户缓冲区
RDMA：数据 → 直接写入目标内存（绕过内核）

### RDMA 三大优势

| 优势 | 说明 | 收益 |
|------|------|------|
| **零拷贝** | 数据直接从发送方到接收方 | 减少 CPU 开销 |
| **内核旁路** | 绕过操作系统内核 | 降低延迟 |
| **CPU 卸载** | CPU 不参与数据传输 | 提高并行度 |

### 性能对比

| 指标 | 传统 TCP | RDMA |
|------|----------|------|
| 延迟 | 10-20 μs | 1-2 μs |
| CPU 占用 | 高 | 极低 |
| 带宽 | 10-25 Gbps | 100+ Gbps |

### RDMA 协议

- **InfiniBand**：原始 RDMA，性能最强
- **RoCE** (RDMA over Converged Ethernet)：以太网上的 RDMA
- **iWARP**：基于 TCP 的 RDMA

### 传输类型

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| **RC** (Reliable Connection) | 可靠、面向连接 | 最重要场景 |
| **UC** (Unreliable Connection) | 不可靠、面向连接 | 性能优先 |
| **UD** (Unreliable Datagram) | 不可靠、无连接 | 多对多通信 |

### 核心概念

| 概念 | 说明 |
|------|------|
| **QP** (Queue Pair) | 队列对，通信基本单元 |
| **MR** (Memory Region) | 注册的内存区域 |
| **CQ** (Completion Queue) | 完成队列 |
| **PD** (Protection Domain) | 保护域 |

## GPU + RDMA 协同

### 完整数据流

```
应用 (GPU) → CUDA 显存 → RDMA 网卡 → 网络 → RDMA 网卡 → CUDA 显存 → 应用 (GPU)
```

### 性能收益

- **延迟降低**：GPU 直接通信，减少拷贝
- **带宽提升**：RDMA 提供更高带宽
- **CPU 卸载**：CPU 不参与数据传输

### 环境配置

```bash
# 查看 RDMA 设备
ibstat

# 测试 RDMA 带宽
ib_send_bw -d mlx5_0 -a
```

## 概念澄清：MPI vs RDMA

| 层次 | 技术 | 作用 |
|------|------|------|
| 编程接口 | MPI | 并行编程标准 |
| 通信层 | OpenMPI | 消息传递实现 |
| 传输层 | RDMA | 硬件传输 |

**MPI 可以使用 RDMA 作为传输层**，这就是为什么现代 MPI 这么快。

## 示例代码

本章配套示例在 `04-hardware/` 目录：

```bash
cd ch04-hardware/04-hardware

# CUDA-aware 示例
mpicc -o cuda_aware cuda_aware.cu -lcudart

# RDMA 示例
make
./rdma_write_server &
./rdma_write_client
```

- `cuda_aware.cu` - CUDA-aware MPI
- `rdma_write_server.c` / `rdma_write_client.c` - RDMA Write

## 本章测验

- [ ] 理解 CUDA-aware MPI 的优势
- [ ] 掌握 RDMA 核心概念
- [ ] 理解 GPU + RDMA 协同
- [ ] 能在环境中验证 GPU/RDMA

## 下一步

学完本章后，进入 [第五章：上层栈结合](./ch05-stack/README.md) 学习 NCCL 与 PyTorch 的结合。
