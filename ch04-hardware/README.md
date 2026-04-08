# 第四章：硬件结合 - GPU 与 RDMA 支持

## 本章简介

本章讲解现代 HPC 环境中 MPI 与 GPU、RDMA 技术的深度结合，是高性能计算的关键内容。

## 知识点

### GPU 支持 (CUDA-aware MPI)
- **CUDA-aware MPI**：GPU 内存直接通信，无需 CPU 中转
- **GPU Direct**：NVIDIA 的 GPU 间直接通信技术
- **数据传输优化**：如何最大化 GPU 通信效率

### RDMA 基础
- **RDMA 是什么**：Remote Direct Memory Access，远程直接内存访问
- **核心优势**：
  - 零拷贝：数据直接从发送方内存到接收方内存
  - 内核旁路：绕过操作系统内核，减少延迟
  - CPU 卸载：CPU 不参与数据传输
- **协议**：InfiniBand、RoCE (RDMA over Converged Ethernet)
- **传输类型**：RC (Reliable Connection), UC (Unreliable Connection), UD (Unreliable Datagram)

### GPU + RDMA 协同
- **CUDA-aware + RDMA**：GPU 数据直接通过 RDMA 网络传输
- **性能收益**：延迟降低、带宽提升、CPU 卸载

## 核心概念

### RDMA 操作
- **QP (Queue Pair)**：队列对，通信的基本单元
- **MR (Memory Region)**：注册的内存区域
- **CQ (Completion Queue)**：完成队列，通知操作完成
- **PD (Protection Domain)**：保护域，隔离通信

### RDMA  verbs
- `IBV_POST_SEND` - 发送
- `IBV_POST_RECV` - 接收
- `IBV_POST_RDMA_READ` - 远程读取
- `IBV_POST_RDMA_WRITE` - 远程写入

## 学习目标

1. 理解 CUDA-aware MPI 的工作原理
2. 掌握 RDMA 核心概念和优势
3. 理解 GPU + RDMA 协同的优势
4. 能够在环境中配置和验证 GPU/RDMA

## 示例代码

本章配套示例在 `04-hardware/` 目录：
- `cuda_aware_mpi.c` - CUDA-aware MPI 示例
- `rdma_write/` - RDMA Write 操作示例

## 下一步

学完本章后，进入 [第五章：上层栈结合](./ch05-stack/README.md) 学习 NCCL 与 PyTorch 的结合。
