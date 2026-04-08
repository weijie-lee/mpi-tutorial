# MPI 完整教程（含 GPU/RDMA/NCCL/PyTorch）

本教程是一份**完整的 MPI 入门到进阶学习资料**，从零开始讲解 MPI 并行编程，覆盖基础概念、核心编程、硬件加速（GPU/RDMA）、以及在深度学习中与 NCCL/PyTorch 的结合。

适合：
- 刚接触并行计算/MPI 的初学者
- 需要做多节点多GPU训练的深度学习研究者
- 需要复习MPI进阶概念的开发者

## 特点

- ✅ **由浅入深**：从第一个 Hello World 到多节点多GPU实战，循序渐进
- ✅ **可运行代码**：每个章节都配有对应可编译运行的示例代码
- ✅ **覆盖现代特性**：专门讲解 CUDA-aware MPI、RDMA、NCCL 协同、PyTorch DDP 这些现在常用的内容
- ✅ **全链路可观测**：提供从 PyTorch API 到 RDMA 网卡的完整调用链路观测工具
- ✅ **实用优化建议**：包含调试、性能优化要点，帮你避开常见坑

## 目录结构（按章节划分）

```
mpi-tutorial/
├── ch01-intro/                    # 基础概念入门
│   ├── README.md                  # 章节介绍与知识点
│   └── 01-basics/                 # 示例代码
├── ch02-core/                     # 核心编程模型
│   ├── README.md
│   └── 02-core/
├── ch03-advanced/                 # 进阶核心主题
│   ├── README.md
│   └── 03-advanced/
├── ch04-hardware/                 # GPU 与 RDMA 支持
│   ├── README.md
│   └── 04-hardware/
├── ch05-stack/                    # NCCL 与 PyTorch 结合
│   ├── README.md
│   └── 05-pytorch/
├── ch06-applications/             # 完整应用实例
│   ├── README.md
│   └── 06-applications/
├── ch07-optimize/                 # 环境与调试优化
│   └── README.md
├── ch08-rdma-verbs/               # RDMA Verbs 编程
│   ├── README.md
│   └── 08-rdma-verbs/
├── ch09-kubernetes-pytorchjob/    # Kubernetes 部署
│   └── README.md
└── ch10-fullstack-observe/        # 全链路观测实战
    ├── README.md
    └── 10-fullstack-observe/
```

## 学习路径

按顺序阅读跟着跑代码最快：

1. **[ch01-intro](./ch01-intro/)** - 基础概念入门：了解 MPI 是什么，为什么需要它
2. **[ch02-core](./ch02-core/)** - 核心编程模型：点对点+集合通信编程
3. **[ch03-advanced](./ch03-advanced/)** - 进阶主题：派生类型、拓扑、RMA
4. **[ch04-hardware](./ch04-hardware/)** - GPU 与 RDMA：CUDA-aware MPI、RDMA 基础
5. **[ch05-stack](./ch05-stack/)** - NCCL 与 PyTorch：DDP 分布式训练
6. **[ch06-applications](./ch06-applications/)** - 完整应用：二维 Jacobi 迭代
7. **[ch07-optimize](./ch07-optimize/)** - 调试优化：编译运行、调试、性能分析
8. **[ch08-rdma-verbs](./ch08-rdma-verbs/)** - RDMA Verbs 编程
9. **[ch09-kubernetes-pytorchjob](./ch09-kubernetes-pytorchjob/)** - Kubernetes 部署
10. **[ch10-fullstack-observe](./ch10-fullstack-observe/)** - 全链路观测实战 ⭐ NEW

## 快速开始

### 编译示例
```bash
cd ch02-core/02-core
make
```

### 运行示例
```bash
mpirun -np 4 ./hello_mpi
```

## 环境要求

- Linux 系统
- MPI 实现（OpenMPI 4.x 推荐）
- 可选：CUDA（GPU 示例）
- 可选：RDMA 网卡（RDMA 示例）
