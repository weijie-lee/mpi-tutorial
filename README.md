# MPI 完整教程（含 GPU/RDMA/NCCL/PyTorch）

本教程是一份完整的 MPI 入门到进阶学习资料，覆盖了基础概念、核心编程、硬件加速（GPU/RDMA）、以及与 NCCL/PyTorch 的结合。

## 目录结构

```
mpi-tutorial/
├── README.md           # 本文件
├── docs/               # 文档（按章节）
│   ├── 01-intro.md     # 基础概念入门
│   ├── 02-core.md      # 核心编程模型
│   ├── 03-advanced.md  # 进阶核心主题
│   ├── 04-hardware.md  # GPU 与 RDMA 支持
│   ├── 05-stack.md     # NCCL 与 PyTorch 结合
│   ├── 06-optimize.md  # 实现环境与调试优化
│   └── 07-rdma-verbs.md # RDMA Verbs 原生编程入门
└── examples/           # 可运行示例代码
    ├── 01-basics/      # 基础示例
    ├── 02-core/        # 核心通信示例
    ├── 03-advanced/    # 进阶示例
    ├── 04-hardware/    # GPU/RDMA 示例
    ├── 05-pytorch/     # PyTorch 示例
    ├── 06-applications/# 完整应用
    └── 07-rdma-verbs/  # RDMA Verbs 编程示例
```

## 学习路径

1. 先阅读 [01-intro.md](docs/01-intro.md) 了解基础概念
2. 跑通基础示例，理解点对点和集合通信
3. 进阶主题根据兴趣选读
4. 如果关注深度学习，重点看 GPU/RDMA/NCCL/PyTorch 部分

## 环境要求

- MPI 实现：OpenMPI 4.0+ 推荐，支持 CUDA/RDMA
- C 编译器：gcc 7+ 或 clang 6+
- CUDA（可选，GPU 示例需要）：CUDA 11+
- PyTorch（可选，深度学习示例需要）：PyTorch 1.10+
- NCCL（可选，PyTorch 多 GPU 示例需要）
- libibverbs-dev（可选，RDMA Verbs 编程示例需要）

## 快速开始

```bash
# 编译基础示例
cd examples/01-basics
mpicc -o hello hello.c
mpirun -np 4 ./hello
```

## 许可证

MIT

