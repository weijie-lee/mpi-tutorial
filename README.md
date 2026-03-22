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
- ✅ **实用优化建议**：包含调试、性能优化要点，帮你避开常见坑

## 目录结构

```
mpi-tutorial/
├── README.md           # 本文件
├── docs/               # 文档（按章节）
│   ├── 01-intro.md     # 基础概念入门
│   ├── 02-core.md      # 核心编程模型（点对点+集合通信）
│   ├── 03-advanced.md  # 进阶核心主题（派生类型、拓扑、RMA等）
│   ├── 04-hardware.md  # GPU 与 RDMA 支持（CUDA-aware MPI）
│   ├── 05-stack.md     # NCCL 与 PyTorch 结合（DDP分布式训练）
│   ├── 06-applications.md # 完整应用实例：二维Jacobi迭代
│   ├── 07-optimize.md  # 实现环境与调试优化（调试+性能调优）
│   └── 08-rdma-verbs.md # RDMA Verbs 原生编程入门
└── examples/           # 可运行示例代码
    ├── 01-basics/      # 基础示例（hello、计时、错误处理）
    ├── 02-core/        # 核心通信示例（sendrecv、死锁、非阻塞、集合通信、pi计算）
    ├── 03-advanced/    # 进阶示例（派生类型、笛卡尔拓扑、通信域分裂、RMA）
    ├── 04-hardware/    # GPU/RDMA 示例（CUDA-aware MPI + RDMA Write）
    ├── 05-pytorch/     # PyTorch 示例（MPI初始化、DDP分布式训练）
    ├── 06-applications/# 完整应用（Jacobi二维迭代并行求解泊松方程）
    ├── 08-rdma-verbs/  # RDMA Verbs 原生编程示例（server/client）
    └── build_examples.sh # 一键编译所有C/C++/CUDA示例
```

## 学习路径

按顺序阅读跟着跑代码最快：

1. **入门**：先阅读 [01-intro.md](docs/01-intro.md) 了解基础概念，在你的环境上跑通第一个示例，确认MPI环境正常
2. **核心**：阅读 [02-core.md](docs/02-core.md)，理解点对点通信和集合通信，把里面的示例都跑一遍
3. **进阶**：根据需要选读 [03-advanced.md](docs/03-advanced.md) 里的进阶主题
4. **硬件加速**：如果你有GPU/RDMA，一定要看 [04-hardware.md](docs/04-hardware.md)，理解GPU直接通信怎么工作
5. **深度学习**：看多节点多GPU训练，看 [05-stack.md](docs/05-stack.md)，理解MPI+NCCL+PyTorch分工
6. **完整应用**：看 [06-applications.md](docs/06-applications.md)，学习如何把一个完整问题并行化
7. **调优**：遇到性能问题，看 [07-optimize.md](docs/07-optimize.md) 调试优化
8. **原生RDMA编程**：想直接写RDMA程序，看 [08-rdma-verbs.md](docs/08-rdma-verbs.md)

## 第一步：检查你的环境

先确认你的MPI是否支持需要的特性：

```bash
# 检查 MPI 版本
mpicc --version

# 检查 OpenMPI 是否支持 CUDA
ompi_info --parsable | grep cuda
# 应该看到 "cuda:support = 1"

# 检查是否支持 RDMA/Verbs
ompi_info --parsable | grep verbs
# 有输出说明支持RDMA
```

## 环境要求

| 组件 | 要求 | 说明 |
|------|------|------|
| MPI 实现 | OpenMPI 4.0+ | 推荐，对GPU/RDMA支持好，社区活跃 |
| C编译器 | gcc 7+ / clang 6+ | 编译C示例 |
| CUDA | CUDA 11+（可选）| 编译GPU示例需要 |
| PyTorch | PyTorch 1.10+（可选）| 运行Python示例需要 |
| NCCL | 最新版（可选）| PyTorch多GPU通信需要 |
| libibverbs-dev | libibverbs-dev（可选）| RDMA Verbs编程示例需要 |

本教程在 **RoCE v2 RDMA 网络 + NVIDIA A100 GPU + OpenMPI 4.1.5 + CUDA 11.8** 环境测试过。

## 快速开始编译所有示例

```bash
git clone https://github.com/weijie-lee/mpi-tutorial.git
cd mpi-tutorial/examples
chmod +x build_examples.sh
./build_examples.sh
```

然后就可以跑各个示例了，比如：

```bash
# 跑 Monte Carlo 计算 π 示例
cd 02-core
mpirun -np 4 ./pi_monte_carlo
```

## 章节要点速览

| 章节 | 内容 |
|------|------|
| [01-intro.md](docs/01-intro.md) | 什么是MPI，MPI发展史，MPI vs OpenMP，环境检查，第一个程序 |
| [02-core.md](docs/02-core.md) | 点对点通信（阻塞/非阻塞），死锁，集合通信（Bcast/Scatter/Gather/Reduce/Allreduce），完整计算 π 示例 |
| [03-advanced.md](docs/03-advanced.md) | 派生数据类型，虚拟拓扑，通信域分裂，单侧通信（RMA），MPI-IO |
| [04-hardware.md](docs/04-hardware.md) | CUDA-aware MPI 原理和示例，RDMA 原理和优势，如何检查支持 |
| [05-stack.md](docs/05-stack.md) | MPI vs NCCL 分工，PyTorch DDP 如何配合 MPI，完整多节点多GPU训练示例 |
| [06-applications.md](docs/06-applications.md) | 完整应用：二维Jacobi迭代并行求解泊松方程，讲解域分解和边界交换 |
| [07-optimize.md](docs/07-optimize.md) | 各种MPI实现对比，编译运行方法，调试工具，性能优化要点 |
| [08-rdma-verbs.md](docs/08-rdma-verbs.md) | RDMA Verbs 原生编程入门，完整 client/server 可运行示例 |

## 贡献

欢迎提 Issue 和 Pull Request！

## 许可证

MIT
