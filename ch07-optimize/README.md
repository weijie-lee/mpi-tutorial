# 第七章：环境与调试优化

## 本章简介

本章讲解 MPI 开发环境的搭建、程序编译运行方法，以及调试和性能优化技巧。

## 知识点

### 编译与运行
- **编译器包装器**：`mpicc`, `mpicxx`, `mpifort`
- **常用选项**：`-o` 输出文件, `-lm` 链接库, `-L` 库路径
- **运行命令**：`mpirun` / `mpiexec`
- **常用选项**：
  - `-np N` 指定进程数
  - `-hostfile` 指定运行主机
  - `-bind-to` 绑定 CPU 核心

### 主流 MPI 实现
- **OpenMPI**：开源、社区活跃，默认安装
- **MPICH**：轻量、高性能
- **Intel MPI**：针对 Intel 硬件优化

### 调试方法
- **打印调试**：简单直接
- **MPI 调试器**：`TotalView`, `Arm DDT`, `gdb` 配合
- **通信验证**：`MPI_Init`, `MPI_Comm_rank`, `MPI_Finalize` 正确性检查

### 性能分析与优化
- **通信开销**：减少消息数量、合并小消息
- **负载均衡**：均匀分配计算任务
- **拓扑感知**：利用网络拓扑优化通信
- **工具**：VTune, Score-P, TAU, IPM

### 常见问题
- **死锁**：循环等待，常见于发送接收顺序不匹配
- **内存泄漏**：未释放 MPI 对象
- **数据类型不匹配**：发送接收类型不一致

## 核心概念

### 编译示例
```bash
# 编译 C 程序
mpicc -o program program.c -lm

# 编译 C++ 程序
mpicxx -o program program.cpp

# 编译 CUDA 程序
mpicxx -o program program.cu -lcudart
```

### 运行示例
```bash
# 本地 4 进程
mpirun -np 4 ./program

# 指定主机文件
mpirun -np 8 -hostfile hosts.txt ./program

# 指定 GPU
mpirun -np 2 -x CUDA_VISIBLE_DEVICES=0,1 ./program
```

## 学习目标

1. 掌握 MPI 程序的编译和运行方法
2. 学会使用基本调试技巧
3. 理解性能优化的关键点
4. 能够排查常见 MPI 问题

## 下一步

学完本章后，进入 [第八章：RDMA Verbs 编程](./ch08-rdma-verbs/README.md) 学习 RDMA 原生编程。
