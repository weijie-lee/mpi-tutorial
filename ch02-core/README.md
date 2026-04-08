# 第二章：核心编程模型

## 本章简介

本章是 MPI 编程的核心基础，带你掌握点对点通信和集合通信的编程模型。

## 知识点

- **通信域 (Communicator)**：管理进程组和通信上下文
- **进程组 (Group)**：进程的有序集合
- **进程秩 (Rank)**：进程在通信域中的唯一 ID
- **点对点通信**：`MPI_Send` / `MPI_Recv` 及各种变体
- **阻塞 vs 非阻塞通信**：理解通信语义
- **集合通信**：`MPI_Bcast`, `MPI_Reduce`, `MPI_Scatter`, `MPI_Gather`, `MPI_Allreduce` 等
- **死锁避免**：常见的死锁模式和解决方案
- **数据类型**：基本数据类型和自定义数据类型

## 核心概念

### 点对点通信
```c
// 发送
MPI_Send(buf, count, datatype, dest, tag, comm);
// 接收
MPI_Recv(buf, count, datatype, source, tag, comm, status);
```

### 集合通信
- **广播** `MPI_Bcast`：一个进程发给所有进程
- **归约** `MPI_Reduce`：所有进程的数据合并到一个进程
- **全归约** `MPI_Allreduce`：所有进程都得到归约结果
- **散射** `MPI_Scatter`：一个进程的数据分发给所有进程
- **收集** `MPI_Gather`：所有进程的数据收集到一个进程

## 学习目标

1. 掌握 MPI 基本编程流程（初始化、获取信息、通信、结束）
2. 熟练使用点对点通信 API
3. 理解阻塞/非阻塞通信的区别
4. 掌握常用集合通信的使用场景
5. 学会避免通信死锁

## 示例代码

本章配套示例在 `02-core/` 目录：
- `hello_mpi.c` - 基本初始化和进程信息
- `sendrecv.c` - 点对点通信
- `blocking.c` - 阻塞通信与死锁
- `nonblocking.c` - 非阻塞通信
- `collective_broadcast.c` - 广播
- `collective_reduce.c` - 归约
- `pi计算/` - Monte Carlo 方法计算 π

## 下一步

学完本章后，进入 [第三章：进阶核心主题](./ch03-advanced/README.md) 学习派生类型、拓扑和 RMA。
