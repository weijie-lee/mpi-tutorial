# 第三章：进阶核心主题

## 本章简介

本章深入 MPI 高级特性，包括派生数据类型、进程拓扑和单侧通信（RMA）。

## 知识点

### 派生数据类型
- **连续派生类型**：连续内存块的数据打包
- **向量派生类型**：间隔复制数据
- **索引派生类型**：任意位置数据的组合
- **结构派生类型**：不同类型数据的组合

### 进程拓扑
- **虚拟拓扑**：为进程建立逻辑拓扑结构
- **笛卡尔拓扑**：网格/环形等规则拓扑
- **图拓扑**：任意图的拓扑结构

### 单侧通信 (RMA - Remote Memory Access)
- **窗口 (Window)**：定义可以远程访问的内存区域
- **Put 操作**：写入远程进程内存
- **Get 操作**：读取远程进程内存
- **同步机制**：Fence, Lock/Unlock, Post/Start/Complete/Wait

## 核心概念

### 派生数据类型
```c
// 创建连续类型
MPI_Type_contiguous(count, oldtype, &newtype);

// 创建向量类型（间隔复制）
MPI_Type_vector(count, blocklength, stride, oldtype, &newtype);

// 提交类型
MPI_Type_commit(&newtype);
```

### RMA 操作
```c
// 创建窗口
MPI_Win_create(buf, size, disp_unit, info, comm, &win);

// 写入远程内存
MPI_Put(origin_addr, origin_count, origin_dtype, 
         target_rank, target_disp, target_count, target_dtype, win);

// 读取远程内存
MPI_Get(origin_addr, origin_count, origin_dtype,
        target_rank, target_disp, target_count, target_dtype, win);

// 同步
MPI_Win_fence(win);
```

## 学习目标

1. 掌握派生数据类型的创建和使用
2. 理解进程拓扑的概念和应用
3. 掌握单侧通信的编程模式
4. 理解 RMA 同步机制

## 示例代码

本章配套示例在 `03-advanced/` 目录：
- `derived_type.c` - 派生数据类型
- `cartesian.c` - 笛卡尔拓扑
- `graph_topology.c` - 图拓扑
- `rma_putget.c` - RMA Put/Get 操作

## 下一步

学完本章后，进入 [第四章：硬件结合](./ch04-hardware/README.md) 学习 GPU 与 RDMA 支持。
