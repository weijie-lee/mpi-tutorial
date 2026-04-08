# 第三章：进阶核心主题

## 本章简介

本章深入 MPI 高级特性，包括派生数据类型、进程拓扑和单侧通信（RMA）。

## 派生数据类型

### 为什么需要派生数据类型？

假设你要发送一个结构体：

```c
struct Person {
    char name[20];
    int age;
    float salary;
};
```

普通方式发送很麻烦，需要逐个字段发送。**派生数据类型**让你一次性定义复杂数据的"形状"。

### 主要类型

| 类型 | 用途 | 创建函数 |
|------|------|----------|
| 连续型 | 连续内存块 | MPI_Type_contiguous |
| 向量型 | 间隔复制 | MPI_Type_vector |
| 索引型 | 任意位置 | MPI_Type_indexed |
| 结构型 | 不同类型组合 | MPI_Type_struct |

### 连续类型

```c
// 连续 10 个整数
MPI_Datatype int_array;
MPI_Type_contiguous(10, MPI_INT, &int_array);
MPI_Type_commit(&int_array);

// 发送
MPI_Send(buf, 1, int_array, dest, tag, comm);
```

### 向量类型

适合**列优先**存储的矩阵：

```c
// 每行 10 个元素，共 5 行，跳过 10 个元素取下一行
MPI_Datatype row_type;
MPI_Type_vector(5, 10, 10, MPI_DOUBLE, &row_type);
MPI_Type_commit(&row_type);

// 发送一整列（逻辑上连续）
MPI_Send(matrix, 1, row_type, dest, tag, comm);
```

### 索引类型

发送数组中的**不连续**部分：

```c
int blocklens[] = {3, 2, 4};
int displs[] = {0, 7, 15};  // 起始位置

MPI_Datatype indexed_type;
MPI_Type_indexed(3, blocklens, displs, MPI_INT, &indexed_type);
MPI_Type_commit(&indexed_type);
```

## 进程拓扑

### 什么是虚拟拓扑？

为进程建立**逻辑拓扑结构**，让通信更高效。

常见拓扑：
- **笛卡尔拓扑**：网格、环形、立方体
- **图拓扑**：任意连接关系

### 笛卡尔拓扑

想象一个 2D 网格：

```
P0  P1  P2
P3  P4  P5
P6  P7  P8
```

进程可以向**邻居**发送消息，而不是用 rank 号：

```c
int dims[2] = {3, 3};  // 3x3 网格
int periods[2] = {1, 1};  // 周期边界（首尾相连）
MPI_Comm cart_comm;

MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

// 获取当前进程的邻居
int rank, coords[2];
MPI_Comm_rank(cart_comm, &rank);
MPI_Cart_coords(cart_comm, rank, 2, coords);

// 获取上下左右邻居的 rank
int left, right, up, down;
MPI_Cart_shift(cart_comm, 0, 1, &up, &down);   // 垂直方向
MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // 水平方向
```

### 拓扑的好处

1. **直观**：用坐标而非 rank 通信
2. **高效**：适合物理上相邻的通信模式
3. **可移植**：拓扑映射到实际硬件

## 单侧通信 (RMA)

### 什么是单侧通信？

**双侧通信**：发送方和接收方**同时**参与
```
A: "我要发了" → B: "好的准备好了" → A: "数据" → B: "收到"
```

**单侧通信**（RMA, Remote Memory Access）：只需**一方**发起操作
```
A: 直接写入 B 的内存
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Window** | 可以被远程访问的内存区域 |
| **Put** | 写入远程内存 |
| **Get** | 读取远程内存 |
| **Fence** | 同步机制 |

### 使用流程

```c
// 1. 创建窗口（每个进程提供一块共享内存）
MPI_Win win;
MPI_Alloc_mem(1024, MPI_INFO_NULL, &local_buffer);
MPI_Win_create(local_buffer, 1024, 1, MPI_INFO_NULL, 
               MPI_COMM_WORLD, &win);

// 2. 同步（确保窗口就绪）
MPI_Win_fence(0, win);

// 3. 写入远程内存（从 rank 0 写入 rank 1）
if (rank == 0) {
    MPI_Put(local_data, count, MPI_INT, 1, 0, count, MPI_INT, win);
}

// 4. 再次同步
MPI_Win_fence(0, win);

// 5. 结束
MPI_Win_free(&win);
MPI_Free_mem(local_buffer);
```

### RMA vs 双侧通信

| 特性 | 双侧 (Send/Recv) | 单侧 (RMA) |
|------|------------------|------------|
| 参与方 | 发送方 + 接收方 | 发起方 |
| 同步 | 隐式 | 需显式 fence |
| 适用场景 | 动态数据流 | 固定模式访问 |

## 示例代码

本章配套示例在 `03-advanced/` 目录：

- `derived_type.c` - 派生数据类型
- `cartesian.c` - 笛卡尔拓扑
- `comm_split.c` - 通信域分裂
- `rma_putget.c` - RMA Put/Get 操作

```bash
cd ch03-advanced/03-advanced
mpicc -o derived_type derived_type.c
mpirun -np 4 ./derived_type
```

## 本章测验

- [ ] 掌握派生数据类型的创建和使用
- [ ] 理解笛卡尔拓扑的应用场景
- [ ] 掌握 RMA 编程模式
- [ ] 理解单侧 vs 双侧通信的区别

## 下一步

学完本章后，进入 [第四章：硬件结合](./ch04-hardware/README.md) 学习 GPU 与 RDMA 支持。
