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

`stride` 的单位是元素个数。`MPI_Type_vector(5, 10, 10, ...)` 是 5 段各 10 个元素、段间距也是 10，也就是一块连续内存，不是一列。列向量是每段 1 个元素、间距为行宽：

```c
MPI_Datatype column;
MPI_Type_vector(nrows, 1, ncols, MPI_DOUBLE, &column);
MPI_Type_commit(&column);
MPI_Send(&matrix[0][column_index], 1, column, dest, tag, comm);
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

`MPI_Get` 写入的本地缓冲，要等这个访问纪元结束之后才能读。在下一次 `MPI_Win_fence` 之前打印它，内容没有定义。窗口内存用 `MPI_Win_allocate` 分配；把栈上的变量交给 `MPI_Win_create`，有的 OpenMPI 组件会返回 `MPI_ERR_WIN`。

### 使用流程

```c
MPI_Win win;
int *data = NULL;
MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL,
                 MPI_COMM_WORLD, &data, &win);

MPI_Win_fence(0, win);
if (rank == 1) {
    MPI_Get(&recv_data, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
}
MPI_Win_fence(0, win);          // 到这里 recv_data 才有定义
// 下一个纪元里再 Put，并再 fence 一次之后，目标进程才能读窗口

MPI_Win_free(&win);             // 同时释放 Win_allocate 的内存
```

### RMA vs 双侧通信

| 特性 | 双侧 (Send/Recv) | 单侧 (RMA) |
|------|------------------|------------|
| 参与方 | 发送方 + 接收方 | 发起方 |
| 同步 | 隐式 | 需显式 fence |
| 适用场景 | 动态数据流 | 固定模式访问 |

## 示例代码

本章配套示例在 `03-advanced/` 目录：

- `derived_type.c` - 结构体派生类型，并核对收到的字段
- `vector_indexed.c` - 用向量类型取一列，用索引类型取不连续元素
- `cartesian.c` - 笛卡尔拓扑。`reorder=1` 时使用新通信域里的 rank
- `comm_split.c` - 通信域分裂，以及 `MPI_Group_incl`
- `rma_putget.c` - RMA Put/Get。Get 的结果在 fence 之后检查

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

学完本章后，进入 [第四章：硬件结合](../ch04-hardware/README.md)。向量类型参数和 RMA 纪元见 [基础知识补充](../docs/基础知识补充.md)。
