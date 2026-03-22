# 三、进阶核心主题

## 1. 派生数据类型

### 为什么需要派生数据类型？
场景：你要发送一个结构体，或者数组中非连续的一块数据，如果没有派生数据类型，你需要：
1. 手动把数据拷贝到一个连续缓冲区
2. 发送这个缓冲区
3. 接收端再手动拆包分开

这很麻烦，而且浪费内存拷贝。派生数据类型就是让 MPI 知道你的数据布局，自动处理非连续数据发送。

### 基本用法

以 C 结构体为例：
```c
typedef struct {
    int id;
    double x;
    double y;
} Particle;
```

定义对应 MPI 派生类型：
```c
// examples/03-advanced/derived_type.c
int count = 3;                  // 三个字段
int blocklengths[3] = {1, 1, 1}; // 每个字段一个元素
MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
MPI_Aint offsets[3];            // 每个字段相对于结构体起始的偏移

offsets[0] = offsetof(Particle, id);
offsets[1] = offsetof(Particle, x);
offsets[2] = offsetof(Particle, y);

MPI_Datatype MPI_Particle;      // 新类型句柄
MPI_Type_create_struct(count, blocklengths, offsets, types, &MPI_Particle);
MPI_Type_commit(&MPI_Particle); // 提交类型才能使用

// 之后就可以直接发送 Particle 了：
Particle p;
MPI_Send(&p, 1, MPI_Particle, dest, tag, comm);

// 使用完释放
MPI_Type_free(&MPI_Particle);
```

### 其他场景
- `MPI_Type_contiguous`: 创建连续多个相同类型
- `MPI_Type_vector`: 创建块状非连续数组（比如矩阵中的几行）
- `MPI_Type_indexed`: 自定义每个块的偏移和长度

## 2. 进程虚拟拓扑

MPI 允许你给进程定义一个虚拟拓扑，帮助 MPI 做路由优化，也方便你根据拓扑位置找邻居进程。

### 笛卡尔拓扑（最常用）
适用于网格/三维空间分块的计算（比如 CFD、物理模拟）。

示例：创建一个 2x2 二维网格拓扑：
```c
int dims[2] = {2, 2};        // 每个维度的进程数
int periods[2] = {0, 0};     // 是否周期边界（0=不周期，1=周期）
int reorder = 1;             // 允许 MPI 重新排序 rank 优化通信
MPI_Comm cart_comm;
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
```

之后你可以根据坐标得到 rank，或者根据 rank 得到坐标：
```c
int coords[2];
MPI_Cart_coords(cart_comm, rank, 2, coords); // 获取当前进程坐标
MPI_Cart_rank(cart_comm, coords, &rank);     // 根据坐标获取 rank
```

还可以很方便地找上下左右邻居：
```c
int left_rank, right_rank;
MPI_Cart_shift(cart_comm, 0, 1, &rank, &left_rank);  // 0 维度移 1 步
```

## 3. 动态进程管理

静态 MPI 程序是 `mpirun` 一次性启动所有进程，动态进程管理允许你在运行时再启动新进程。

### `MPI_Comm_spawn` 启动新进程
```c
// 在当前通信域之外，再启动 n 个相同程序的进程
MPI_Comm intercomm;
MPI_Comm_spawn("./myprogram", MPI_ARGV_NULL, n,
               MPI_INFO_NULL, 0, MPI_COMM_WORLD,
               &intercomm, MPI_ERRCODES_IGNORE);
```
父进程和子进程通过 `intercomm` 互相通信。

### `MPI_Comm_split` 分裂通信域
把原来的一个通信域按颜色分裂成多个新通信域，同一个颜色的进程在一个新通信域：
```c
// 比如按奇偶分裂：
int color = rank % 2;
MPI_Comm newcomm;
MPI_Comm_split(MPI_COMM_WORLD, color, rank, &newcomm);
```
常用于把进程分组做不同任务。

## 4. 单侧通信（RMA / One-Sided Communication）

传统消息传递是双边的，发收都要双方参与。**单侧通信（Remote Memory Access，RMA）**允许一个进程直接读写另一个进程的内存，不需要对方进程主动参与。

### 基本概念
- **窗口（Window）**：每个进程把自己一块内存注册成一个可远程访问的窗口
- 三个基本操作：
  1. `MPI_Put`: 把本地数据写到对端窗口
  2. `MPI_Get`: 把对端窗口数据读到本地
  3. `MPI_Accumulate`: 在对端窗口做原子归约操作（比如加法）

### 示例代码框架
```c
MPI_Win win;
double *remote_buf;
// 暴露本地 buffer 给远程访问
MPI_Win_create(remote_buf, size, sizeof(double), MPI_INFO_NULL, comm, &win);

// 同步后开始操作
MPI_Win_fence(0, win);

// 把本地 data 写到对端 rank 的偏移位置
MPI_Put(data, count, MPI_DOUBLE, target_rank, offset, count, MPI_DOUBLE, win);

// 同步保证完成
MPI_Win_fence(0, win);
```

### 适用场景
- 动态负载均衡，进程需要随时访问任意其他进程的数据
- 检查点/重启
- 稀疏计算，不规则访问

## 5. 并行 IO：MPI-IO

当多个进程需要同时读写同一个大文件时，直接用 C 标准库 `fopen` 容易出问题，而且性能差。MPI-IO 提供了并行文件访问接口：

- 支持多个进程并发读写同一个文件的不同区域
- 支持文件视图（把一个连续文件逻辑划分给不同进程）
- 针对并行文件系统做优化，比单独读写快很多

基本用法示例：
```c
MPI_File fh;
MPI_File_open(comm, "data.bin", MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);

// 设置本进程要访问的文件区域
MPI_Offset offset = rank * BLOCK_SIZE;
MPI_File_seek(fh, offset, MPI_SEEK_SET);

// 读写
MPI_File_write(fh, buffer, count, MPI_DOUBLE, MPI_STATUS_IGNORE);

MPI_File_close(&fh);
```

## 示例代码

- [derived_type.c](../examples/03-advanced/derived_type.c) - 派生数据类型示例（结构体发送）
- [cartesian.c](../examples/03-advanced/cartesian.c) - 笛卡尔拓扑示例
- [comm_split.c](../examples/03-advanced/comm_split.c) - 通信域分裂示例
- [rma_putget.c](../examples/03-advanced/rma_putget.c) - 单侧通信Put/Get示例

## 下一步

→ 下一章：[硬件结合：GPU 与 RDMA 支持](04-hardware.md)
