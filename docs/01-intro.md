# 一、基础概念入门

## 1. 什么是 MPI

**MPI** 全称 **Message Passing Interface**，中文叫**消息传递接口**，是一个**并行计算的通信标准**，不是一门独立的编程语言。

### 核心定位
- 它只是定义了一套编程接口（函数调用规范）
- 具体实现由不同的库提供（OpenMPI、MPICH 等）
- 用户可以用 C/C++/Fortran/Python 等语言调用 MPI 接口做并行编程

### 核心价值
在**分布式内存架构**中，每个进程有独立的地址空间，一个进程不能直接读写另一个进程的内存，必须通过**消息传递**来交换数据。MPI 就是为这种场景标准化了通信接口，让并行程序更容易移植。

简单来说：
- **共享内存**（比如多线程）：多个线程读同一块内存 → 不需要显式消息传递
- **分布式内存**（比如多节点、多网卡机器）：每个进程只能访问自己内存 → 需要显式消息传递，这就是 MPI 解决的问题

### 常见 MPI 实现
目前生产环境常用的两个实现：
- **OpenMPI**：开源、社区活跃，对GPU/RDMA支持好，很多超算和云环境用这个
- **MPICH**：更轻量，性能也不错，很多发行版默认带的是这个
- **Intel MPI**：英特尔针对自家硬件优化的闭源实现，在Xeon集群上性能很好

在我们的环境中已经安装了 **OpenMPI 4.x**，并且支持 CUDA 和 RDMA。

## 2. MPI 发展历史

### 诞生背景
90年代初，并行计算开始兴起，但不同厂商的并行计算机使用各自不同的通信接口，程序移植非常困难。学术界和工业界一起推动了标准化工作，诞生了 MPI。

### 版本演进
- **MPI-1** (1994): 基础点对点通信、集合通信，定义了最核心的接口
- **MPI-2** (1996): 加入了动态进程管理、单侧通信（RMA）、并行IO（MPI-IO）
- **MPI-3** (2012): 增强了单侧通信、支持非阻塞集合通信、改进了错误处理
- **MPI-4** (2021): 最新版本，对GPU支持、灵活进程拓扑、可扩展IO做了进一步增强

### 最新进展
现代 MPI 实现已经原生支持：
- GPU 直接通信（CUDA-aware/ROCM-aware MPI）
- RDMA 高速网络协议（InfiniBand/RoCE）
- 数千到数十万节点的可扩展

## 3. MPI vs 其他并行模型

### MPI vs OpenMP
| 特性 | MPI | OpenMP |
|------|-----|--------|
| 内存模型 | 分布式内存 | 共享内存 |
| 可扩展性 | 支持多节点，可到数十万核 | 一般单节点，最多几十核 |
| 编程复杂度 | 需要显式划分数据、消息传递 | 基于编译制导，增量并行 |
| 应用场景 | 大规模集群计算 | 单节点多核并行 |

实际应用中经常**混合使用**: MPI 做节点间通信，OpenMP 做节点内多核并行。

### MPI vs 原生进程间通信（socket/pipes）
| 特性 | MPI | socket/pipes |
|------|-----|--------------|
| 抽象层次 | 高层通信原语（广播、归约等） | 底层字节流 |
| 可移植性 | 标准接口，所有平台支持 | 需要自己处理地址、连接、错误 |
| 性能 | 针对并行计算优化，集合通信有算法优化 | 需要自己实现优化 |
| 使用场景 | 大规模并行计算 | 通用网络服务 |

### MPI + GPU 混合架构
现在主流的超算都是「多节点 - 多GPU」架构：
- 每个节点内有多张GPU卡
- 节点之间通过高速网络（IB/RoCE）连接
- 一般编程模型是：**一个进程对应一张GPU**，进程之间通过 MPI 通信

在我们的环境中，每个节点有 8 张 A100 GPU，并且节点之间通过 RoCE v2 RDMA 网络连接，可以直接运行后面章节的硬件加速示例。

## 4. 核心概念理解

### Rank（进程编号）
每个 MPI 进程会被分配一个唯一的整数编号，从 `0` 开始递增，叫做 rank。这是进程在通信域中的身份标识。

### Communicator（通信域）
进程分组，同一个通信域内的进程可以互相通信。最常用的是 `MPI_COMM_WORLD`，它包含了程序启动时所有的进程。

### 数据类型
MPI 定义了一系列和 C 语言类型对应的数据类型（比如 `MPI_INT`、`MPI_FLOAT`、`MPI_DOUBLE` 等），发送接收时需要指定。

### 通信模式
- **点对点通信**：两个进程之间直接发送/接收
- **集合通信**：一个通信域内所有进程一起参与（广播、归约、散射、聚集等）

## 5. 环境检查与编译运行

### 检查环境
在我们的机器上可以这样检查 MPI 环境：

```bash
# 检查 OpenMPI 版本
ompi_info --version

# 检查是否支持 CUDA
ompi_info | grep -i cuda

# 检查是否支持 RDMA/IB
ompi_info | grep -i "Verbs\|RDMA"
```

### 编译工具链
MPI 提供了封装好的编译器 wrappers，自动链接头文件和库：
- `mpicc`  - 编译 C 代码
- `mpicxx` - 编译 C++ 代码
- `mpif90` - 编译 Fortran 代码

不需要手动链接 `-lmpi`，wrapper 已经处理好了。

### 运行命令：mpirun 完整用法

`mpirun`（也叫 `mpiexec`）是 MPI 程序的启动器，负责在多个节点上启动所有 MPI 进程并建立通信。

#### 基本用法

**单节点运行**（我们默认环境单节点有 8 张 GPU，通常一个 GPU 对应一个进程）：
```bash
# 启动 1 个进程（测试用）
mpirun -np 1 ./hello

# 启动 4 个进程（单节点内）
mpirun -np 4 ./hello

# 启动 8 个进程（占满单节点的所有 GPU，最常用）
mpirun -np 8 ./hello
```

**多节点运行**（多个机器一起跑）：

**方式一：命令行直接指定主机**
```bash
# 单节点 1 号机跑 8 个进程，单节点 2 号机跑 8 个进程，总共 16 个进程
mpirun -np 16 -H node01:8,node02:8 ./hello
```

**方式二：使用 hostfile（推荐，更清晰）**

创建一个 `hostfile` 文本文件，每行写一个节点，后面加 `slots=N` 表示这个节点最多能跑多少个进程（一般等于 GPU 数量）：
```
# 这是我的 hostfile 内容
node01 slots=8
node02 slots=8
node03 slots=8
```

然后运行：
```bash
# 在 3 个节点上总共启动 24 个进程（每个节点 8 个）
mpirun -np 24 --hostfile hostfile ./hello

# 或者只在两个节点上跑 16 个进程
mpirun -np 16 --hostfile hostfile ./hello
```

#### 常用参数详解

| 参数 | 作用 | 示例 |
|------|------|------|
| `-np N` | 启动 **总共 N 个** MPI 进程 | `-np 8` |
| `--hostfile file` | 从文件读取节点列表和 slots | `--hostfile myhosts` |
| `-H node1:8,node2:8` | 命令行直接指定节点和 slots | `-H node1:8,node2:8` |
| `-npernode N` | **每个节点启动 N 个进程**，自动计算总数 | `-npernode 8 -N 2` 启动 16 进程 |
| `-N N` | 总共使用 N 个节点 | 配合 `-npernode` 使用 |
| `--bind-to core` | 把进程绑定到 CPU 核心，提升性能 | 默认一般已经开了 |
| `--map-by socket` | 按 NUMA socket 分配进程，每个 socket 一个进程组 | `--map-by socket:pe=4` |
| `--allow-run-as-root` | 允许 root 用户运行（有时候在容器里需要） | |
| `-x VAR_NAME` | 导出环境变量到所有 MPI 进程 | `-x CUDA_VISIBLE_DEVICES` |

#### 进程绑定最佳实践

在我们的环境中（单节点 8 张 GPU，每个进程一个 GPU），推荐这样绑定保证性能：
```bash
# 每个 socket 对应一个 NUMA 节点，通常每张 GPU 连接到一个 NUMA 节点
# 8 卡机器一般是 2 个 CPU socket，每个 socket 对应 4 张 GPU
mpirun -np 8 --map-by socket:pe=4 ./hello
```

这样可以让进程访问本地 GPU 延迟更低，性能更好。

#### 在 SLURM 集群上运行

如果你的集群用 SLURM 调度，直接用 `srun` 启动，它会自动调用 MPI：
```bash
# 单节点 8GPU
srun -N 1 --ntasks-per-node=8 ./hello

# 2 节点，每节点 8GPU，总共 16 进程
srun -N 2 --ntasks-per-node=8 ./hello
```

一般不需要自己加 `mpirun`，`srun` 会处理好。

#### 环境变量传递

如果你需要给 MPI 程序传递环境变量，比如 NCCL 调试：
```bash
# 方法一：-x 参数传递
mpirun -np 8 -x NCCL_DEBUG=INFO ./myapp

# 方法二：先 export 再 mpirun
export NCCL_DEBUG=INFO
mpirun -np 8 ./myapp
```

## 示例代码：第一个 MPI 程序

```c
// examples/01-basics/hello.c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 初始化 MPI 环境，必须第一个调用
    MPI_Init(&argc, &argv);

    // 获取当前进程的 rank（编号），从 0 开始
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 获取通信域中总进程数
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 获取处理器名称
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(hostname, &len);

    printf("Hello from rank %d/%d on %s\n", rank, size, hostname);

    // 结束 MPI 环境，必须最后调用
    MPI_Finalize();
    return 0;
}
```

### 编译运行

```bash
cd examples/01-basics
mpicc -O2 -o hello hello.c
mpirun -np 4 ./hello
```

可能输出（顺序不一定固定）：
```
Hello from rank 0 of 4 on node01
Hello from rank 1 of 4 on node01
Hello from rank 2 of 4 on node01
Hello from rank 3 of 4 on node01
```

## 更多基础示例

### 示例 2：获取当前时间，计算程序运行时长

```c
// examples/01-basics/timing.c
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI 提供了高精度计时器
    double start = MPI_Wtime();
    
    // 模拟做一些计算工作，这里睡 1 秒
    sleep(1);
    
    double end = MPI_Wtime();
    
    if (rank == 0) {
        printf("Total elapsed time: %.3f seconds\n", end - start);
    }

    MPI_Finalize();
    return 0;
}
```

运行：
```bash
mpicc -o timing timing.c
mpirun -np 4 ./timing
```

> 💡 **小知识**：`MPI_Wtime()` 返回的是从过去某个时间点到现在的秒数，是所有进程都同步的吗？不一定！它只是每个进程本地的时钟。如果你要计算一段跨进程通信的时间，建议让 rank 0 测量起止。

### 示例 3：错误处理

```c
// examples/01-basics/error_handling.c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int err;
    err = MPI_Init(&argc, &argv);
    
    // 默认情况下，任何 MPI 错误都会直接终止整个程序
    // 可以通过错误处理器改变行为
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    int rank;
    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (err != MPI_SUCCESS) {
        char errstr[BUFSIZ];
        int errlen;
        MPI_Error_string(err, errstr, &errlen);
        fprintf(stderr, "MPI Error: %s\n", errstr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("No error, rank = %d\n", rank);
    MPI_Finalize();
    return 0;
}
```

## 练习

1. 修改 `hello.c`，让只有 rank 0 打印 "I am the master!"，其他 rank 打印 "I am a worker"
2. 使用 `MPI_Wtime()` 测量启动 1、2、4、8 个进程的时间，看看随进程数增长变化大吗？
3. 尝试在多个节点上运行（如果有多节点环境），看看 `MPI_Get_processor_name` 的输出

## 下一步

→ 下一章：[核心编程模型](02-core.md)
