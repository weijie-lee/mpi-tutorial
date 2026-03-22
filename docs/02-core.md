# 二、核心编程模型

## 1. 基本概念

### 通信域（Communicator）与进程组（Group）
- **进程组（Group）**：一组参与通信的进程的有序集合
- **通信域（Communicator）**：包装了进程组和通信上下文，给消息提供一个独立的"命名空间"
- 默认通信域：`MPI_COMM_WORLD`，包含了程序启动时所有的进程

**为什么需要通信域？** 把不同模块的通信隔离开，避免不同模块的消息互相匹配错。比如一个库自己内部通信用自己的通信域，不会和应用层的消息冲突。

### 进程秩（Rank）
- 每个进程在通信域中会被分配一个唯一的整数编号，从 `0` 开始计数
- 通信时，源和目的都是用 rank 来标识
- 作用：让每个进程知道自己的"身份"，从而决定做什么计算

### 预定义数据类型
MPI 预定义了和 C 语言基本类型对应的 MPI 数据类型：

| C 类型 | MPI 类型 |
|--------|----------|
| `char` | `MPI_CHAR` |
| `int` | `MPI_INT` |
| `float` | `MPI_FLOAT` |
| `double` | `MPI_DOUBLE` |
| `long` | `MPI_LONG` |
| `long long` | `MPI_LONG_LONG` |
| `long double` | `MPI_LONG_DOUBLE` |

还有一些特殊类型，后面讲自定义数据类型的时候再说。

### 消息结构
一个 MPI 消息包含：
1. **数据缓冲区**：要发送的数据所在内存地址
2. **数据长度**：发送多少个元素
3. **数据类型**：每个元素是什么类型
4. **标签（Tag）**：用户自定义的整数，用来区分不同类型的消息
5. **源/目的秩**：消息从哪来，到哪去

## 2. 程序基本结构

所有 MPI 程序都遵循这个结构：
```c
#include <mpi.h>

int main(int argc, char** argv) {
    // 1. 初始化 MPI 环境，必须第一个调用
    MPI_Init(&argc, &argv);

    // 2. 获取当前进程信息
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 3. 你的计算和通信逻辑...

    // 4. 结束 MPI 环境，必须最后调用
    MPI_Finalize();
    return 0;
}
```

> ⚠️ **重要规则**：`MPI_Init` 之前只能调用很少的 MPI 函数，`MPI_Finalize` 之后不能再调用任何 MPI 函数。

这部分已经在第一个示例中看到了，下面看核心的通信操作。

## 3. 点对点通信（Point-to-Point）

点对点通信就是**一个进程发，一个进程收**，是最基础的通信方式。

### 基础阻塞通信：`MPI_Send` / `MPI_Recv`

```c
// 发送
int MPI_Send(
    const void *buf,      // 发送缓冲区首地址
    int count,            // 发送元素个数（不是字节数！）
    MPI_Datatype datatype,// 每个元素的数据类型
    int dest,             // 目标进程 rank
    int tag,              // 消息标签，用来区分不同消息
    MPI_Comm comm         // 通信域
);

// 接收
int MPI_Recv(
    void *buf,            // 接收缓冲区首地址
    int count,            // 最多接收多少个元素
    MPI_Datatype datatype,// 数据类型
    int source,           // 源进程 rank
    int tag,              // 消息标签（MPI_ANY_TAG 匹配任意标签）
    MPI_Comm comm,        // 通信域
    MPI_Status *status    // 返回状态（包含实际源、标签、错误码）
);
```

### 完整示例：rank 0 发消息给 rank 1

```c
// examples/02-core/sendrecv.c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        int data = 42;
        printf("Rank 0: sending %d to rank 1\n", data);
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int data;
        MPI_Status status;
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: received %d from rank 0\n", data);
    }

    MPI_Finalize();
    return 0;
}
```

编译运行：
```bash
cd examples/02-core
mpicc -o sendrecv sendrecv.c
mpirun -np 2 ./sendrecv
```

输出应该是：
```
Rank 0: sending 42 to rank 1
Rank 1: received 42 from rank 0
```

### 获取接收消息的实际长度
有时候你不知道对方发了多少数据，可以用 `MPI_Get_count` 获取实际长度：

```c
int count;
MPI_Get_count(&status, MPI_INT, &count);
// count 就是实际收到的元素个数
```

### 四种通信模式
MPI 定义了四种发送模式，区别在于什么时候 `MPI_Send` 返回：

| 模式 | 发送函数 | 返回条件 | 适用场景 |
|------|----------|----------|----------|
| 标准模式 | `MPI_Send` | MPI 可以安全重用缓冲区就返回，不一定已经被接收 | 通用场景，最常用 |
| 同步模式 | `MPI_Ssend` | 直到接收方开始接收才返回 | 同步点，对程序正确性更容易推理 |
| 缓冲模式 | `MPI_Bsend` | 立刻返回，用户提供缓冲 | 需要提前知道流量，避免阻塞 |
| 就绪模式 | `MPI_Rsend` | 只有接收已经 posted 才能调用 | 已知接收方准备好了，减少握手 |

大部分情况下用标准模式 `MPI_Send` 就够了。

### 阻塞 vs 非阻塞

- **阻塞通信**：`MPI_Send`/`MPI_Recv` 调用后直到操作完成才返回，不能立刻使用缓冲区
- **非阻塞通信**：`MPI_Isend`/`MPI_Irecv` 立刻返回，操作在后台进行，之后用 `MPI_Wait` 等待完成

**优势：** 可以把计算和通信重叠，掩盖通信延迟。比如先启动通信，然后做计算，等计算完了再等通信完成，这样通信和计算重叠执行，总时间就减少了。

非阻塞基本用法：
```c
MPI_Request req;          // 请求句柄
// 启动发送，立刻返回
MPI_Isend(buf, count, type, dest, tag, comm, &req);
// ... 做一些不依赖发送完成的计算 ...
// 等待发送完成
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

完整示例：[examples/02-core/nonblocking.c](../examples/02-core/nonblocking.c)

### 死锁问题与避免

最常见的死锁写法：两个进程都要给对方发数据，都先 send 再 recv：

```c
// examples/02-core/deadlock.c
#include <mpi.h>

int main(int argc, char** argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data = rank;
    if (rank == 0) {
        // 发给 rank 1
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        // 等 rank 1 发给自己
        MPI_Recv(&data, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Send(&data, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // 两个都卡在 MPI_Send 等对方接收，死锁了...
    MPI_Finalize();
    return 0;
}
```

运行这个程序可能会卡住，就是死锁了。

**避免死锁方法：**
1. **固定顺序：** rank 小的先发，rank 大的先收
2. **非阻塞通信：** `MPI_Isend`/`MPI_Irecv` 都立刻返回，然后再 `MPI_Waitall`
3. **使用 `MPI_Sendrecv`：** MPI 保证这个原子操作不会死锁

### MPI_Sendrecv 交换数据示例：
```c
// rank 0 和 rank 1 交换数据，不会死锁
if (rank == 0) {
    MPI_Sendrecv(&sendbuf, 1, MPI_INT, 1, 0,
                 &recvbuf, 1, MPI_INT, 1, 1,
                 MPI_COMM_WORLD, &status);
} else {
    MPI_Sendrecv(&sendbuf, 1, MPI_INT, 0, 1,
                 &recvbuf, 1, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, &status);
}
```

### 消息匹配机制
接收方匹配消息需要同时满足两个条件：
1. 源 rank 匹配（或你用了 `MPI_ANY_SOURCE` 匹配任意源）
2. 标签 tag 匹配（或你用了 `MPI_ANY_TAG` 匹配任意标签）

所以即使都是发给你的消息，不同标签也不会混，这是保证正确性的重要机制。

## 4. 集合通信（Collective Communication）

集合通信是**通信域中所有进程都必须参与**的通信操作，比你自己用点对点实现更高效，因为 MPI 实现会针对网络拓扑做专门优化。

### 常见集合通信操作图解

#### 1. 屏障同步：`MPI_Barrier`
所有进程都调用 `MPI_Barrier` 之后才会一起继续往下走，用来做同步点：
```c
// 所有进程到达这里，等所有人都到了才一起走
MPI_Barrier(MPI_COMM_WORLD);
```

**常用场景：** 计时，保证所有人都准备好了再开始。

#### 2. 广播：`MPI_Bcast`
从一个根进程把**同一份数据**发给所有进程：
```c
// root 进程把 data 广播给所有进程
MPI_Bcast(data, count, datatype, root, comm);
```

**示意图：**
```
rank 0 (root): [1 2 3 4] → 广播给所有人 → 每个进程都有 [1 2 3 4]
```

#### 3. 发散（Scatter）：`MPI_Scatter`
根进程把数据分成块，每个进程分一块：
```c
// root: sendbuf 里有 size 块，分给每个进程一块
MPI_Scatter(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype,
            root, comm);
```

**示意图（4个进程）：**
```
root: [块0 块1 块2 块3] → 发散 →
rank0: 块0, rank1: 块1, rank2: 块2, rank3:块3
```

#### 4. 收集（Gather）：`MPI_Gather`
正好反过来，每个进程把一块数据收集到根进程拼起来：
```c
MPI_Gather(sendbuf, sendcount, sendtype,
           recvbuf, recvcount, recvtype,
           root, comm);
```

**示意图：**
```
rank0: 块0 ──┐
rank1: 块1 ──┤ → 收集到 root → root: [块0 块1 块2 块3]
rank2: 块2 ──┤
rank3: 块3 ──┘
```

#### 5. 全收集：`MPI_Allgather`
每个进程收集所有块到自己这里，相当于每个进程都拿到全部数据：
```c
MPI_Allgather(sendbuf, sendcount, sendtype,
              recvbuf, recvcount, recvtype,
              comm);
```

**和 Gather 的区别：** 所有进程都拿到了完整结果，不止 root。

#### 6. 全局归约：`MPI_Reduce` / `MPI_Allreduce`
把所有进程的数据做一个归约操作（求和、找最大最小等），结果放到根进程（`MPI_Reduce`）或所有进程（`MPI_Allreduce`）。

比如计算全局总和：
```c
// 每个进程的 local_sum 归约到 root 得到全局总和
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, root, comm);
```

常用操作符：
| 操作符 | 含义 |
|--------|------|
| `MPI_SUM` | 求和 |
| `MPI_MIN` | 找最小值 |
| `MPI_MAX` | 找最大值 |
| `MPI_PROD` | 乘积 |
| `MPI_LAND` | 逻辑与 |
| `MPI_LOR` | 逻辑或 |
| `MPI_BAND` | 按位与 |
| `MPI_BOR` | 按位或 |

**示例：** 求所有进程局部最大值的全局最大值：
```c
double local_max = ...; // 每个进程自己算出来的局部最大值
double global_max;
MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
// 现在只有 rank 0 有 global_max
// 如果用 MPI_Allreduce，所有进程都能拿到 global_max
```

#### 7. 全归约：`MPI_Allreduce`
在深度学习训练中非常常用！比如所有卡算完梯度，然后所有卡都需要梯度平均值，就用 `MPI_Allreduce`。

### 为什么集合通信更快？
- 实现层使用了树型拓扑、环形拓扑等优化算法，减少网络跳数
- 可以利用网络硬件的广播能力
- 更好地利用网络带宽，减少冲突
- MPI 实现厂商（比如 NVIDIA、Mellanox）会针对硬件做深度优化

### 使用原则
> 💡 能用集合通信就不要自己用点对点实现集合操作。标准 MPI 实现的集合通信比你手写的快很多，而且不容易错。

## 完整可运行示例：集合通信求 π

```c
// examples/02-core/collectives.c
// 用 Monte Carlo 方法计算 π，演示 MPI_Reduce 的用法
```

每个进程算自己那块的结果，然后归约求和得到最终 π。

## 练习题

1. 运行 `deadlock.c`，看看是不是会卡住？为什么？
2. 修改 `sendrecv.c`，让 rank 0 发一个数组给 rank 1，rank 1 打印出来
3. 用 `MPI_Bcast` 让 rank 0 把一个数组广播给所有进程，每个进程打印数组，验证对不对
4. 修改 `collectives.c`，改成求最大值，看看结果对不对

## 示例代码

- [sendrecv.c](../examples/02-core/sendrecv.c) - 基本发送接收
- [deadlock.c](../examples/02-core/deadlock.c) - 死锁示例（不要这么写）
- [nonblocking.c](../examples/02-core/nonblocking.c) - 非阻塞通信
- [collectives.c](../examples/02-core/collectives.c) - 集合通信计算 π 示例

## 下一步

→ 下一章：[进阶核心主题](03-advanced.md)
