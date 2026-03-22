# 二、核心编程模型

## 1. 基本概念

### 通信域（Communicator）与进程组（Group）
- **进程组（Group）**：一组参与通信的进程的集合
- **通信域（Communicator）**：包装了进程组和通信上下文，给消息提供一个独立的"命名空间"
- 默认通信域：`MPI_COMM_WORLD`，包含所有启动的进程

**为什么需要通信域？** 把不同的通信隔离开，避免不同模块的消息互相匹配错。

### 进程秩（Rank）
- 每个进程在通信域中会被分配一个唯一的整数编号，从 `0` 开始计数
- 通信时，源和目的都是用 rank 来标识
- 作用：让每个进程知道自己的"身份"，从而决定做什么计算

### 消息结构
一个 MPI 消息包含：
1. **数据缓冲区**：要发送的数据所在内存地址
2. **数据长度**：多少个元素
3. **数据类型**：每个元素是什么类型（MPI_INT，MPI_DOUBLE 等）
4. **标签（Tag）**：用户自定义的整数，用来区分不同类型的消息
5. **源/目的秩**：消息从哪来，到哪去

## 2. 程序基本结构

所有 MPI 程序都遵循这个结构：
```c
#include <mpi.h>

int main(int argc, char** argv) {
    // 1. 初始化 MPI 环境
    MPI_Init(&argc, &argv);

    // 2. 获取当前进程信息
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 3. 你的计算和通信逻辑...

    // 4. 结束 MPI 环境
    MPI_Finalize();
    return 0;
}
```

这部分已经在第一个示例中看到了，下面看核心的通信操作。

## 3. 点对点通信（Point-to-Point）

点对点通信就是**一个进程发，一个进程收**，是最基础的通信方式。

### 基础阻塞通信：`MPI_Send` / `MPI_Recv`

```c
// 发送
int MPI_Send(
    const void *buf,      // 发送缓冲区
    int count,            // 发送元素个数
    MPI_Datatype datatype,// 元素数据类型
    int dest,             // 目标进程 rank
    int tag,              // 消息标签
    MPI_Comm comm         // 通信域
);

// 接收
int MPI_Recv(
    void *buf,            // 接收缓冲区
    int count,            // 最多接收多少元素
    MPI_Datatype datatype,// 数据类型
    int source,           // 源进程 rank
    int tag,              // 消息标签（MPI_ANY_TAG 匹配任意标签）
    MPI_Comm comm,        // 通信域
    MPI_Status *status    // 返回状态（包含实际标签、错误码等）
);
```

示例：rank 0 发消息给 rank 1

```c
// examples/02-core/sendrecv.c
```

### 四种通信模式
MPI 定义了四种发送模式，区别在于什么时候`MPI_Send`返回：

| 模式 | 发送函数 | 返回条件 | 适用场景 |
|------|----------|----------|----------|
| 标准模式 | `MPI_Send` | MPI 可以安全重用缓冲区就返回，不一定已经被接收 | 通用场景，最常用 |
| 同步模式 | `MPI_Ssend` | 直到接收方开始接收才返回 | 同步点，对程序正确性更容易推理 |
| 缓冲模式 | `MPI_Bsend` | 立刻返回，用户提供缓冲 | 需要提前知道流量，避免阻塞 |
| 就绪模式 | `MPI_Rsend` | 只有接收已经 posted 才能调用 | 已知接收方准备好了，减少握手 |

### 阻塞 vs 非阻塞

- **阻塞通信**：`MPI_Send`/`MPI_Recv` 调用后直到操作完成才返回，不能立刻使用缓冲区
- **非阻塞通信**：`MPI_Isend`/`MPI_Irecv` 立刻返回，操作在后台进行，之后用 `MPI_Wait` 等待完成

**优势：** 可以把计算和通信重叠，掩盖通信延迟。

非阻塞基本用法：
```c
MPI_Request req;          // 请求句柄
MPI_Isend(buf, count, type, dest, tag, comm, &req); // 启动发送，立刻返回
// ... 做一些不依赖发送完成的计算 ...
MPI_Wait(&req, MPI_STATUS_IGNORE); // 等待发送完成
```

示例：[examples/02-core/nonblocking.c](../examples/02-core/nonblocking.c)

### 死锁问题与避免

最常见的死锁写法：
```c
if (rank == 0) {
    MPI_Send(buf, 10, MPI_INT, 1, 0, comm);
    MPI_Recv(buf, 10, MPI_INT, 1, 1, comm, &status);
} else if (rank == 1) {
    MPI_Send(buf, 10, MPI_INT, 0, 1, comm);
    MPI_Recv(buf, 10, MPI_INT, 0, 0, comm, &status);
}
```
两个都先 send 再 recv，但如果 MPI_Send 是同步模式，就会互相等待，死锁。

**避免方法：**
1. 固定发送接收顺序：比如 rank 小先发，rank 大先收
2. 使用非阻塞通信：`MPI_Isend`/`MPI_Irecv` 立刻返回，然后 `MPI_Waitall` 等待
3. 使用 `MPI_Sendrecv`：原子完成发送和接收，MPI 保证不死锁

### 消息匹配机制
接收方匹配消息的条件同时满足：
1. 源 rank 匹配（或使用 `MPI_ANY_SOURCE`）
2. 标签 tag 匹配（或使用 `MPI_ANY_TAG`）

所以即使都是发给你的消息，不同标签也不会混。

## 4. 集合通信（Collective Communication）

集合通信是**通信域中所有进程都参与**的通信操作，比你自己用点对点实现更高效，因为 MPI 实现会针对网络拓扑做专门优化。

### 常见集合通信操作

#### 1. 屏障同步：`MPI_Barrier`
所有进程都调用 `MPI_Barrier` 之后才会一起继续往下走，用来做同步：
```c
MPI_Barrier(MPI_COMM_WORLD);
```

#### 2. 广播：`MPI_Bcast`
从一个根进程把相同的数据发给所有进程：
```c
// root 进程把 data 广播给所有进程
MPI_Bcast(data, count, datatype, root, comm);
```
![MPI_Bcast 示意图](https://user-images.githubusercontent.com/.../bcast.png)

#### 3. 发散（Scatter）：`MPI_Scatter`
根进程把数据按块分给每个进程，每个进程分到一块：
```c
// root: 发送缓冲区，每个进程一块
// 每个进程接收一块到 recvbuf
MPI_Scatter(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype,
            root, comm);
```

#### 4. 收集（Gather）：`MPI_Gather`
正好反过来，每个进程把一块数据收集到根进程：
```c
MPI_Gather(sendbuf, sendcount, sendtype,
           recvbuf, recvcount, recvtype,
           root, comm);
```

#### 5. 全收集：`MPI_Allgather`
每个进程收集所有块到自己这里，相当于每个进程都拿到全部数据：
```c
MPI_Allgather(sendbuf, sendcount, sendtype,
              recvbuf, recvcount, recvtype,
              comm);
```

#### 6. 全局归约：`MPI_Reduce` / `MPI_Allreduce`
把所有进程的数据做一个归约操作（sum、min、max、乘、与等），结果放到根进程（`MPI_Reduce`）或所有进程（`MPI_Allreduce`）。

比如计算全局总和：
```c
// 每个进程的 local_sum 归约到 root 得到全局总和
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, root, comm);
```
常用操作符：
- `MPI_SUM`: 求和
- `MPI_MIN`: 找最小值
- `MPI_MAX`: 找最大值
- `MPI_PROD`: 乘积
- `MPI_LAND`/`MPI_LOR`: 逻辑与/或

### 为什么集合通信更快？
- 实现层使用了树型拓扑、环形拓扑等优化算法，减少网络跳数
- 可以利用网络硬件的广播能力
- 更好地利用网络带宽，减少冲突

### 使用原则
> 能用集合通信就不要自己用点对点实现集合操作。标准 MPI 实现的集合通信比你手写的快很多，而且不容易错。

## 示例代码

- [sendrecv.c](../examples/02-core/sendrecv.c) - 基本发送接收
- [deadlock.c](../examples/02-core/deadlock.c) - 死锁示例（不要这么写）
- [nonblocking.c](../examples/02-core/nonblocking.c) - 非阻塞通信
- [collectives.c](../examples/02-core/collectives.c) - 集合通信示例

## 下一步

→ 下一章：[进阶核心主题](03-advanced.md)
