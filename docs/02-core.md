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

下面列出**所有常用集合通信原语**，包括定义、用法、深度学习场景对应：

---

### 1. 屏障同步：`MPI_Barrier`
**定义**：所有进程都调用 `MPI_Barrier` 之后才会一起继续往下走，用来做同步点。

```c
// 所有进程到达这里，等所有人都到了才一起走
MPI_Barrier(MPI_COMM_WORLD);
```

**深度学习对应场景**：
- 初始化结束后，开始训练前同步
- 验证前同步，确保所有进程都完成训练一轮
- 检查点保存前同步

---

### 2. 广播：`MPI_Bcast`
**定义**：从一个根进程把**同一份数据**发给所有进程，调用完后所有进程都有这份数据。

```c
// root 进程把 data 广播给所有进程
// data: 缓冲区（root 是发送数据，非root是接收数据）
// count: 元素个数
// datatype: 数据类型
// root: 根进程 rank
// comm: 通信域
MPI_Bcast(data, count, datatype, root, comm);
```

**示意图：**
```
rank 0 (root): [1 2 3 4] → 广播给所有人 → 每个进程都得到 [1 2 3 4]
```

**深度学习对应场景**：
- 初始化：root 加载预训练权重，广播给所有进程，保证所有进程初始权重一致
- 每隔 N 轮，root 把当前最好模型参数广播给所有进程

---

### 3. 发散：`MPI_Scatter`
**定义**：根进程把数组分成连续块，每个进程分一块，根分完后每个进程只拿到自己那块。

```c
// root 端 sendbuf 每个进程一块，分发出去
// sendbuf: 根发送缓冲区
// sendcount: 每个进程拿多少个元素
// sendtype: 元素类型
// recvbuf: 接收缓冲区（每个进程自己的）
// recvcount: 接收多少个
// recvtype: 接收类型
// root: 根 rank
// comm: 通信域
MPI_Scatter(sendbuf, sendcount, sendtype,
            recvbuf, recvcount, recvtype,
            root, comm);
```

**示意图（4个进程）：**
```
root: [块0 块1 块2 块3] → 发散 →
rank0: 块0, rank1: 块1, rank2: 块2, rank3: 块3
```

**深度学习对应场景**：
- 数据并行：整个数据集在 root，分给每个进程，每个进程只训练自己分片
- 参数分片：模型参数按层分给不同进程，做模型并行

---

### 4. 发散（变长）：`MPI_Scatterv`
**定义**：和 `MPI_Scatter` 类似，但每个进程分到的长度可以不一样，每个进程长度不同。

```c
// sendcounts: 数组，每个进程对应的长度
// displs: 每个块在 sendbuf 中的偏移（字节数不对，是元素个数偏移！）
MPI_Scatterv(sendbuf, sendcounts, displs, sendtype,
             recvbuf, recvcount, recvtype,
             root, comm);
```

**深度学习对应场景**：
- 不平衡数据分片，每个进程样本数不一样
- 异构集群，不同进程显存不一样，batch size 不同

---

### 5. 收集：`MPI_Gather`
**定义**：正好反过来，每个进程把自己一块数据收集到根进程，根拼起来。

```c
// 每个进程 sendbuf 一块，收集到 root 的 recvbuf
// 每个进程放一块，按 rank 顺序拼
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

**深度学习对应场景**：
- 评估：每个进程算自己分片验证集指标，收集到 root 汇总算整体指标
- 分布式搜索：每个进程搜不同超参，收集所有结果到 root 选最优

---

### 6. 收集（变长）：`MPI_Gatherv`
**定义**：和 `MPI_Gather` 类似，但每个进程发送长度可以不一样。

```c
MPI_Gatherv(sendbuf, sendcount, displs, sendtype,
            recvbuf, recvcounts, displs, recvtype,
            root, comm);
```

---

### 7. 全收集：`MPI_Allgather`
**定义**：每个进程把自己一块收集到**所有**进程，所有进程都拿到完整的拼好结果。

```c
// 和 Gather 区别：收集完结果所有进程都有，不止 root
MPI_Allgather(sendbuf, sendcount, sendtype,
              recvbuf, recvcount, recvtype,
              comm);
```

**和 Gather 的区别：** Gather 只有 root 有结果，Allgather 所有进程都有结果。

**深度学习对应场景**：
- 分布式 embedding：每个进程负责一部分词向量，所有进程需要所有词向量才能做训练，Allgather 把所有分片拼起来给所有人
- 每个进程统计自己batch的指标，所有进程都要拿到所有指标

---

### 8. 全收集（变长）：`MPI_Allgatherv`
**定义**：Allgather 的变长版本，每个进程长度不同。

---

### 9. 全局归约：`MPI_Reduce`
**定义**：把所有进程的数据按归约操作（求和、最大、最小等）计算，结果放在 root 进程。

```c
// 每个进程的 localval 做归约，结果放到 root 的 globalval
// op: 归约操作，比如 MPI_SUM 就是求和
MPI_Reduce(&localval, &globalval, count, datatype, op, root, comm);
```

**常用操作符：**

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

**深度学习对应场景**：
- 多个进程验证，每个进程算自己的误差，归约到 root 算平均误差
- 所有进程统计梯度范数，root 找最大梯度范数用来做梯度裁剪

---

### 10. 全归约：`MPI_Allreduce`
**定义**：归约完结果**所有进程**都得到，不只是 root。

```c
// 每个进程 localval，归约完所有进程都得到 globalval
MPI_Allreduce(&localval, &globalval, count, datatype, op, comm);
```

**这是深度学习训练最常用的集合通信！** 每次迭代完，所有卡都需要平均梯度，Allreduce 正好干这个用。

**深度学习对应场景**：
- 数据并行训练：每个卡算完自己batch梯度，Allreduce 做梯度平均，所有卡梯度一致
- 分布式 batch norm：每个卡统计 mean/var，Allreduce 得到全局统计量

**和 Reduce 区别：** Reduce 只有 root 有结果，Allreduce 所有进程都有结果。数据平行训练每个卡都要更新梯度，所以必须用 Allreduce。

---

### 11. 归约分散：`MPI_ReduceScatter`
**定义**：先归约再分散，每个进程得到归约结果的一块。比 Allreduce 更高效，适合大张量。

```c
// sendbuf 每个进程一块，先归约，结果分给各个进程，每个进程一块
MPI_ReduceScatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
```

**深度学习对应场景**：
- 大数据量 Allreduce，模型参数分片，归约完每个进程拿自己那块更新，比完整 Allreduce 更高效，减少通信量

---

### 12. 扫描：`MPI_Scan`
**定义**：前缀归约，每个进程 i 得到 rank 0..i 的归约结果。

```c
// 每个进程输入 val，输出 out[i] = op(val[0..i])
MPI_Scan(&val, &out, count, datatype, op, comm);
```

**深度学习对应场景**：
- 分层扫描，累计梯度统计
- 动态规划分布式，每个步骤需要前缀和

---

### 13. 全对全交换：`MPI_Alltoall`
**定义**：每个进程给每个进程发一块，每个进程从每个进程收一块，所有交换一次完成。

```c
// sendbuf[i] 是发给 rank i 的块，recvbuf[i] 是从 rank i 收到的块
MPI_Alltoall(sendbuf, sendcount, sendtype,
             recvbuf, recvcount, recvtype, comm);
```

**深度学习对应场景**：
- 流水线并行：每个层在不同进程，前向反向需要交换激活/梯度，Alltoall 一次交换所有
- 专家混合 MoE：每个token路由到不同专家，需要 Alltoall 交换

---

## 为什么集合通信更快？
- 实现层使用了树型拓扑、环形拓扑等优化算法，减少网络跳数
- 可以利用网络硬件的广播能力
- 更好地利用网络带宽，减少冲突
- MPI 实现厂商（比如 NVIDIA、Mellanox）会针对硬件做深度优化

## 使用原则
> 💡 能用集合通信就不要自己用点对点实现集合操作。标准 MPI 实现的集合通信比你手写的快很多，而且不容易错。

## 完整可运行示例：集合通信求 π

```c
// examples/02-core/pi_monte_carlo.c
// 使用 Monte Carlo 方法计算 π，演示 MPI_Reduce 实际应用
```

每个进程并行投自己那部分点，最后 Reduce 求和得到总数。

## 练习题

1. 运行 `deadlock.c`，看看是不是会卡住？为什么？
2. 修改 `sendrecv.c`，让 rank 0 发一个数组给 rank 1，rank 1 打印出来
3. 用 `MPI_Bcast` 让 rank 0 把一个数组广播给所有进程，每个进程打印数组，验证对不对
4. 修改 `collectives.c`，改成求最大值，看看结果对不对

## 示例代码

- [sendrecv.c](../examples/02-core/sendrecv.c) - 基本发送接收
- [deadlock.c](../examples/02-core/deadlock.c) - 死锁示例（不要这么写）
- [nonblocking.c](../examples/02-core/nonblocking.c) - 非阻塞通信
- [collectives.c](../examples/02-core/collectives.c) - 多个集合通信完整示例
- [pi_monte_carlo.c](../examples/02-core/pi_monte_carlo.c) - Monte Carlo 计算 π 完整示例

## 下一步

→ 下一章：[进阶核心主题](03-advanced.md)
