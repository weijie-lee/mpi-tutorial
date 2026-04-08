# 第二章：核心编程模型

## 本章简介

本章是 MPI 编程的核心基础，带你掌握点对点通信和集合通信的编程模型。

## MPI 程序基本框架

任何 MPI 程序都遵循这个流程：

```c
int main(int argc, char* argv[]) {
    // 1. 初始化
    MPI_Init(&argc, &argv);
    
    // 2. 获取进程信息
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 我的 ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 总共多少进程
    
    // 3. 通信与计算（核心逻辑）
    // ... 各种 MPI 调用 ...
    
    // 4. 结束
    MPI_Finalize();
    return 0;
}
```

## 点对点通信

### 什么是点对点通信？

两个进程之间的**一对一**通信，就像打电话：

```
进程 A  ────── 进程 B
   │            │
   └─ 发送消息 ─┘
```

### 阻塞 vs 非阻塞

| 类型 | 特点 | 风险 |
|------|------|------|
| **阻塞 (Blocking)** | 发送/接收完成后才返回 | 可能死锁 |
| **非阻塞 (Non-blocking)** | 立即返回，后续同步 | 需要小心管理 |

### 阻塞通信 API

```c
// 发送
int MPI_Send(void* buf, int count, MPI_Datatype datatype, 
             int dest, int tag, MPI_Comm comm);

// 接收
int MPI_Recv(void* buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status* status);
```

**参数解释**：
- `buf`：发送/接收数据的缓冲区
- `count`：数据个数
- `datatype`：数据类型
- `dest/source`：目标/源进程 rank
- `tag`：消息标签（用于匹配）
- `comm`：通信域
- `status`：接收状态（仅接收需要）

### 数据类型对照表

| C 类型 | MPI 类型 |
|--------|----------|
| int | MPI_INT |
| float | MPI_FLOAT |
| double | MPI_DOUBLE |
| char | MPI_CHAR |
| long | MPI_LONG |

### 通信示例

```c
if (rank == 0) {
    int data = 42;
    MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    printf("Sent: %d\n", data);
} else if (rank == 1) {
    int data;
    MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Received: %d\n", data);
}
```

### 死锁问题

**死锁 = 所有人都等着别人先动**

```c
// ❌ 错误的顺序导致死锁
if (rank == 0) {
    MPI_Send(...);  // 发给 rank 1
    MPI_Recv(...);  // 等 rank 1 发送
} else if (rank == 1) {
    MPI_Send(...);  // 发给 rank 0
    MPI_Recv(...);  // 等 rank 0 发送
}
// 双方都在等对方接收，死锁！
```

**避免死锁的方法**：
1. 保持发送/接收顺序一致
2. 使用 `MPI_Sendrecv`（发送接收组合）
3. 使用非阻塞通信

## 集合通信

### 什么是集合通信？

多个进程（通常是所有进程）**共同参与**的通信模式，就像开会：

```
         进程 0
        /  |  \
       /   |   \
   进程 1  进程 2  进程 3
```

### 常用集合通信原语

#### 1. 广播 (Broadcast)

一个进程的数据**发送给所有进程**：

```
rank 0: [data] ──→ 所有人收到 [data]
rank 1: [data]
rank 2: [data]
```

```c
int data = 100;
if (rank == 0) {
    // 只有 rank 0 有数据
}
MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
// 现在所有进程都有 data = 100
```

#### 2. 归约 (Reduce)

所有进程的数据**汇总到一个进程**：

```
rank 0: [1] ─┐
rank 1: [2] ─┼→ [6] (rank 0)
rank 2: [3] ─┘
```

```c
int local_data = rank + 1;  // 1, 2, 3
int result;

MPI_Reduce(&local_data, &result, 1, MPI_INT, 
           MPI_SUM, 0, MPI_COMM_WORLD);
// rank 0 得到 result = 6 (1+2+3)
```

**常用归约操作**：
- `MPI_SUM`：求和
- `MPI_PROD`：求积
- `MPI_MAX`：最大值
- `MPI_MIN`：最小值

#### 3. 全归约 (Allreduce)

所有进程都得到归约结果：

```c
MPI_Allreduce(&local_data, &result, 1, MPI_INT, 
              MPI_SUM, MPI_COMM_WORLD);
// 所有进程都得到 result = 6
```

#### 4. 散射 (Scatter)

一个进程的数据**分散**给所有进程：

```
rank 0: [1,2,3] → rank 0:1, rank 1:2, rank 2:3
```

#### 5. 收集 (Gather)

所有进程的数据**收集**到一个进程：

```
rank 0:1, rank 1:2, rank 2:3 → rank 0: [1,2,3]
```

## 通信域与进程组

### 通信域 (Communicator)

通信域 = 进程组 + 通信上下文

- `MPI_COMM_WORLD`：所有进程的默认通信域
- 可以创建自定义通信域，隔离不同任务的通信

### 进程组 (Group)

进程的有序集合：

```c
MPI_Group world_group, new_group;
MPI_Comm_group(MPI_COMM_WORLD, &world_group);

// 创建只包含偶数 rank 的组
int ranks[] = {0, 2, 4, 6};
MPI_Group_incl(world_group, 4, ranks, &new_group);

// 基于组创建新通信域
MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
```

## 示例代码

本章配套示例在 `02-core/` 目录：

```bash
cd ch02-core/02-core
mpicc -o sendrecv sendrecv.c
mpirun -np 2 ./sendrecv
```

- `sendrecv.c` - 点对点通信
- `deadlock.c` - 死锁示例与解决
- `nonblocking.c` - 非阻塞通信
- `collectives.c` - 集合通信
- `all-collectives.c` - 各种集合通信
- `pi_monte_carlo.c` - Monte Carlo 计算 π

## 性能小贴士

1. **减少通信次数**：合并多条小消息为一条大消息
2. **选择合适的通信模式**：阻塞 vs 非阻塞
3. **避免死锁**：注意发送接收顺序
4. **使用集合通信**：比自己实现更高效

## 本章测验

- [ ] 掌握 MPI 基本编程流程
- [ ] 熟练使用 MPI_Send / MPI_Recv
- [ ] 理解阻塞/非阻塞通信区别
- [ ] 掌握广播、归约等集合通信

## 下一步

学完本章后，进入 [第三章：进阶核心主题](./ch03-advanced/README.md) 学习派生类型、拓扑和 RMA。
