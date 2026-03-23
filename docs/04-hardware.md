# 四、硬件结合：GPU 与 RDMA 支持

现代大规模并行计算（尤其是深度学习训练）都是多节点多GPU架构，MPI 对 GPU 和 RDMA 的原生支持非常关键。本章从基础概念到实践，完整介绍 RDMA 和 GPU-aware MPI。

---

## 第一部分：RDMA 完整教程

### 1. RDMA 基础概念

#### 什么是 RDMA

**RDMA** = **Remote Direct Memory Access**，**远程直接内存访问**，是一种**硬件加速的网络通信技术**。

传统 TCP/IP 通信数据路径：
```
发送方：应用 → 内核缓冲区 → 网卡 → 网络 → 接收方网卡 → 内核缓冲区 → 应用
                       ↑                ↑             
                 多次内存拷贝，内核中断，CPU全程参与
```

RDMA 通信数据路径：
```
发送方：应用内存 → 网卡 → 网络 → 接收方网卡 → 接收方应用内存
                         ↑             
          内核完全旁路，不需要CPU参与拷贝，直接DMA
```

#### RDMA 核心优势

| 优势 | 说明 |
|------|------|
| **更低延迟** | 3-10 微秒 vs TCP 20-50 微秒 |
| **更高带宽** | 接近线速，利用率 >95% |
| **更低CPU占用** | 内核旁路，CPU不用处理网络拷贝，可以专心做计算 |
| **更好扩展性** | 大规模集群下，CPU不会被网络占满 |

#### RDMA 三种基本操作

| 操作 | 说明 |
|------|------|
| **RDMA Write** | 本地内存直接写到远程内存，不需要远程CPU参与 |
| **RDMA Read** | 直接从远程内存读数据到本地，不需要远程CPU参与 |
| **SEND/RECV** | 类似传统消息传递，需要对方CPU参与接收 |

#### RDMA 硬件类型

| 类型 | 说明 | 应用场景 |
|------|------|----------|
| **InfiniBand (IB)** | 专用高性能网络，传统HPC主流 | 超级计算机、AI训练集群 |
| **RoCE v2** | RDMA over Converged Ethernet，以太网上跑RDMA | 现代数据中心、云厂商AI集群 |
| **iWARP** | 标准以太网RDMA，基于TCP | 广域RDMA场景 |

现在 AI 训练集群主流用 **RoCE v2**，可以复用现有以太网基础设施，成本更低，性能和 IB 差不多。

---

### 2. RDMA 关键概念理解

#### 内存注册（Memory Registration）和 MR

RDMA 网卡需要直接访问应用内存，必须先把内存**注册**给RDMA驱动：

- 注册后得到一个 **Memory Region (MR)**
- MR 包含地址、长度、访问权限、远程访问密钥（rkey/lkey）
- 只有注册过的内存才能被RDMA网卡直接访问
- 注册有开销，尽量一次性注册，重复使用，不要频繁注册销毁

```c
// libverbs 注册内存基本流程
struct ibv_mr *mr = ibv_reg_mr(pd, buf, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
// 使用...
ibv_dereg_mr(mr);
```

#### 保护域（Protection Domain, PD）

- 保护域是一个命名空间，把MR和QP归属到一起
- 只有同一个PD下的MR才能被QP使用
- 相当于一个安全隔离单元

#### 队列对（Queue Pair, QP）

- RDMA 所有操作都通过 QP 完成
- 一个 QP 包含两个队列：
  - **发送队列（SQ）**：放要发送的请求
  - **接收队列（RQ）**：放完成的请求
- 每个QP对应一个到对端QP的连接
- 一般一个QP对应一个连接，一对多需要多个QP

#### 完成队列（Completion Queue, CQ）

- 操作完成后，完成事件会放到CQ里
- 应用通过轮询或等待CQ获取操作完成通知
- 可以多个QP共享同一个CQ

#### 状态转移

QP 需要经历状态转移才能正常工作：
```
RESET → INIT → RTR → RTS
↑      ↑     ↑
↑      │     └── 对端地址信息就绪，可以接收
│      └── 本地信息就绪
└── 初始状态
```

代码里通常由 rdma_cm 库帮你处理这个，不用手动写。

---

### 3. RDMA 编程模型（基于 libverbs + rdma_cm）

#### 最简单的 RDMA Write 示例

服务端（接收方）注册内存，等待客户端写：

```c
// examples/04-hardware/rdma_write_server.c
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    // 1. 创建保护域
    struct ibv_context *ctx = ibv_open_device(ibv_get_device_list(NULL));
    struct ibv_pd *pd = ibv_alloc_pd(ctx);

    // 2. 分配并注册内存
    char *buf = malloc(1024);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, 1024, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

    // 3. 创建CQ和QP
    struct ibv_cq *cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
    struct ibv_qp_init_attr qp_attr = {
        .send_cq = cq,
        .recv_cq = cq,
        .cap = { .max_send_wr = 16, .max_recv_wr = 16, .max_sge = 1 },
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);

    // 4. rdma_cm 监听连接...
    // （完整代码见示例文件）

    // 连接建立后，客户端可以直接RDMA Write写到服务端buf
    // 服务端不需要任何动作，数据就来了！

    printf("Received data: %s\n", buf);
    return 0;
}
```

> 💡 关键点：**服务端不需要调用任何发送接收函数，客户端直接写进来**，这就是"远程直接内存访问"。

完整可运行代码见 [rdma_write_server.c](../examples/04-hardware/rdma_write_server.c) 和 [rdma_write_client.c](../examples/04-hardware/rdma_write_client.c)。

---

### 4. RDMA 编程两步走：建立连接 + 数据传输

#### 建立连接（使用 rdma_cm）

```c
// 服务端
struct rdma_cm_id *listen_id;
struct rdma_event_channel *ec = rdma_create_event_channel();
rdma_create_id(ec, &listen_id, NULL, RDMA_PS_TCP);
rdma_bind_addr(listen_id, (struct sockaddr *)&addr);
rdma_listen(listen_id, 10);

// 等待连接事件，accept
rdma_get_cm_event(ec, &event);
struct rdma_cm_id *conn_id = event->id;
rdma_accept(conn_id, NULL);

// 客户端
rdma_create_id(ec, &conn_id, NULL, RDMA_PS_TCP);
rdma_connect(conn_id, &addr, NULL);
```

连接建立后，双方交换 **对方的 MR 信息**：地址、rkey。交换完就可以开始RDMA操作了。

#### 发送 RDMA Write 请求

```c
struct ibv_send_wr wr;
struct ibv_sge sge;

// 本地要发送的数据
sge.addr = (uintptr_t)local_buf;
sge.length = length;
sge.lkey = local_mr->lkey;

// RDMA Write 到远程
wr.wr_id = 0;
wr.sg_list = &sge;
wr.num_sge = 1;
wr.opcode = IBV_WR_RDMA_WRITE;
wr.rdma.remote_addr = remote_buf_addr;
wr.rdma.rkey = remote_rkey;
wr.next = NULL;

ibv_post_send(qp, &wr, NULL);
```

然后等完成队列：
```c
struct ibv_wc wc;
while (ibv_poll_cq(cq, 1, &wc) < 1);
if (wc.status != IBV_WC_SUCCESS) {
    // 出错了
}
```

完成！数据已经写到远程内存了，全程不需要远程CPU参与。

---

### 5. 什么时候用 RDMA Read，什么时候用 RDMA Write

| 场景 | 推荐操作 |
|------|----------|
| 生产者推送数据给消费者 | RDMA Write |
| 消费者主动拉取数据 | RDMA Read |
| 双方需要握手确认 | SEND/RECV |

在深度学习训练中，AllReduce 经常用 **一对多 RDMA Write + 多对一 Reduce + 多对一 RDMA Write 回发**，充分利用RDMA的低延迟高带宽。

---

### 6. MPI 如何使用 RDMA

主流 MPI 都内置了 RDMA 支持，**你不需要自己写 libverbs 代码**，MPI 会自动帮你用好 RDMA。

| MPI 实现 | RDMA 支持 | 启用方式 |
|----------|-----------|----------|
| OpenMPI | ✅ | 编译时 `--enable-verbs`，运行时自动检测 |
| MVAPICH | ✅ | 原生针对InfiniBand/RDMA优化，默认启用 |
| Intel MPI | ✅ | 自动检测RDMA设备，默认启用 |
| MPICH | ✅ | 需要编译时启用verbs支持 |

只要MPI库编译时支持了RDMA，**不需要修改你的应用代码**，MPI会自动选择最优的通信路径。

---

### 7. 如何检查 MPI 是否支持 RDMA

```bash
# OpenMPI 检查 verbs 支持
ompi_info --parsable | grep -i verbs
# 如果输出有 "MCA verbs: available = 1" 就是支持

# 检查可用的 BTL (Byte Transfer Layer)
ompi_info --parsable | grep btl:
# 看到 `openib` 说明支持RDMA

# 检查 ibv_devinfo 看看系统有没有RDMA设备
ibv_devinfo
# 会列出所有 RDMA 设备
```

### 8. OpenMPI 强制使用 RDMA 网络

如果系统有多个网卡，可以强制 MPI 只用 RDMA：

```bash
# 只使用 openib (IB/RDMA) 和 self (进程内)
mpirun --mca btl openib,self -np 8 ./your_app

# 禁用TCP，优先用RDMA
mpirun --mca btl ^tcp -np 8 ./your_app

# 指定IB端口（如果有多个）
mpirun --mca btl_openib_if_include mlx5_0 -np 8 ./your_app
```

---

### 9. RDMA 性能对比

| 指标 | TCP/IP (100G Ethernet) | RDMA (100G RoCE) |
|------|-------------------|-------------------------|
| 单消息 8KB 延迟 | ~25-40 微秒 | **~5-8 微秒** |
| 峰值带宽利用率 | ~75% | **~95-98%** |
| 64KB 消息带宽 | ~70 Gbps | **~95 Gbps** |
| CPU 占用（满带宽） | ~2-4 核心 | **<0.5 核心** |

在多节点多GPU训练中，RDMA 能减少 30%-50% 通信时间，提升整体训练吞吐量。

---

## 第二部分：GPU 与 GPU-aware MPI

### 1. 为什么需要 GPU-aware MPI？

如果没有 GPU-aware MPI，要发送GPU上的数据必须这么做：

```c
// 不支持 GPU-aware 时代码：需要两次额外拷贝
cudaMemcpy(cpu_buf, gpu_buf, size, cudaMemcpyDeviceToHost);  // GPU → CPU
MPI_Send(cpu_buf, size, MPI_BYTE, dest, tag, comm);            // CPU 发送
// 接收方：
MPI_Recv(cpu_buf, size, MPI_BYTE, source, tag, comm, status); // CPU 接收
cudaMemcpy(gpu_buf, cpu_buf, size, cudaMemcpyHostToDevice);  // CPU → GPU
```

这额外的两次 PCIe 拷贝非常浪费时间，还浪费 PCIe 带宽。

**GPU-aware MPI 可以直接发送设备端内存**，不需要 CPU 拷贝：

```c
// GPU-aware MPI：直接发设备指针！
MPI_Send(gpu_buf, count, MPI_FLOAT, dest, tag, comm);
```

MPI 底层会直接用 GPU 数据地址走 RDMA，省去拷贝。

### 2. 主流实现支持

| MPI 实现 | CUDA 支持 | ROCm 支持 | 启用方式 |
|----------|-----------|-----------|----------|
| OpenMPI | ✅ 原生支持 | ✅ | `./configure --with-cuda=/usr/local/cuda` |
| Intel MPI | ✅ | ❌ | 环境变量 `I_MPI_CUDA=1` |
| MPICH | ✅ | ✅ | `--enable-g=yes --with-cuda=/usr/local/cuda` |
| MVAPICH | ✅ | ✅ | 原生带GPU支持 |

### 3. CUDA-aware MPI 完整示例

```cpp
// examples/04-hardware/cuda_aware.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    const int N = 1024;
    float *d_buf;
    cudaMalloc(&d_buf, N * sizeof(float));

    if (rank == 0) {
        // 初始化数据在GPU上
        float *h_buf = new float[N];
        for (int i = 0; i < N; i++) {
            h_buf[i] = i;
        }
        cudaMemcpy(d_buf, h_buf, N * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_buf;
        printf("Rank 0: sending %d floats from GPU to rank 1\n", N);
        // 直接发送GPU指针！不需要拷贝到CPU
        MPI_Send(d_buf, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        // 直接接收放到GPU
        MPI_Recv(d_buf, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 拷回CPU验证
        float *h_buf = new float[N];
        cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Rank 1: received first 5 elements: %f %f %f %f %f\n",
               h_buf[0], h_buf[1], h_buf[2], h_buf[3], h_buf[4]);
        // 验证正确性
        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (h_buf[i] != i) {
                ok = false;
                break;
            }
        }
        printf("Verification: %s\n", ok ? "PASS" : "FAIL");
        delete[] h_buf;
    }

    cudaFree(d_buf);
    MPI_Finalize();
    return 0;
}
```

### 4. 编译运行

```bash
cd examples/04-hardware
nvcc -c cuda_aware.cu -o cuda_aware.o
mpic++ cuda_aware.o -o cuda_aware -lcuda
mpirun -np 2 ./cuda_aware
```

如果运行输出 `Verification: PASS`，说明你的 MPI 正确支持 CUDA-aware。

### 5. GPU + RDMA 最佳组合：GPU Direct RDMA

**GPU Direct RDMA** = GPU 数据直接走 RDMA，**完全不需要 CPU  involvement**：

```
GPU显存 → GPU SMC → NIC → 网络 → 对端 NIC → 对端 GPU SMC → 对端显存
                          ↑
            没有CPU拷贝，没有PCIe拷贝到CPU再回来
```

这是目前多节点多GPU训练最快的通信路径：
- 需要 GPU-aware MPI + RDMA 同时支持
- 需要 NVIDIA GPU 支持（Pascal 架构之后都支持）
- 需要主板/BIOS 开启 PCIe ACS 绕过
- OpenMPI 4.0+ 原生支持

开启后，AllReduce 等集合通信延迟能再降 20%-30%。

---

## 第三部分：常见问题排查

### Q: 我的MPI编译时没有CUDA支持怎么办？

**A:** 需要重新编译MPI，编译时指定CUDA路径。以OpenMPI为例：
```bash
./configure --prefix=/opt/openmpi --with-cuda=/usr/local/cuda
make -j$(nproc) && make install
```

或者使用你的包管理器安装预编译的支持CUDA的MPI版本（比如`openmpi-cuda`等）。

### Q: 运行CUDA-aware程序报错了怎么办？

常见错误和解决：

1. **"CUDA device pointer not recognized" / "Invalid rank buffer"**  
   → 说明你的MPI不支持CUDA-aware，重新编译MPI或者换支持的版本。

2. **cudaMemcpy failed after MPI communication**  
   → 检查是否正确绑定GPU，每个进程绑定到不同的GPU：`cudaSetDevice(rank % n_gpus)`

3. **运行结果不对，数据传错了**  
   → 检查数据类型和计数：`MPI_FLOAT`对应`float*`，`MPI_DOUBLE`对应`double*`，不要错配。

### Q: 系统有RDMA设备但MPI不用怎么办？

检查：
1. MPI是否编译时支持verbs（参考上文检查方法）
2. 是否有防火墙/ACL限制了RDMA端口（RDMA用10-bit域，通常需要配置IPoIB或DC路由）
3. 尝试强制指定BTL：`--mca btl openib,self`

### Q: GPU Direct RDMA 不工作怎么办？

检查：
1. 你的GPU是否支持？（Kepler 不支持，Pascal+ 支持）
2. BIOS是否开启了ACS绕过？很多超算主板默认需要手动开
3. OpenMPI是否启用了GPU Direct RDMA？`ompi_info | grep gdr` 应该看到 `cuda_gdr_support = 1`

---

## 示例代码

| 示例 | 说明 |
|------|------|
| [cuda_aware.cu](../examples/04-hardware/cuda_aware.cu) | CUDA-aware MPI 基础示例，验证正确性 |
| [rdma_write_server.c](../examples/04-hardware/rdma_write_server.c) | RDMA Write 服务端最简示例 |
| [rdma_write_client.c](../examples/04-hardware/rdma_write_client.c) | RDMA Write 客户端，直接写服务端内存 |

## 下一步

→ 下一章：[上层栈结合：NCCL 与 PyTorch](05-stack.md)

→ 想看 RDMA 在 NCCL/PyTorch 中的实际表现？看 [第十章：全链路观测实战](10-fullstack-observe.md)
