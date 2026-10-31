# 第八章：RDMA Verbs 编程入门

## 本章简介

本章深入 RDMA 技术，学习 RDMA Verbs 原生编程接口，这是高性能网络通信的核心技术。

## RDMA Verbs 概述

### 什么是 Verbs？

**Verbs** = **API**（应用程序接口）

RDMA Verbs 是 RDMA 硬件的编程接口，类似于：
- socket API → TCP/IP 网络
- RDMA Verbs → RDMA 网络

### 为什么需要原生 RDMA 编程？

- **更细粒度控制**：直接管理 RDMA 硬件
- **极致性能**：去掉中间层开销
- **学习原理**：理解 RDMA 底层机制

## 核心对象

### 1. IBV_CONTEXT（设备上下文）

打开 RDMA 设备：

```c
int num = 0;
struct ibv_device **list = ibv_get_device_list(&num);
struct ibv_context *context = ibv_open_device(list[0]);  // 参数是 ibv_device*，不是 NULL
```

### 2. IBV_pd（保护域）

保护域 = 安全隔离单位

```c
struct ibv_pd *pd;
pd = ibv_alloc_pd(context);
```

### 3. IBV_mr（内存注册）

将内存注册为 RDMA 可访问：

```c
struct ibv_mr *mr;
mr = ibv_reg_mr(pd, buffer, size, 
                 IBV_ACCESS_LOCAL_WRITE |
                 IBV_ACCESS_REMOTE_READ |
                 IBV_ACCESS_REMOTE_WRITE);
```

**权限标志**：
- `IBV_ACCESS_LOCAL_WRITE`：本地可写
- `IBV_ACCESS_REMOTE_READ`：远程可读
- `IBV_ACCESS_REMOTE_WRITE`：远程可写

### 4. IBV_qp（队列对）

队列对 = 通信基本单元

包含：
- **SQ** (Send Queue)：发送队列
- **RQ** (Receive Queue)：接收队列

```c
struct ibv_qp_init_attr init_attr = {
    .send_cq = cq,
    .recv_cq = cq,
    .qp_type = IBV_QPT_RC,  // 可靠连接
    .cap = {
        .max_send_wr = 100,
        .max_recv_wr = 100,
    },
    .sq_sig_all = 1,
};

struct ibv_qp *qp;
qp = ibv_create_qp(pd, &init_attr);
```

### 5. IBV_cq（完成队列）

通知操作完成：

```c
struct ibv_cq *cq;
cq = ibv_create_cq(context, 100, NULL, NULL, 0);
```

## RDMA 操作类型

### 1. Send/Receive（消息模式）

传统消息传递方式：

```c
// 接收方准备接收
struct ibv_sge sge = {...};
struct ibv_recv_wr wr = {.sg_list = &sge, .num_sge = 1};
ibv_post_recv(qp, &wr, &bad_wr);

// 发送方发送
struct ibv_send_wr wr = {
    .opcode = IBV_WR_SEND,
    .sg_list = &sge,
    .num_sge = 1,
};
ibv_post_send(qp, &wr, &bad_wr);
```

### 2. RDMA Write（远程写入）

直接写入远程内存：

```c
struct ibv_send_wr wr = {
    .opcode = IBV_WR_RDMA_WRITE,
    .wr.rdma.rkey = remote_mr->rkey,  // 远程内存 key
    .wr.rdma.remote_addr = remote_addr, // 远程地址
    .sg_list = &sge,
    .num_sge = 1,
};
ibv_post_send(qp, &wr, &bad_wr);
```

### 3. RDMA Read（远程读取）

直接读取远程内存：

```c
struct ibv_send_wr wr = {
    .opcode = IBV_WR_RDMA_READ,
    .wr.rdma.rkey = remote_mr->rkey,
    .wr.rdma.remote_addr = remote_addr,
    .sg_list = &sge,
    .num_sge = 1,
};
ibv_post_send(qp, &wr, &bad_wr);
```

## 通信流程

### 完整流程

```
Server                              Client
  │                                    │
  ├─ 打开设备 (ibv_open_device)        │
  ├─ 创建 PD (ibv_alloc_pd)            │
  ├─ 注册内存 (ibv_reg_mr)             │
  ├─ 创建 CQ (ibv_create_cq)            │
  ├─ 创建 QP (ibv_create_qp)            │
  │                                    │
  │←────── 连接建立 (通过 socket) ──────→│
  │         交换 QP 号、MR 信息          │
  │                                    │
  ├─ 修改 QP 状态为 RTS                │
  │                                    ├─ 修改 QP 状态为 RTS
  │                                    │
  ├──── RDMA Write / Read ──────────→ │
  │                                    │
  ├──── 轮询 CQ 完成 ─────────────────→│
  │                                    │
```

### QP 状态转换

```
INIT → RTR (Ready To Receive) → RTS (Ready To Send)
         ↑                          ↓
         └──────── RESET ←─────────┘
```

跳过 INIT 时，直接改到 RTR 会失败。RTS 还要带 timeout、retry_cnt 和 rnr_retry。RoCE 上 LID 经常是 0，地址句柄要走 GID，并且 `hop_limit` 至少为 1。

```c
/* RESET → INIT：端口、pkey、访问权限 */
/* INIT → RTR：对端 QP、PSN、MTU、地址句柄 */
/* RTR → RTS：本端 PSN、timeout、retry、rnr_retry */
```

示例用 TCP 交换 `qpn`、LID、rkey、地址和 GID。以前靠人手抄两行格式不一致的 GID，对端 QP 根本建不起来。

## 示例代码

本章配套示例在 `08-rdma-verbs/` 目录：

```bash
cd ch08-rdma-verbs/08-rdma-verbs
make
./server 18516 &
./client 127.0.0.1 18516
```

没有 active 端口时打印 `RESULT: SKIP`。有设备时，客户端依次做 Send、RDMA Write 和 RDMA Read，服务端检查 Send 的内容和偏移 512 上的 Write。

- `server.c` - 投递 Receive，再检查 RDMA Write
- `client.c` - Send、RDMA Write、RDMA Read
- `common.h` - INIT / RTR / RTS 和带外交换

## 本章测验

- [ ] 理解 RDMA Verbs 核心概念
- [ ] 掌握 RDMA 编程流程
- [ ] 实现 RDMA Read/Write
- [ ] 理解连接管理和内存注册

## 下一步

学完本章后，进入 [第九章：Kubernetes 部署](../ch09-kubernetes-pytorchjob/README.md)。状态机和带外交换见 [基础知识补充](../docs/基础知识补充.md)。
