# 七、RDMA Verbs 编程入门

前面我们讲了 MPI 如何使用 RDMA，但如果你想**直接使用 RDMA Verbs 接口编程**（比如写自己的通信库），本章会带你入门。

## 1. RDMA Verbs 基础概念

### 核心概念

| 概念 | 说明 |
|------|------|
| **Verbs** | RDMA 的底层编程接口，类似 socket API 之于TCP/IP |
| **PD (Protection Domain)** | 保护域，用于隔离不同用户的内存和QP |
| **MR (Memory Region)** | 内存区域，注册给RDMA网卡使用，允许网卡访问这块内存 |
| **QP (Queue Pair)** | 队列对，包含发送队列 (SQ) 和接收队列 (RQ)，所有操作都通过QP提交 |
| **CQ (Completion Queue)** | 完成队列，操作完成后网卡在这里放完成通知 |
| **GID** | 全局标识符，每个网口一个，用于标识节点 |
| **LID** | 链路标识符，InfiniBand 上用，RoCE 一般不用 |

### 基本流程

RDMA 通信的基本步骤：

1. **设备探测**：打开 RDMA 设备，获取网口信息
2. **地址交换**：双方交换 GID 和 QPN，建立连接
3. **内存注册**：把要通信的内存注册成 MR
4. **建立 QP**：创建 QP，并把状态调到 Ready 状态
5. **发起操作**：把 RDMA 操作（Send/Recv/Put/Get）提交到 SQ/RQ
6. **等待完成**：从 CQ 拿完成事件，判断操作是否完成
7. **销毁回收**：销毁 QP、释放 MR、关闭设备

## 2. 两种服务类型

RDMA 支持两种基本的服务类型：

### RC (Reliable Connected) - 可靠连接
- 面向连接，保证消息不丢不重不乱序
- 最常用，类似 TCP
- 适合点对点稳定通信

### UD (Unreliable Datagram) - 不可靠数据报
- 无连接，不保证送达，可能丢包乱序
- 类似 UDP
- 适合追求极致性能、上层自己做重传的场景

入门我们先讲 **RC**，绝大多数应用都用这个。

## 3. 完整示例：RDMA 回显服务器

我们写一个最简单的示例：客户端发送数据到服务器，服务器把数据 RDMA 读回去，再发回来。

### 示例代码结构

```
examples/07-rdma-verbs/
├── common.h       # 公共头文件和工具函数
├── server.c       # 服务器端
└── client.c       # 客户端
```

### 第一步：交换地址信息

RDMA 通信前，客户端和服务器需要交换对方的：
- GID（对方网口的地址）
- QPN（队列对编号）

这一步需要你自己用其他方式交换（比如TCP、文件、共享文件系统），我们这里简化成：服务器启动后输出信息，用户手动把信息给客户端。

### 完整代码

先看公共头文件 `common.h`：

```c
// examples/07-rdma-verbs/common.h
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>

#define BUFFER_SIZE 1024
#define GID_INDEX 0  // 通常用第一个GID

// 交换用的地址信息
struct rdma_addr {
    uint32_t qpn;
    union ibv_gid gid;
};

// 全局RDMA资源
struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    char *buffer;
};

// 注册内存
static inline struct ibv_mr *register_mr(struct ibv_pd *pd, void *addr, size_t size) {
    return ibv_reg_mr(pd, addr, size,
        IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE);
}

// 创建QP并初始化状态
static inline struct ibv_qp *create_qp(struct ibv_pd *pd, struct ibv_cq *cq, int qp_size) {
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = qp_size;
    qp_attr.cap.max_recv_wr = qp_size;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    return ibv_create_qp(pd, &qp_attr);
}

// 把QP调到RTR状态
static inline int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, union ibv_gid remote_gid) {
    struct ibv_qp_attr attr = {0};
    struct ibv_qp_init_attr init_attr = {0};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.gid = remote_gid;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER |
        IBV_QP_AV);
}

// 把QP调到RTS状态
static inline int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {0};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}
```

服务器端 `server.c`：

```c
// examples/07-rdma-verbs/server.c
#include "common.h"

int main() {
    // 1. 获取第一个RDMA设备
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        printf("No RDMA devices found\n");
        exit(1);
    }
    struct ibv_device *dev = dev_list[0];

    // 2. 打开设备
    struct rdma_context rc = {0};
    rc.ctx = ibv_open_device(dev);

    // 3. 分配PD
    rc.pd = ibv_alloc_pd(rc.ctx);

    // 4. 分配缓冲区并注册MR
    rc.buffer = malloc(BUFFER_SIZE);
    rc.mr = register_mr(rc.pd, rc.buffer, BUFFER_SIZE);
    printf("MR registered, lkey: 0x%x\n", rc.mr->lkey);

    // 5. 创建CQ
    rc.cq = ibv_create_cq(rc.ctx, 16, NULL, NULL, 0);

    // 6. 创建QP
    rc.qp = create_qp(rc.pd, rc.cq, 16);

    // 7. 获取本地GID
    struct ibv_port_attr port_attr;
    ibv_query_port(rc.ctx, 1, &port_attr);
    union ibv_gid my_gid;
    ibv_query_gid(rc.ctx, 1, GID_INDEX, &my_gid);

    // 输出地址信息，让用户交给客户端
    struct rdma_addr my_addr = {
        .qpn = rc.qp->qp_num,
        .gid = my_gid
    };
    printf("=== Server address info ===\n");
    printf("QPN: %u\n", my_addr.qpn);
    printf("GID: %016llx%016llx\n",
           (unsigned long long)my_addr.gid.global.subnet_prefix,
           (unsigned long long)my_addr.gid.global.interface_id);
    printf("Now copy this to client, enter client's QPN and GID:\n");

    // 读取客户端地址
    struct rdma_addr client_addr;
    scanf("%u", &client_addr.qpn);
    unsigned long long subnet, interface;
    scanf("%llx %llx", &subnet, &interface);
    client_addr.gid.global.subnet_prefix = subnet;
    client_addr.gid.global.interface_id = interface;

    // 8. 把QP转到RTR然后RTS
    modify_qp_to_rtr(rc.qp, client_addr.qpn, client_addr.gid);
    modify_qp_to_rts(rc.qp);

    printf("Connection established. Waiting for client...\n");

    // 9. 投递接收请求
    struct ibv_sge sg = {
        .addr = (uintptr_t)rc.buffer,
        .length = BUFFER_SIZE,
        .lkey = rc.mr->lkey
    };
    struct ibv_recv_wr recv_wr = {
        .wr_id = 1,
        .sg_list = &sg,
        .num_sge = 1,
        .next = NULL
    };
    struct ibv_recv_wr *bad_recv;
    ibv_post_recv(rc.qp, &recv_wr, &bad_recv);

    // 10. 等待接收完成
    struct ibv_wc wc;
    int ne = ibv_poll_cq(rc.cq, 1, &wc);
    if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
        printf("Recv failed: %s\n", ibv_wc_status_str(wc.status));
        exit(1);
    }

    printf("Received message from client: %s\n", rc.buffer);

    // 接收成功，清理退出
    ibv_destroy_qp(rc.qp);
    ibv_destroy_cq(rc.cq);
    ibv_dereg_mr(rc.mr);
    ibv_dealloc_pd(rc.pd);
    ibv_close_device(rc.ctx);
    ibv_free_device_list(dev_list);
    free(rc.buffer);

    printf("Done\n");
    return 0;
}
```

客户端 `client.c`：

```c
// examples/07-rdma-verbs/client.c
#include "common.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <server-qpn> <server-gid-subnet> <server-gid-interface>\n");
        printf("Example: %s 2 123456789abcdef 123456789abcdef\n", argv[0]);
        exit(1);
    }

    // 1. 获取RDMA设备
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    struct rdma_context rc = {0};
    rc.ctx = ibv_open_device(dev_list[0]);
    rc.pd = ibv_alloc_pd(rc.ctx);

    // 2. 准备缓冲区和MR
    rc.buffer = malloc(BUFFER_SIZE);
    strcpy(rc.buffer, "Hello RDMA!");
    rc.mr = register_mr(rc.pd, rc.buffer, BUFFER_SIZE);

    // 3. 创建CQ和QP
    rc.cq = ibv_create_cq(rc.ctx, 16, NULL, NULL, 0);
    rc.qp = create_qp(rc.pd, rc.cq, 16);

    // 4. 获取本地GID并输出
    union ibv_gid my_gid;
    ibv_query_gid(rc.ctx, 1, GID_INDEX, &my_gid);
    printf("Client QPN: %u\n", rc.qp->qp_num);
    printf("Client GID: %016llx%016llx\n",
           (unsigned long long)my_gid.global.subnet_prefix,
           (unsigned long long)my_gid.global.interface_id);

    // 解析服务器地址
    struct rdma_addr server_addr = {
        .qpn = atoi(argv[1]),
    };
    server_addr.gid.global.subnet_prefix = strtoull(argv[2], NULL, 16);
    server_addr.gid.global.interface_id = strtoull(argv[3], NULL, 16);

    // 5. 连接
    modify_qp_to_rtr(rc.qp, server_addr.qpn, server_addr.gid);
    modify_qp_to_rts(rc.qp);

    printf("Connected, sending message...\n");

    // 6. 发送消息
    struct ibv_sge sg = {
        .addr = (uintptr_t)rc.buffer,
        .length = strlen(rc.buffer) + 1,
        .lkey = rc.mr->lkey
    };
    struct ibv_send_wr send_wr = {
        .wr_id = 1,
        .sg_list = &sg,
        .num_sge = 1,
        .opcode = IBV_WR_SEND,
        .send_flags = IBV_SEND_SIGNALED,
        .next = NULL
    };
    struct ibv_send_wr *bad_send;
    ibv_post_send(rc.qp, &send_wr, &bad_send);

    // 7. 等待发送完成
    struct ibv_wc wc;
    int ne = ibv_poll_cq(rc.cq, 1, &wc);
    if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
        printf("Send failed: %s\n", ibv_wc_status_str(wc.status));
        exit(1);
    }

    printf("Message sent successfully\n");

    // 清理
    ibv_destroy_qp(rc.qp);
    ibv_destroy_cq(rc.cq);
    ibv_dereg_mr(rc.mr);
    ibv_dealloc_pd(rc.pd);
    ibv_close_device(rc.ctx);
    ibv_free_device_list(dev_list);
    free(rc.buffer);

    return 0;
}
```

## 4. 编译和运行

### 编译

需要安装 `libibverbs` 开发包：

```bash
# Ubuntu/Debian
apt-get install libibverbs-dev

# RHEL/CentOS
yum install libibverbs-devel
```

Makefile 示例：

```makefile
CC = gcc
CFLAGS = -O2 -Wall
LIBS = -libverbs

all: server client

server: server.c
	$(CC) $(CFLAGS) -o server server.c $(LIBS)

client: client.c
	$(CC) $(CFLAGS) -o client client.c $(LIBS)

clean:
	rm -f server client
```

### 运行

需要两台机器都有 RDMA 网卡，并且网络打通：

**在服务器上运行：**
```bash
./server
=== Server address info ===
QPN: 2
GID: 000000000000000000001cfeffff0001
Now copy this to client, enter client's QPN and GID:
```

**在客户端上运行（粘贴服务器地址）：**
```bash
./client 2 0000000000000000 00001cfeffff0001
Client QPN: 2
Client GID: 000000000000000000001cfeffff0002
Connected, sending message...
Message sent successfully
```

**然后把客户端输出的 QPN 和 GID 粘贴到服务器：**
```
2 0000000000000000 00001cfeffff0002
Connection established. Waiting for client...
Received message from client: Hello RDMA!
Done
```

运行成功！这就是最基本的 RDMA Verbs 通信。

## 5. RDMA 核心操作类型

RDMA 支持几种基本操作：

| Opcode | 说明 |
|--------|------|
| `IBV_WR_SEND` / `IBV_WR_RECV` | 传统的 Send/Recv，数据发到对方内存 |
| `IBV_WR_RDMA_READ` | 本地主动**读**远程内存到本地 |
| `IBV_WR_RDMA_WRITE` | 本地主动**写**本地数据到远程内存 |
| `IBV_WR_ATOMIC_CMP_AND_SWAP` | 原子比较交换 |
| `IBV_WR_ATOMIC_FETCH_AND_ADD` | 原子加法并取回原值 |

### 对比 Send/Recv vs RDMA Read/Write

- **Send/Recv**：双方都参与，发送方发，接收方要提前投接收请求
- **RDMA Read/Write**：单方操作，发起方直接读写对方内存，对方CPU不需要参与
- **优势**：RDMA 读写可以实现真正的零拷贝，远程内存访问不需要对方操作系统介入

## 6. 关键要点和常见坑

### 1. 内存必须注册才能用

RDMA 网卡需要知道你这块内存是允许它访问的，所以必须调用 `ibv_reg_mr` 注册，拿到 `lkey`/`rkey` 才能用。注册有开销，尽量一次注册重复使用，不要每次通信都重新注册。

### 2. GID 索引问题

RoCE v2 需要正确选择 GID 索引，如果你有多个网口，或者 IPv4/IPv6 都有，索引不对会连不通。一般用 `ibv_devinfo` 命令查看：

```bash
ibv_devinfo
# 看 GID table 找到你要用的那个GID，索引从0开始
```

### 3. MTU 协商错误

两端 MTU 必须一致，代码里我们写死 `IBV_MTU_1024`，如果你的卡不支持会连接失败，改成对应 MTU。

### 4. 完成事件轮询

`ibv_poll_cq` 是轮询，返回 0 表示没有完成事件，需要你继续轮。不要一次只轮一次就睡觉，这样延迟会很高。低延迟应用通常会忙等轮询。

### 5. 权限控制

注册 MR 时，根据需要给权限：
- 本地写：`IBV_ACCESS_LOCAL_WRITE`
- 允许远程读：`IBV_ACCESS_REMOTE_READ`
- 允许远程写：`IBV_ACCESS_REMOTE_WRITE`
不给对权限会报错 `REMOTE_ACCESS_ERROR`。

## 7. 进一步学习

- [RDMAmojo](https://www.rdmamojo.com/) - 很好的RDMA编程教程
- [libibverbs 官方文档](https://www.rdmaconsortium.org/) - 权威文档
- [Linux RDMA stack](https://github.com/linux-rdma/rdma-core) - 源码在这里

## 示例代码

- [common.h](../examples/07-rdma-verbs/common.h) - 公共工具函数
- [server.c](../examples/07-rdma-verbs/server.c) - 服务器端示例
- [client.c](../examples/07-rdma-verbs/client.c) - 客户端示例

## 回到目录

→ [返回首页](../README.md)
