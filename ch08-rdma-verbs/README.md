# 第八章：RDMA Verbs 编程入门

## 本章简介

本章深入 RDMA 技术，学习 RDMA Verbs 原生编程接口，这是高性能网络通信的核心技术。

## 知识点

### RDMA Verbs 基础
- **什么是 Verbs**：RDMA 操作的高级 API
- **核心对象**：
  - **IBV_CONTEXT**：RDMA 设备上下文
  - **IBV_pd (Protection Domain)**：保护域
  - **IBV_mr (Memory Region)**：注册的内存区域
  - **IBV_qp (Queue Pair)**：队列对，通信基本单元
  - **IBV_cq (Completion Queue)**：完成队列
  - **IBV_srq (Shared Receive Queue)**：共享接收队列

### RDMA 操作类型
- **Send/Receive**：传统消息模式
- **RDMA Write**：直接写入远程内存
- **RDMA Read**：直接读取远程内存
- **Atomic**：原子操作（Fetch-and-add, Compare-and-swap）

### 通信流程
1. **设备探测**：打开 RDMA 设备
2. **创建 PD**：创建保护域
3. **注册 MR**：将内存注册为 RDMA 可访问
4. **创建 QP**：创建队列对
5. **建立连接**：交换 QP 信息（通常通过 socket）
6. **数据传输**：Post Send/Recv 或 RDMA 操作
7. **完成通知**：轮询 CQ 获取完成事件

## 核心概念

### 创建 Queue Pair
```c
struct ibv_qp_init_attr attr = {
    .send_cq = cq,
    .recv_cq = cq,
    .qp_type = IBV_QPT_RC,
    .sq_sig_all = 1,
};
qp = ibv_create_qp(pd, &attr);
```

### RDMA Write
```c
struct ibv_sge sg = {
    .addr = (uint64_t)local_buf,
    .length = size,
    .lkey = mr->lkey,
};
struct ibv_send_wr wr = {
    .opcode = IBV_WR_RDMA_WRITE,
    .wr.rdma.rkey = remote_mr->rkey,
    .wr.rdma.remote_addr = remote_addr,
    .sg_list = &sg,
    .num_sge = 1,
};
ibv_post_send(qp, &wr, &bad_wr);
```

## 学习目标

1. 理解 RDMA Verbs 核心概念
2. 掌握 RDMA 编程的基本流程
3. 学会实现 RDMA Read/Write
4. 理解连接管理和内存注册

## 示例代码

本章配套示例在 `08-rdma-verbs/` 目录：
- `rdma_server.c` / `rdma_client.c` - RDMA 通信示例

## 下一步

学完本章后，进入 [第九章：Kubernetes 部署](./ch09-kubernetes-pytorchjob/README.md) 学习在 K8s 上运行 MPI 任务。
