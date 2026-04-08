/*
 * examples/04-hardware/rdma_write_client.c
 * RDMA Write 最简单示例 - 客户端
 * 连接服务端，拿到服务端暴露的内存地址和 rkey，直接 RDMA Write 写进去
 * 不需要服务端 CPU 参与
 * 编译：mpicc -O2 -o rdma_write_client rdma_write_client.c -lrdmacm -libverbs
 * 运行：./rdma_write_client <server-ip> <port> <server-buf-addr-hex> <rkey-hex>
 * 地址和 rkey 从服务端输出复制过来
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#define BUF_SIZE 1024

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <server-ip> <port> <remote-buf-addr-hex> <remote-rkey-hex>\n", argv[0]);
        fprintf(stderr, "Example: %s 192.168.1.100 12345 0xaaaabbbb 0x1234\n", argv[0]);
        return 1;
    }

    char* server_ip = argv[1];
    int port = atoi(argv[2]);
    uintptr_t remote_addr = (uintptr_t)strtoull(argv[3], NULL, 16);
    uint32_t remote_rkey = (uint32_t)strtoul(argv[4], NULL, 16);

    printf("Client: connecting to %s:%d\n", server_ip, port);
    printf("Client: remote buffer addr = %p, rkey = %#x\n", (void*)remote_addr, remote_rkey);

    // --------------------------
    // 第一步：获取 RDMA 设备，和服务端一样
    int num_devices;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        perror("ibv_get_device_list");
        return 1;
    }
    if (num_devices == 0) {
        fprintf(stderr, "No RDMA devices found\n");
        ibv_free_device_list(device_list);
        return 1;
    }

    struct ibv_context* ctx = ibv_open_device(device_list[0]);
    ibv_free_device_list(device_list);
    if (!ctx) {
        perror("ibv_open_device");
        return 1;
    }

    // --------------------------
    // 分配保护域
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        perror("ibv_alloc_pd");
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 分配发送缓冲区，注册 MR
    char* send_buf = (char*)malloc(BUF_SIZE);
    strncpy(send_buf, "Hello from RDMA client! This message is written directly to server memory via RDMA.", BUF_SIZE);

    struct ibv_mr* send_mr = ibv_reg_mr(pd, send_buf, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!send_mr) {
        perror("ibv_reg_mr");
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 创建完成队列
    struct ibv_cq* cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
    if (!cq) {
        perror("ibv_create_cq");
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 创建队列对 QP
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.cap.max_send_wr  = 16;
    qp_attr.cap.max_recv_wr  = 16;
    qp_attr.cap.max_send_sge = 1;   // 最大 send scatter-gather 元素
    qp_attr.cap.max_recv_sge = 1;   // 最大 recv scatter-gather 元素
    qp_attr.cap.max_inline_data = 0; // 不使用 inline data
    qp_attr.qp_type        = IBV_QPT_RC;
    struct ibv_qp* qp = ibv_create_qp(pd, &qp_attr);
    if (!qp) {
        perror("ibv_create_qp");
        ibv_destroy_cq(cq);
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // rdma_cm 连接服务端
    struct rdma_event_channel* ec = rdma_create_event_channel();
    if (!ec) {
        perror("rdma_create_event_channel");
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    struct rdma_cm_id* conn_id;
    int ret = rdma_create_id(ec, &conn_id, NULL, RDMA_PS_TCP);
    if (ret) {
        perror("rdma_create_id");
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }
    // 绑定我们创建的 QP 到 rdma_cm_id
    conn_id->qp = qp;

    // 连接服务端地址
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(server_ip);
    addr.sin_port = htons(port);

    printf("Client: resolving address...\n");
    ret = rdma_resolve_addr(conn_id, NULL, (struct sockaddr*)&addr, 1000);
    if (ret) {
        perror("rdma_resolve_addr");
        rdma_destroy_id(conn_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // 等待连接完成
    struct rdma_cm_event* event;
    ret = rdma_get_cm_event(ec, &event);
    if (ret) {
        perror("rdma_get_cm_event");
        rdma_destroy_id(conn_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }
    rdma_ack_cm_event(event);
    printf("Client: connected\n");

    // --------------------------
    // 现在连接建立好了，执行 RDMA Write！
    // 直接把我们的 send_buf 写到服务端暴露的远程地址
    struct ibv_sge sge;
    struct ibv_send_wr wr;

    // 描述我们要发送的内存块
    sge.addr = (uintptr_t)send_buf;    // 本地数据地址
    sge.length = strlen(send_buf) + 1; // 长度，包括末尾 '\0'
    sge.lkey = send_mr->lkey;       // 本地 MR 的 lkey

    // 准备 RDMA Write 工作请求
    wr.wr_id = 1;          // 我们自己给这个请求一个 id，完成时会返回
    wr.sg_list = &sge;    // 散列 gather 列表
    wr.num_sge = 1;       // 一个块
    wr.opcode = IBV_WR_RDMA_WRITE; // 操作类型：RDMA Write
    wr.wr.rdma.remote_addr = remote_addr; // 远程地址，就是服务端 buf 地址
    wr.wr.rdma.rkey = remote_rkey;      // 远程 rkey，服务端给的
    wr.next = NULL;

    // --------------------------
    // 提交发送请求给硬件
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        perror("ibv_post_send");
        fprintf(stderr, "ret = %d\n", ret);
        rdma_destroy_id(conn_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(send_mr);
        free(send_buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    printf("Client: RDMA Write posted\n");

    // --------------------------
    // 等待 RDMA Write 完成
    struct ibv_wc wc;
    int done = 0;
    while (!done) {
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0) {
            perror("ibv_poll_cq");
            break;
        }
        if (n == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "RDMA Write failed: %s\n", ibv_wc_status_str(wc.status));
                done = 1;
                ret = 1;
            } else {
                printf("Client: RDMA Write completed successfully!\n");
                done = 1;
            }
        }
    }

    // --------------------------
    // 清理
    rdma_destroy_qp(conn_id);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(send_mr);
    free(send_buf);
    ibv_dealloc_pd(pd);
    rdma_destroy_id(conn_id);
    rdma_destroy_event_channel(ec);
    ibv_close_device(ctx);

    return ret;
}
