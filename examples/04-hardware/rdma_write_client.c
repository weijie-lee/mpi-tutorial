/*
 * RDMA Write 最简单示例 - 客户端
 * 连接服务端，拿到服务端内存地址和rkey，直接RDMA Write写进去
 * 不需要服务端CPU参与
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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <server-ip> <port>\n", argv[0]);
        return 1;
    }

    char* server_ip = argv[1];
    int port = atoi(argv[2]);

    // 1. 获取RDMA设备
    int num_devices;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        perror("ibv_get_device_list");
        return 1;
    }
    if (num_devices == 0) {
        fprintf(stderr, "No RDMA devices found\n");
        return 1;
    }

    struct ibv_context* ctx = ibv_open_device(device_list[0]);
    ibv_free_device_list(device_list);

    // 2. 分配保护域
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        perror("ibv_alloc_pd");
        return 1;
    }

    // 3. 分配我们自己的发送缓冲区
    char* send_buf = (char*)malloc(BUF_SIZE);
    strncpy(send_buf, "Hello from RDMA client!", BUF_SIZE);

    struct ibv_mr* send_mr = ibv_reg_mr(pd, send_buf, BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!send_mr) {
        perror("ibv_reg_mr");
        return 1;
    }

    // 4. 创建完成队列
    struct ibv_cq* cq = ibv_create_cq(ctx, 16, NULL, NULL, 0);
    if (!cq) {
        perror("ibv_create_cq");
        return 1;
    }

    // 5. 创建队列对
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.cap.max_send_wr  = 16;
    qp_attr.cap.max_recv_wr  = 16;
    qp_attr.cap.max_sge    = 1;
    qp_attr.qp_type        = IBV_QPT_RC;

    struct ibv_qp* qp = ibv_create_qp(pd, &qp_attr);
    if (!qp) {
        perror("ibv_create_qp");
        return 1;
    }

    // 6. rdma_cm 连接服务端
    struct rdma_event_channel* ec = rdma_create_event_channel();
    if (!ec) {
        perror("rdma_create_event_channel");
        return 1;
    }

    struct rdma_cm_id* conn_id;
    int ret = rdma_create_id(ec, &conn_id, NULL, RDMA_PS_TCP);
    if (ret) {
        perror("rdma_create_id");
        return 1;
    }
    conn_id->qp = qp; // 绑定我们创建的QP

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(server_ip);
    addr.sin_port = htons(port);

    printf("Client: connecting to %s:%d\n", server_ip, port);
    ret = rdma_connect(conn_id, (struct sockaddr*)&addr, NULL);
    if (ret) {
        perror("rdma_connect");
        return 1;
    }

    // 等待连接完成
    struct rdma_cm_event* event;
    ret = rdma_get_cm_event(ec, &event);
    if (ret) {
        perror("rdma_get_cm_event");
        return 1;
    }
    rdma_ack_cm_event(event);
    printf("Client: connected\n");

    // 在实际应用中，这里需要通过out-of-band交换服务端的地址和rkey
    // 这里我们简化，假设已经知道服务端暴露的buf信息
    // 实际中你可以通过TCP先交换这些元数据再RDMA

    uintptr_t remote_addr = (uintptr_t)strtoull(argv[3], NULL, 16);
    uint32_t remote_rkey = (uint32_t)strtoul(argv[4], NULL, 16);

    printf("Client: remote addr = %p, rkey = %#x\n", (void*)remote_addr, remote_rkey);

    // 7. 准备RDMA Write
    struct ibv_sge sge;
    struct ibv_send_wr wr;

    sge.addr = (uintptr_t)send_buf;
    sge.length = strlen(send_buf) + 1;
    sge.lkey = send_mr->lkey;

    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.rdma.remote_addr = remote_addr;
    wr.rdma.rkey = remote_rkey;
    wr.next = NULL;

    // 8. 发送RDMA Write
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
        perror("ibv_post_send");
        fprintf(stderr, "ret = %d\n", ret);
        return 1;
    }

    printf("Client: RDMA Write posted\n");

    // 9. 等待完成
    struct ibv_wc wc;
    int done = 0;
    while (!done) {
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0) {
            perror("ibv_poll_cq");
            return 1;
        }
        if (n == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "RDMA Write failed: %s\n", ibv_wc_status_str(wc.status));
                return 1;
            }
            done = 1;
        }
    }

    printf("Client: RDMA Write completed successfully!\n");
    printf("Client: data '%s' has been written to server memory directly\n", send_buf);

    // 清理
    rdma_destroy_qp(conn_id->qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(send_mr);
    ibv_dealloc_pd(pd);
    free(send_buf);
    rdma_destroy_id(conn_id);
    rdma_destroy_event_channel(ec);
    ibv_close_device(ctx);

    return 0;
}
