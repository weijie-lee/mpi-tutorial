/*
 * RDMA Write 最简单示例 - 服务端
 * 服务端注册一块内存，等待客户端直接RDMA Write写进来
 * 不需要服务端CPU参与，客户端写完服务端直接看到数据
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#define BUF_SIZE 1024

int main(int argc, char** argv) {
    int port = 12345;
    if (argc >= 2) {
        port = atoi(argv[1]);
    }

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

    // 选第一个设备
    struct ibv_context* ctx = ibv_open_device(device_list[0]);
    ibv_free_device_list(device_list);

    // 2. 分配保护域
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        perror("ibv_alloc_pd");
        return 1;
    }

    // 3. 分配缓冲区并注册MR
    char* buf = (char*)malloc(BUF_SIZE);
    if (!buf) {
        perror("malloc");
        return 1;
    }
    memset(buf, 0, BUF_SIZE);

    // 注册MR，允许远程写
    struct ibv_mr* mr = ibv_reg_mr(pd, buf, BUF_SIZE,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
        perror("ibv_reg_mr");
        return 1;
    }

    printf("Server: buffer registered, addr = %p, rkey = %u\n", (void*)buf, mr->rkey);

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

    // 6. rdma_cm 监听连接
    struct rdma_event_channel* ec = rdma_create_event_channel();
    if (!ec) {
        perror("rdma_create_event_channel");
        return 1;
    }

    struct rdma_cm_id* listen_id;
    int ret = rdma_create_id(ec, &listen_id, NULL, RDMA_PS_TCP);
    if (ret) {
        perror("rdma_create_id");
        return 1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    ret = rdma_bind_addr(listen_id, (struct sockaddr*)&addr);
    if (ret) {
        perror("rdma_bind_addr");
        return 1;
    }

    ret = rdma_listen(listen_id, 1);
    if (ret) {
        perror("rdma_listen");
        return 1;
    }

    printf("Server: listening on port %d\n", port);
    printf("Server: waiting for client connection...\n");

    // 等待连接事件
    struct rdma_cm_event* event;
    ret = rdma_get_cm_event(ec, &event);
    if (ret) {
        perror("rdma_get_cm_event");
        return 1;
    }

    struct rdma_cm_id* conn_id = event->id;
    ret = rdma_accept(conn_id, NULL);
    if (ret) {
        perror("rdma_accept");
        return 1;
    }
    rdma_ack_cm_event(event);
    printf("Server: client connected\n");

    // 交换信息：把我们的buf地址和rkey发给客户端（这里简化，实际需要对方已经知道了
    printf("Server: ready, waiting for client to write...\n");

    // 客户端直接写进来，我们什么都不用做，等一会儿读结果
    sleep(2); // 等客户端写

    printf("\nServer: data received: '%s'\n", buf);

    // 清理
    rdma_destroy_qp(conn_id->qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    free(buf);
    rdma_destroy_id(listen_id);
    rdma_destroy_event_channel(ec);
    ibv_close_device(ctx);

    return 0;
}
