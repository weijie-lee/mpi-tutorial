/*
 * examples/04-hardware/rdma_write_server.c
 * RDMA Write 最简单示例 - 服务端
 * 服务端注册一块内存，等待客户端直接 RDMA Write 写进来
 * 不需要服务端 CPU 参与，客户端写完服务端直接看到数据
 * 编译：mpicc -O2 -o rdma_write_server rdma_write_server.c -lrdmacm -libverbs
 * 运行：
 *   服务端：./rdma_write_server [port]
 *   客户端：./rdma_write_client <server-ip> [port] <server-buf-addr> <rkey>
 * （地址和 rkey 服务端启动会打印出来，复制给客户端）
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

    // --------------------------
    // 第一步：获取 RDMA 设备列表
    int num_devices;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        perror("ibv_get_device_list");
        return 1;
    }
    if (num_devices == 0) {
        fprintf(stderr, "No RDMA devices found in this system\n");
        ibv_free_device_list(device_list);
        return 1;
    }

    // 选第一个 RDMA 设备打开
    struct ibv_context* ctx = ibv_open_device(device_list[0]);
    // 用完释放设备列表
    ibv_free_device_list(device_list);
    if (!ctx) {
        perror("ibv_open_device");
        return 1;
    }

    // --------------------------
    // 第二步：分配保护域（Protection Domain）
    // PD 是一个命名空间，MR 和 QP 都要属于同一个 PD
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        perror("ibv_alloc_pd");
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 第三步：分配缓冲区并注册 MR（Memory Region）
    // RDMA 只能访问注册过的内存，需要给内存授予远程访问权限
    char* buf = (char*)malloc(BUF_SIZE);
    if (!buf) {
        perror("malloc");
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }
    memset(buf, 0, BUF_SIZE);

    // 注册 MR，允许本地写和远程写
    struct ibv_mr* mr = ibv_reg_mr(
        pd,             // 所属保护域
        buf,            // 内存起始地址
        BUF_SIZE,       // 内存大小（字节）
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE // 访问权限
    );
    if (!mr) {
        perror("ibv_reg_mr");
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    printf("=== Server info for client ===\n");
    printf("Buffer address (hex): %p\n", (void*)buf);
    printf("Remote key (hex): %#x\n", mr->rkey);
    printf("============================\n");
    printf("Waiting for client connection on port %d...\n", port);

    // --------------------------
    // 第四步：创建完成队列（Completion Queue）
    // RDMA 操作完成后，完成事件会放到 CQ 里，我们轮询 CQ 知道操作完成
    struct ibv_cq* cq = ibv_create_cq(
        ctx,    // ibv context
        16,     // 最多容纳多少完成事件
        NULL,   // 用户上下文
        NULL,   // completion channel
        0       // cq entry size
    );
    if (!cq) {
        perror("ibv_create_cq");
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 第五步：创建队列对（Queue Pair，QP）
    // 每个连接一个 QP，包含发送队列和接收队列
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.send_cq = cq;    // 发送完成放到这个 CQ
    qp_attr.recv_cq = cq;    // 接收完成放到这个 CQ
    qp_attr.cap.max_send_wr  = 16;  // 最大未完成发送请求
    qp_attr.cap.max_recv_wr  = 16;  // 最大未完成接收请求
    qp_attr.cap.max_sge    = 1;   // 最大 scatter-gather 元素
    qp_attr.qp_type        = IBV_QPT_RC; // Reliable Connection 可靠连接
    struct ibv_qp* qp = ibv_create_qp(pd, &qp_attr);
    if (!qp) {
        perror("ibv_create_qp");
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 第六步：rdma_cm 监听连接
    // rdma_cm 是 RDMA connection manager，帮你处理连接建立
    struct rdma_event_channel* ec = rdma_create_event_channel();
    if (!ec) {
        perror("rdma_create_event_channel");
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    struct rdma_cm_id* listen_id;
    int ret = rdma_create_id(ec, &listen_id, NULL, RDMA_PS_TCP);
    if (ret) {
        perror("rdma_create_id");
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }
    // 绑定 QP 到 rdma_cm_id
    listen_id->qp = qp;

    // 绑定地址监听
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    ret = rdma_bind_addr(listen_id, (struct sockaddr*)&addr);
    if (ret) {
        perror("rdma_bind_addr");
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // 开始监听
    ret = rdma_listen(listen_id, 1);
    if (ret) {
        perror("rdma_listen");
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    // --------------------------
    // 第七步：等待客户端连接
    struct rdma_cm_event* event;
    ret = rdma_get_cm_event(ec, &event);
    if (ret) {
        perror("rdma_get_cm_event");
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }

    struct rdma_cm_id* conn_id = event->id;
    // accept 连接
    ret = rdma_accept(conn_id, NULL);
    if (ret) {
        perror("rdma_accept");
        rdma_destroy_id(conn_id);
        rdma_ack_cm_event(event);
        rdma_destroy_id(listen_id);
        rdma_destroy_event_channel(ec);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(cq);
        ibv_dereg_mr(mr);
        free(buf);
        ibv_dealloc_pd(pd);
        ibv_close_device(ctx);
        return 1;
    }
    // ack 事件
    rdma_ack_cm_event(event);
    printf("Client connected\n");
    printf("Waiting for RDMA Write from client...\n");

    // --------------------------
    // 关键点：客户端直接 RDMA Write 写进来，服务端什么都不用做！
    // 不需要 post 接收，不需要处理完成，数据直接就写到内存了
    // 等待几秒让客户端写完，我们直接读
    sleep(3);

    // --------------------------
    // 打印收到的数据
    printf("\n=== Received data from client ===\n");
    printf("'%s'\n", buf);
    printf("============================\n");

    // --------------------------
    // 清理资源
    rdma_destroy_qp(conn_id);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr);
    free(buf);
    ibv_dealloc_pd(pd);
    rdma_destroy_id(listen_id);
    rdma_destroy_id(conn_id);
    rdma_destroy_event_channel(ec);
    ibv_close_device(ctx);

    return 0;
}
