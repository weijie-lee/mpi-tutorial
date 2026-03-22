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
    if (!rc.ctx) {
        printf("Failed to open device\n");
        exit(1);
    }

    // 3. 分配PD
    rc.pd = ibv_alloc_pd(rc.ctx);
    if (!rc.pd) {
        printf("Failed to allocate PD\n");
        exit(1);
    }

    // 4. 分配缓冲区并注册MR
    rc.buffer = malloc(BUFFER_SIZE);
    if (!rc.buffer) {
        printf("Failed to allocate buffer\n");
        exit(1);
    }
    rc.mr = register_mr(rc.pd, rc.buffer, BUFFER_SIZE);
    if (!rc.mr) {
        printf("Failed to register MR\n");
        exit(1);
    }
    printf("MR registered, lkey: 0x%x\n", rc.mr->lkey);

    // 5. 创建CQ
    rc.cq = ibv_create_cq(rc.ctx, 16, NULL, NULL, 0);
    if (!rc.cq) {
        printf("Failed to create CQ\n");
        exit(1);
    }

    // 6. 创建QP
    rc.qp = create_qp(rc.pd, rc.cq, 16);
    if (!rc.qp) {
        printf("Failed to create QP\n");
        exit(1);
    }

    // 7. 获取本地GID
    struct ibv_port_attr port_attr;
    int ret = ibv_query_port(rc.ctx, 1, &port_attr);
    if (ret != 0) {
        printf("Failed to query port\n");
        exit(1);
    }
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
    printf("Now copy this to client, then enter client's QPN and GID:\n");
    printf("> ");

    // 读取客户端地址
    struct rdma_addr client_addr;
    ret = scanf("%u", &client_addr.qpn);
    if (ret != 1) {
        printf("Failed to read QPN\n");
        exit(1);
    }
    unsigned long long subnet, interface;
    ret = scanf("%llx %llx", &subnet, &interface);
    if (ret != 2) {
        printf("Failed to read GID\n");
        exit(1);
    }
    client_addr.gid.global.subnet_prefix = subnet;
    client_addr.gid.global.interface_id = interface;

    // 8. 把QP转到RTR然后RTS
    ret = modify_qp_to_rtr(rc.qp, client_addr.qpn, client_addr.gid);
    if (ret != 0) {
        printf("Failed to modify QP to RTR\n");
        exit(1);
    }
    ret = modify_qp_to_rts(rc.qp);
    if (ret != 0) {
        printf("Failed to modify QP to RTS\n");
        exit(1);
    }

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
    ret = ibv_post_recv(rc.qp, &recv_wr, &bad_recv);
    if (ret != 0) {
        printf("Failed to post recv\n");
        exit(1);
    }

    // 10. 等待接收完成
    struct ibv_wc wc;
    int ne = ibv_poll_cq(rc.cq, 1, &wc);
    while (ne == 0) {
        ne = ibv_poll_cq(rc.cq, 1, &wc);
    }
    if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
        printf("Recv failed: %s\n", ibv_wc_status_str(wc.status));
        exit(1);
    }

    printf("\n>>> Received message from client: %s\n", rc.buffer);

    // 接收成功，清理退出
    ibv_destroy_qp(rc.qp);
    ibv_destroy_cq(rc.cq);
    ibv_dereg_mr(rc.mr);
    ibv_dealloc_pd(rc.pd);
    ibv_close_device(rc.ctx);
    ibv_free_device_list(dev_list);
    free(rc.buffer);

    printf("\nDone\n");
    return 0;
}
