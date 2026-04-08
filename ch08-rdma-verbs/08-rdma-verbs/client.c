#include "common.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <server-qpn> <server-gid-subnet> <server-gid-interface>\n", argv[0]);
        printf("Example: %s 2 0000000000000000 00001cfeffff0001\n", argv[0]);
        exit(1);
    }

    // 1. 获取RDMA设备
    int num_devices;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        printf("No RDMA devices found\n");
        exit(1);
    }
    struct ibv_device *dev = dev_list[0];

    struct rdma_context rc = {0};
    rc.ctx = ibv_open_device(dev);
    if (!rc.ctx) {
        printf("Failed to open device\n");
        exit(1);
    }

    rc.pd = ibv_alloc_pd(rc.ctx);
    if (!rc.pd) {
        printf("Failed to allocate PD\n");
        exit(1);
    }

    // 2. 准备缓冲区和MR
    rc.buffer = malloc(BUFFER_SIZE);
    if (!rc.buffer) {
        printf("Failed to allocate buffer\n");
        exit(1);
    }
    strcpy(rc.buffer, "Hello RDMA! This is a message from client.");
    rc.mr = register_mr(rc.pd, rc.buffer, BUFFER_SIZE);
    if (!rc.mr) {
        printf("Failed to register MR\n");
        exit(1);
    }

    // 3. 创建CQ和QP
    rc.cq = ibv_create_cq(rc.ctx, 16, NULL, NULL, 0);
    if (!rc.cq) {
        printf("Failed to create CQ\n");
        exit(1);
    }

    rc.qp = create_qp(rc.pd, rc.cq, 16);
    if (!rc.qp) {
        printf("Failed to create QP\n");
        exit(1);
    }

    // 4. 获取本地GID并输出
    union ibv_gid my_gid;
    ibv_query_gid(rc.ctx, 1, GID_INDEX, &my_gid);
    printf("=== Client address info (paste this to server) ===\n");
    printf("Client QPN: %u\n", rc.qp->qp_num);
    printf("Client GID: %016llx %016llx\n",
           (unsigned long long)my_gid.global.subnet_prefix,
           (unsigned long long)my_gid.global.interface_id);

    // 解析服务器地址
    struct rdma_addr server_addr = {
        .qpn = (uint32_t)atoi(argv[1]),
    };
    server_addr.gid.global.subnet_prefix = strtoull(argv[2], NULL, 16);
    server_addr.gid.global.interface_id = strtoull(argv[3], NULL, 16);

    // 5. 连接
    int ret = modify_qp_to_rtr(rc.qp, server_addr.qpn, server_addr.gid);
    if (ret != 0) {
        printf("Failed to modify QP to RTR\n");
        exit(1);
    }
    ret = modify_qp_to_rts(rc.qp);
    if (ret != 0) {
        printf("Failed to modify QP to RTS\n");
        exit(1);
    }

    printf("\nConnected, sending message...\n");

    // 6. 发送消息
    struct ibv_sge sg = {
        .addr = (uintptr_t)rc.buffer,
        .length = strlen(rc.buffer) + 1,  // include null terminator
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
    ret = ibv_post_send(rc.qp, &send_wr, &bad_send);
    if (ret != 0) {
        printf("Failed to post send\n");
        exit(1);
    }

    // 7. 等待发送完成
    struct ibv_wc wc;
    int ne = ibv_poll_cq(rc.cq, 1, &wc);
    while (ne == 0) {
        ne = ibv_poll_cq(rc.cq, 1, &wc);
    }
    if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
        printf("Send failed: %s\n", ibv_wc_status_str(wc.status));
        exit(1);
    }

    printf("\n>>> Message sent successfully!\n");

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
