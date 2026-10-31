// RDMA Verbs 客户端：Send，然后 RDMA Write，再 RDMA Read 回来核对。
// 编译：make
// 运行：./client <server-ip> [port]

#include "common.h"

static int connect_retry(const char *ip, int port) {
    for (int i = 0; i < 50; i++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return -1;
        }
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
            close(fd);
            return -1;
        }
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            return fd;
        }
        close(fd);
        usleep(100000);
    }
    return -1;
}

static int post_and_wait(struct verbs_res *local, struct ibv_send_wr *wr) {
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(local->qp, wr, &bad);
    if (rc) {
        fprintf(stderr, "ibv_post_send: %s\n", strerror(rc));
        return -1;
    }
    struct ibv_wc wc;
    int ne = verbs_poll(local->cq, &wc);
    if (ne != 1 || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "completion failed: %s\n",
                ne == 1 ? ibv_wc_status_str(wc.status) : "timeout");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <server-ip> [port]\n", argv[0]);
        return 1;
    }
    const char *ip = argv[1];
    int port = (argc >= 3) ? atoi(argv[2]) : 18516;

    struct verbs_res local;
    if (verbs_open(&local) != 0) {
        printf("RESULT: SKIP no active RDMA device\n");
        return 0;
    }
    if (verbs_create(&local) != 0) {
        perror("verbs_create");
        verbs_destroy(&local);
        return 1;
    }

    memcpy(local.buffer, "hello-verbs", sizeof("hello-verbs"));
    memcpy(local.buffer + 256, "write-ok", sizeof("write-ok"));

    int conn = connect_retry(ip, port);
    if (conn < 0) {
        perror("connect");
        verbs_destroy(&local);
        return 1;
    }

    char server_line[256];
    if (verbs_read_line(conn, server_line, sizeof(server_line)) <= 0) {
        fprintf(stderr, "no server info\n");
        close(conn);
        verbs_destroy(&local);
        return 1;
    }
    struct peer_info remote;
    char mine[256];
    verbs_format(&local, mine, sizeof(mine));
    if (verbs_parse(server_line, &remote) || verbs_write_all(conn, mine) ||
        verbs_to_rts(&local, &remote)) {
        fprintf(stderr, "connection setup failed\n");
        close(conn);
        verbs_destroy(&local);
        return 1;
    }

    char ready[64];
    if (verbs_read_line(conn, ready, sizeof(ready)) <= 0 || strcmp(ready, "READY") != 0) {
        fprintf(stderr, "server was not ready\n");
        close(conn);
        verbs_destroy(&local);
        return 1;
    }

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local.buffer;
    sge.length = sizeof("hello-verbs");
    sge.lkey = local.mr->lkey;
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    int failed = post_and_wait(&local, &wr);

    if (!failed) {
        sge.addr = (uintptr_t)local.buffer + 256;
        sge.length = sizeof("write-ok");
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = 2;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote.addr + 512;
        wr.wr.rdma.rkey = remote.rkey;
        failed = post_and_wait(&local, &wr);
    }

    if (!failed) {
        sge.addr = (uintptr_t)local.buffer + 512;
        sge.length = sizeof("write-ok");
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = 3;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remote.addr + 512;
        wr.wr.rdma.rkey = remote.rkey;
        failed = post_and_wait(&local, &wr);
        if (!failed && strcmp(local.buffer + 512, "write-ok") != 0) {
            fprintf(stderr, "RDMA Read mismatch: '%s'\n", local.buffer + 512);
            failed = 1;
        } else if (!failed) {
            printf("RDMA Read back: '%s'\n", local.buffer + 512);
        }
    }

    if (!failed && verbs_write_all(conn, "DONE\n")) {
        failed = 1;
    }
    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    close(conn);
    verbs_destroy(&local);
    return failed ? 1 : 0;
}
