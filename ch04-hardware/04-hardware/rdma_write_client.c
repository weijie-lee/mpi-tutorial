/*
 * RDMA Write 客户端。
 * 编译：gcc -O2 -Wall -Wextra -o rdma_write_client rdma_write_client.c -libverbs
 * 运行：./rdma_write_client <server-ip> [port]
 */

#include "rdma_common.h"

static int connect_with_retry(const char *ip, int port) {
    for (int attempt = 0; attempt < 50; attempt++) {
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <server-ip> [port]\n", argv[0]);
        return 1;
    }
    const char *ip = argv[1];
    int port = (argc >= 3) ? atoi(argv[2]) : 18515;

    struct rdma_res local;
    if (rdma_open(&local) != 0) {
        printf("RESULT: SKIP no active RDMA device\n");
        return 0;
    }
    if (rdma_create_qp(&local) != 0) {
        perror("create qp");
        rdma_close(&local);
        return 1;
    }

    const char *payload = "hello-rdma-write";
    memcpy(local.buf, payload, strlen(payload) + 1);

    int conn = connect_with_retry(ip, port);
    if (conn < 0) {
        perror("connect");
        rdma_close(&local);
        return 1;
    }

    char line[256];
    if (rdma_read_line(conn, line, sizeof(line)) <= 0) {
        fprintf(stderr, "failed to read server peer info\n");
        close(conn);
        rdma_close(&local);
        return 1;
    }
    struct rdma_peer remote;
    if (rdma_parse_peer(line, &remote)) {
        fprintf(stderr, "bad server peer info: %s\n", line);
        close(conn);
        rdma_close(&local);
        return 1;
    }
    rdma_format_peer(&local, 0, line, sizeof(line));
    if (rdma_write_all(conn, line) || rdma_qp_to_rts(&local, &remote)) {
        fprintf(stderr, "could not bring QP to RTS\n");
        close(conn);
        rdma_close(&local);
        return 1;
    }
    if (rdma_read_line(conn, line, sizeof(line)) <= 0 || strcmp(line, "RTS") != 0) {
        fprintf(stderr, "server did not reach RTS\n");
        close(conn);
        rdma_close(&local);
        return 1;
    }

    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t)local.buf;
    sge.length = (uint32_t)strlen(payload) + 1;
    sge.lkey = local.mr->lkey;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote.addr;
    wr.wr.rdma.rkey = remote.rkey;

    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(local.qp, &wr, &bad);
    if (rc) {
        fprintf(stderr, "ibv_post_send: %s\n", strerror(rc));
        close(conn);
        rdma_close(&local);
        return 1;
    }

    struct ibv_wc wc;
    int ne = rdma_poll(local.cq, &wc, RDMA_TIMEOUT_MS);
    int failed = 1;
    if (ne == 1 && wc.status == IBV_WC_SUCCESS) {
        failed = 0;
        printf("RDMA Write completion: success\n");
    } else if (ne == 1) {
        fprintf(stderr, "RDMA Write completion: %s\n", ibv_wc_status_str(wc.status));
    } else {
        fprintf(stderr, "RDMA Write timed out\n");
    }

    if (!failed) {
        if (rdma_write_all(conn, "DONE\n")) {
            failed = 1;
        }
    }
    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");

    close(conn);
    rdma_close(&local);
    return failed ? 1 : 0;
}
