// RDMA Verbs 服务端：先投递 Receive，再接受客户端的 Send，最后检查 RDMA Write。
// 编译：make
// 运行：./server [port]

#include "common.h"

int main(int argc, char **argv) {
    int port = (argc >= 2) ? atoi(argv[1]) : 18516;
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

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) || listen(listen_fd, 1)) {
        perror("listen");
        close(listen_fd);
        verbs_destroy(&local);
        return 1;
    }
    printf("LISTENING %d\n", port);
    fflush(stdout);
    int conn = accept(listen_fd, NULL, NULL);
    if (conn < 0) {
        perror("accept");
        close(listen_fd);
        verbs_destroy(&local);
        return 1;
    }

    char line[256];
    verbs_format(&local, line, sizeof(line));
    if (verbs_write_all(conn, line) || verbs_read_line(conn, line, sizeof(line)) <= 0) {
        fprintf(stderr, "peer exchange failed\n");
        close(conn);
        close(listen_fd);
        verbs_destroy(&local);
        return 1;
    }
    struct peer_info remote;
    if (verbs_parse(line, &remote) || verbs_to_rts(&local, &remote)) {
        close(conn);
        close(listen_fd);
        verbs_destroy(&local);
        return 1;
    }

    struct ibv_sge sge = {
        .addr = (uintptr_t)local.buffer,
        .length = 256,
        .lkey = local.mr->lkey,
    };
    struct ibv_recv_wr recv_wr = {
        .wr_id = 1,
        .sg_list = &sge,
        .num_sge = 1,
    };
    struct ibv_recv_wr *bad_recv = NULL;
    if (ibv_post_recv(local.qp, &recv_wr, &bad_recv)) {
        perror("ibv_post_recv");
        close(conn);
        close(listen_fd);
        verbs_destroy(&local);
        return 1;
    }
    if (verbs_write_all(conn, "READY\n")) {
        close(conn);
        close(listen_fd);
        verbs_destroy(&local);
        return 1;
    }

    struct ibv_wc wc;
    int ne = verbs_poll(local.cq, &wc);
    int failed = !(ne == 1 && wc.status == IBV_WC_SUCCESS);
    if (failed) {
        fprintf(stderr, "Send/Recv completion failed\n");
    } else {
        printf("Received SEND: '%s'\n", local.buffer);
        if (strcmp(local.buffer, "hello-verbs") != 0) {
            failed = 1;
        }
    }

    if (verbs_read_line(conn, line, sizeof(line)) <= 0 || strcmp(line, "DONE") != 0) {
        failed = 1;
    }
    if (!failed && strcmp(local.buffer + 512, "write-ok") != 0) {
        fprintf(stderr, "RDMA Write payload mismatch: '%s'\n", local.buffer + 512);
        failed = 1;
    } else if (!failed) {
        printf("RDMA Write payload: '%s'\n", local.buffer + 512);
    }
    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");

    close(conn);
    close(listen_fd);
    verbs_destroy(&local);
    return failed ? 1 : 0;
}
