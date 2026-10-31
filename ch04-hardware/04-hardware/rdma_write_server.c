/*
 * RDMA Write 服务端。
 * TCP 只用来交换队列对和内存注册信息，以及在写完成后做一次同步。
 * 载荷本身由客户端 RDMA Write 直接放进本进程注册过的缓冲区。
 *
 * 编译：gcc -O2 -Wall -Wextra -o rdma_write_server rdma_write_server.c -libverbs
 * 运行：./rdma_write_server [port]
 */

#include "rdma_common.h"

int main(int argc, char **argv) {
    int port = (argc >= 2) ? atoi(argv[1]) : 18515;
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
        rdma_close(&local);
        return 1;
    }
    printf("LISTENING %d\n", port);
    fflush(stdout);

    int conn = accept(listen_fd, NULL, NULL);
    if (conn < 0) {
        perror("accept");
        close(listen_fd);
        rdma_close(&local);
        return 1;
    }

    char line[256];
    rdma_format_peer(&local, 0, line, sizeof(line));
    if (rdma_write_all(conn, line)) {
        fprintf(stderr, "failed to send local peer info\n");
        close(conn);
        close(listen_fd);
        rdma_close(&local);
        return 1;
    }
    if (rdma_read_line(conn, line, sizeof(line)) <= 0) {
        fprintf(stderr, "failed to read remote peer info\n");
        close(conn);
        close(listen_fd);
        rdma_close(&local);
        return 1;
    }
    struct rdma_peer remote;
    if (rdma_parse_peer(line, &remote) || rdma_qp_to_rts(&local, &remote)) {
        fprintf(stderr, "could not bring QP to RTS\n");
        close(conn);
        close(listen_fd);
        rdma_close(&local);
        return 1;
    }
    if (rdma_write_all(conn, "RTS\n")) {
        fprintf(stderr, "failed to notify client\n");
        close(conn);
        close(listen_fd);
        rdma_close(&local);
        return 1;
    }

    if (rdma_read_line(conn, line, sizeof(line)) <= 0) {
        fprintf(stderr, "timed out waiting for RDMA completion notice\n");
        close(conn);
        close(listen_fd);
        rdma_close(&local);
        return 1;
    }

    const char *expect = "hello-rdma-write";
    int failed = strcmp(local.buf, expect) != 0;
    printf("Buffer after RDMA Write: '%s'\n", local.buf);
    printf("RESULT: %s\n", failed ? "FAIL" : "PASS");

    close(conn);
    close(listen_fd);
    rdma_close(&local);
    return failed ? 1 : 0;
}
