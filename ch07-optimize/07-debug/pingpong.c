// examples/07-debug/pingpong.c
// 点对点往返：单向延迟 ≈ RTT/2，单向带宽 ≈ 消息字节数 / 单向时间。
// 模型是 T(n) = α + nβ。小消息看 α，大消息看 β。
// 编译：mpicc -O2 -o pingpong pingpong.c
// 运行：mpirun -np 2 ./pingpong

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_size(int rank, int bytes, int iters) {
    char *sendbuf = (char *)malloc((size_t)bytes);
    char *recvbuf = (char *)malloc((size_t)bytes);
    if (!sendbuf || !recvbuf) {
        free(sendbuf);
        free(recvbuf);
        return 1;
    }
    memset(sendbuf, rank + 1, (size_t)bytes);
    int count = bytes;

    for (int i = 0; i < 5; i++) {
        if (rank == 0) {
            MPI_Send(sendbuf, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recvbuf, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendbuf, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int i = 0; i < iters; i++) {
        if (rank == 0) {
            MPI_Send(sendbuf, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recvbuf, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendbuf, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    double elapsed = MPI_Wtime() - start;

    int bad = 0;
    unsigned char expect = (unsigned char)((1 - rank) + 1);
    if ((unsigned char)recvbuf[0] != expect || (unsigned char)recvbuf[bytes - 1] != expect) {
        bad = 1;
    }

    if (rank == 0) {
        double rtt = elapsed / iters;
        double one_way = rtt / 2.0;
        double mbps = (bytes / one_way) / 1e6;
        printf("bytes=%8d  iters=%4d  one-way=%.3f us  bandwidth=%.2f MB/s  check=%s\n",
               bytes, iters, one_way * 1e6, mbps, bad ? "FAIL" : "ok");
    }

    free(sendbuf);
    free(recvbuf);
    int bad_all = 0;
    MPI_Allreduce(&bad, &bad_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return bad_all ? 1 : 0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Run with exactly 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("=== Ping-pong (alpha + n*beta) ===\n");
    }
    int failed = 0;
    failed |= run_size(rank, 8, 200);
    failed |= run_size(rank, 4096, 100);
    failed |= run_size(rank, 256 * 1024, 40);
    failed |= run_size(rank, 2 * 1024 * 1024, 10);

    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
