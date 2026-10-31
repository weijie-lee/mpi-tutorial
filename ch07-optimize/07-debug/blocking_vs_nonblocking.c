// examples/07-debug/blocking_vs_nonblocking.c
// 同样的计算量：阻塞通信完成后再计算，对比非阻塞通信与计算重叠。
// 共享内存上的差异可能很小；这个程序保证两边收到的数据一致，并打印时间。
// 编译：mpicc -O2 -o blocking_vs_nonblocking blocking_vs_nonblocking.c -lm
// 运行：mpirun -np 2 ./blocking_vs_nonblocking

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N (256 * 1024)
#define REPEATS 20

static double spin_compute(void) {
    double acc = 0.0;
    for (int i = 1; i <= 20000; i++) {
        acc += sqrt((double)i);
    }
    return acc;
}

static int buffer_ok(const int *buf, int n, int marker) {
    if (buf[0] != marker || buf[n - 1] != marker + n - 1) {
        return 0;
    }
    return 1;
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

    int *sendbuf = (int *)malloc((size_t)N * sizeof(int));
    int *recvbuf = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) {
        sendbuf[i] = rank * N + i;
    }

    double sink = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int rep = 0; rep < REPEATS; rep++) {
        if (rank == 0) {
            MPI_Send(sendbuf, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, N, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recvbuf, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendbuf, N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        sink += spin_compute();
    }
    double blocking_time = MPI_Wtime() - t0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    for (int rep = 0; rep < REPEATS; rep++) {
        MPI_Request reqs[2];
        int peer = 1 - rank;
        MPI_Irecv(recvbuf, N, MPI_INT, peer, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sendbuf, N, MPI_INT, peer, 1, MPI_COMM_WORLD, &reqs[1]);
        sink += spin_compute();
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }
    double nonblocking_time = MPI_Wtime() - t1;

    int local_fail = buffer_ok(recvbuf, N, (1 - rank) * N) ? 0 : 1;
    if (sink == 0.0) {
        local_fail = 1;
    }

    if (rank == 0) {
        printf("=== Blocking then compute ===\n");
        printf("Time: %.6f seconds (%d repeats, %d ints)\n", blocking_time, REPEATS, N);
        printf("=== Nonblocking overlapped with compute ===\n");
        printf("Time: %.6f seconds\n", nonblocking_time);
        printf("Overlap is visible only when the compute is long enough to hide the transfer.\n");
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return failed ? 1 : 0;
}
