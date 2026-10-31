// examples/07-debug/allreduce_bench.c
// 只计时 MPI_Allreduce 本身，并核对求和结果。
// 用 int 累加 N=1e6 个 rank+i 会溢出，全局和是错的；这里改用 double。
// 编译：mpicc -O2 -o allreduce_bench allreduce_bench.c
// 运行：mpirun -np 4 ./allreduce_bench

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int bench_count(int rank, int size, int count, int iters) {
    double *sendbuf = (double *)malloc((size_t)count * sizeof(double));
    double *recvbuf = (double *)malloc((size_t)count * sizeof(double));
    if (!sendbuf || !recvbuf) {
        free(sendbuf);
        free(recvbuf);
        return 1;
    }
    for (int i = 0; i < count; i++) {
        sendbuf[i] = (double)(rank + 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 5; i++) {
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    for (int i = 0; i < iters; i++) {
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;

    double expected = (double)size * (size + 1) / 2.0;
    int bad = 0;
    if (recvbuf[0] != expected || recvbuf[count - 1] != expected) {
        bad = 1;
    }

    if (rank == 0) {
        double bytes = (double)count * sizeof(double);
        double avg = elapsed / iters;
        // 算法带宽 = 消息字节 / 单次耗时。总线带宽再乘 2*(p-1)/p。
        double alg_MBps = (bytes / avg) / 1e6;
        double bus = (size == 1) ? alg_MBps : alg_MBps * (2.0 * (size - 1) / size);
        printf("count=%d doubles  iters=%d  avg=%.6e s  alg=%.2f MB/s  bus=%.2f MB/s  check=%s\n",
               count, iters, avg, alg_MBps, bus, bad ? "FAIL" : "ok");
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

    if (rank == 0) {
        printf("=== Allreduce Benchmark (%d processes) ===\n", size);
    }

    int failed = 0;
    failed |= bench_count(rank, size, 1, 100);
    failed |= bench_count(rank, size, 1024, 50);
    failed |= bench_count(rank, size, 65536, 20);

    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
