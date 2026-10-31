// examples/02-core/all-collectives.c
// Barrier / Bcast / Scatter / Gather / Allgather / Reduce / Allreduce / Scan / IN_PLACE。
// 编译：mpicc -O2 -o all-collectives all-collectives.c
// 运行：mpirun -np 4 ./all-collectives

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_fail = 0;
    printf("===== Rank %d / %d =====\n", rank, size);

    printf("Rank %d: waiting at barrier\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== After Barrier, everyone continues ===\n\n");
    }

    int bcast_data = 0;
    if (rank == 0) {
        bcast_data = 12345;
        printf("Rank 0: Bcasting data %d\n", bcast_data);
    }
    MPI_Bcast(&bcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: received bcast data %d\n", rank, bcast_data);
    if (bcast_data != 12345) {
        local_fail = 1;
    }

    int *sendbuf = NULL;
    int recvval = -1;
    if (rank == 0) {
        sendbuf = (int *)malloc((size_t)size * sizeof(int));
        for (int i = 0; i < size; i++) {
            sendbuf[i] = (i + 1) * 10;
        }
        printf("Rank 0: Scatter sending:");
        for (int i = 0; i < size; i++) {
            printf(" %d", sendbuf[i]);
        }
        printf("\n");
    }
    MPI_Scatter(sendbuf, 1, MPI_INT, &recvval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: Scatter received %d\n", rank, recvval);
    if (recvval != (rank + 1) * 10) {
        local_fail = 1;
    }
    free(sendbuf);

    int sendval = recvval;
    int *recvbuf = NULL;
    if (rank == 0) {
        recvbuf = (int *)malloc((size_t)size * sizeof(int));
    }
    MPI_Gather(&sendval, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Rank 0: Gather received:");
        for (int i = 0; i < size; i++) {
            printf(" %d", recvbuf[i]);
            if (recvbuf[i] != (i + 1) * 10) {
                local_fail = 1;
            }
        }
        printf("\n");
        free(recvbuf);
    }

    int allgather_send = rank + 1;
    int *allgather_recv = (int *)malloc((size_t)size * sizeof(int));
    MPI_Allgather(&allgather_send, 1, MPI_INT, allgather_recv, 1, MPI_INT, MPI_COMM_WORLD);
    printf("Rank %d: Allgather received:", rank);
    for (int i = 0; i < size; i++) {
        printf(" %d", allgather_recv[i]);
        if (allgather_recv[i] != i + 1) {
            local_fail = 1;
        }
    }
    printf("\n");
    free(allgather_recv);

    double local = (double)(rank + 1);
    double global_sum = 0.0;
    double expected_sum = (double)size * (size + 1) / 2.0;
    MPI_Reduce(&local, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Rank 0: Reduce sum result: %.1f, expected: %.1f\n", global_sum, expected_sum);
        if (global_sum != expected_sum) {
            local_fail = 1;
        }
    }

    double allreduce_result = 0.0;
    MPI_Allreduce(&local, &allreduce_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: Allreduce sum result = %.1f, expected %.1f\n",
           rank, allreduce_result, expected_sum);
    if (allreduce_result != expected_sum) {
        local_fail = 1;
    }

    // 发送缓冲和接收缓冲是同一块内存时，必须使用 MPI_IN_PLACE，不能把两个参数写成同一个指针。
    double inplace = local;
    MPI_Allreduce(MPI_IN_PLACE, &inplace, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (inplace != expected_sum) {
        local_fail = 1;
    }

    // MPI-3 非阻塞集合通信：调用立刻返回，必须 Wait 之后才能读结果。
    double iall_in = local;
    double iall_out = 0.0;
    MPI_Request iall_req;
    MPI_Iallreduce(&iall_in, &iall_out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &iall_req);
    MPI_Wait(&iall_req, MPI_STATUS_IGNORE);
    if (iall_out != expected_sum) {
        local_fail = 1;
    }

    double scan_result = 0.0;
    MPI_Scan(&local, &scan_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double expected_scan = (double)(rank + 1) * (rank + 2) / 2.0;
    printf("Rank %d: Scan prefix sum = %.1f, expected %.1f\n", rank, scan_result, expected_scan);
    if (scan_result != expected_scan) {
        local_fail = 1;
    }

    int failed = 0;
    MPI_Reduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return (rank == 0 && failed) ? 1 : 0;
}
