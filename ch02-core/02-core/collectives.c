// examples/02-core/collectives.c
// 常用集合通信：Bcast / Reduce / Allreduce / Scatter / Gather。
// 通信域里的每个进程都必须调用同一次集合操作，count 和 datatype 也必须一致。
// 编译：mpicc -O2 -o collectives collectives.c
// 运行：mpirun -np 4 ./collectives

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_fail = 0;

    int data = 0;
    if (rank == 0) {
        data = 12345;
        printf("Root (rank 0) broadcasting data: %d\n", data);
    }
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: received broadcast data: %d\n", rank, data);
    if (data != 12345) {
        local_fail = 1;
    }

    double local = (double)rank * (double)rank;
    double global_sum = 0.0;
    MPI_Reduce(&local, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double expected = (size - 1.0) * size * (2.0 * size - 1.0) / 6.0;
        printf("\nSum of rank^2 from all ranks: %f (expected %f)\n", global_sum, expected);
        if (global_sum != expected) {
            local_fail = 1;
        }
    }

    double global_sum_all = 0.0;
    MPI_Allreduce(&local, &global_sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: global sum via Allreduce: %f\n", rank, global_sum_all);
    if (rank == 0) {
        double expected = (size - 1.0) * size * (2.0 * size - 1.0) / 6.0;
        if (global_sum_all != expected) {
            local_fail = 1;
        }
    }

    // 缓冲按进程数分配。写成固定的 int sendbuf[4] 时，np != 4 会越界。
    int *sendbuf = NULL;
    if (rank == 0) {
        sendbuf = (int *)malloc((size_t)size * sizeof(int));
        if (!sendbuf) {
            local_fail = 1;
        } else {
            printf("\nScattering from root: ");
            for (int i = 0; i < size; i++) {
                sendbuf[i] = i * 10;
                printf("%d ", sendbuf[i]);
            }
            printf("\n");
        }
    }
    int recvval = -1;
    MPI_Scatter(sendbuf, 1, MPI_INT, &recvval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: scattered value: %d\n", rank, recvval);
    if (recvval != rank * 10) {
        local_fail = 1;
    }
    free(sendbuf);

    recvval *= 2;
    int *recvbuf = NULL;
    if (rank == 0) {
        recvbuf = (int *)malloc((size_t)size * sizeof(int));
        if (!recvbuf) {
            local_fail = 1;
        }
    }
    MPI_Gather(&recvval, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0 && recvbuf) {
        printf("After gather (doubled): ");
        for (int i = 0; i < size; i++) {
            printf("%d ", recvbuf[i]);
            if (recvbuf[i] != i * 20) {
                local_fail = 1;
            }
        }
        printf("\n");
    }
    free(recvbuf);

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
