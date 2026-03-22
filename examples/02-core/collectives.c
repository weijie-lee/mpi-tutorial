#include <mpi.h>
#include <stdio.h>

/*
 * 集合通信示例：Bcast + Reduce + Allreduce
 * 运行：mpirun -np 4 ./collectives
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Bcast 示例：root 0 广播数据给所有进程
    int data = 0;
    if (rank == 0) {
        data = 12345;
        printf("Root (rank 0) broadcasting data: %d\n", data);
    }

    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: received broadcast data: %d\n", rank, data);

    // 2. Reduce 示例：所有进程计算 rank 的平方和，结果汇总到 root
    double local = rank * rank;
    double global_sum;

    MPI_Reduce(&local, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum of squares from all ranks: %f\n", global_sum);
    }

    // 3. Allreduce 示例：所有进程都拿到全局总和
    MPI_Allreduce(&local, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: global sum via Allreduce: %f\n", rank, global_sum);

    // 4. Scatter + Gather 示例
    int sendbuf[4];
    int recvval;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            sendbuf[i] = i * 10;
        }
        printf("Scattering values: ");
        for (int i = 0; i < size; i++) printf("%d ", sendbuf[i]);
        printf("\n");
    }

    MPI_Scatter(sendbuf, 1, MPI_INT, &recvval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: scattered value: %d\n", rank, recvval);

    // 每个进程做个计算，recvval *= 2，然后 gather 回去
    recvval *= 2;
    MPI_Gather(&recvval, 1, MPI_INT, sendbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("After gather (doubled): ");
        for (int i = 0; i < size; i++) printf("%d ", sendbuf[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
