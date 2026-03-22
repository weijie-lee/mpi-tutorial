#include <mpi.h>
#include <stdio.h>

/*
 * 这个示例展示了经典的死锁情况：两个进程都先发送再接收，互相等待
 * 编译运行：mpicc -o deadlock deadlock.c && mpirun -np 2 ./deadlock
 * 在某些实现上会卡住
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data = rank;
    MPI_Status status;

    // 错误写法：都先给对方发，再收，容易死锁
    if (rank == 0) {
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
        printf("Rank 0: done\n");
    } else if (rank == 1) {
        MPI_Send(&data, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: done\n");
    }

    MPI_Finalize();
    return 0;
}
