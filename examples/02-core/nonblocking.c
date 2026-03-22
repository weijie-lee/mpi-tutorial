#include <mpi.h>
#include <stdio.h>

/*
 * 非阻塞通信示例：rank 0 和 rank 1 互相发送数据，用非阻塞避免死锁
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    int send_data = rank;
    int recv_data = 0;
    MPI_Request requests[2];
    MPI_Status statuses[2];

    int other = 1 - rank; // rank 0 -> 1, rank 1 -> 0

    // 非阻塞启动发送和接收，立刻返回
    MPI_Isend(&send_data, 1, MPI_INT, other, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&recv_data, 1, MPI_INT, other, 0, MPI_COMM_WORLD, &requests[1]);

    // 这里可以做其他不依赖发送/接收的计算，让计算和通信重叠

    // 等待两个操作都完成
    MPI_Waitall(2, requests, statuses);

    printf("Rank %d: received %d from rank %d\n", rank, recv_data, other);

    MPI_Finalize();
    return 0;
}
