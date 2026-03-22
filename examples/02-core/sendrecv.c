// examples/02-core/sendrecv.c
// 最简单的点对点通信示例：rank 0 发送一个整数给 rank 1
// 编译：mpicc -O2 -o sendrecv sendrecv.c
// 运行：mpirun -np 2 ./sendrecv

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    // 初始化 MPI，获取 rank 和 size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // 至少需要两个进程才能通信
    if (size < 2) {
        fprintf(stderr, "This example needs at least 2 processes, you have %d\n", size);
        // 不够直接退出
        MPI_Finalize();
        return 1;
    }

    // --------------------------
    // rank 0 负责发送数据
    if (rank == 0) {
        // 要发送的数据
        int data = 42;
        printf("Rank 0: sending int data = %d to rank 1\n", data);
        // --------------------------
        // MPI_Send：阻塞发送
        // 参数说明：
        // &data：发送缓冲区地址，就是我们要发送的数据
        // 1：发送 1 个元素（注意：是元素个数，不是字节数！）
        // MPI_INT：每个元素的类型是 int
        // 1：目标进程的 rank 是 1
        // 0：消息标签 tag=0，用来区分不同消息
        // MPI_COMM_WORLD：通信域，在默认通信域里通信
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } 
    // --------------------------
    // rank 1 负责接收数据
    else if (rank == 1) {
        int received_data;  // 接收缓冲区
        MPI_Status status;  // 存储接收完成后的状态信息
        // --------------------------
        // MPI_Recv：阻塞接收
        // 参数说明：
        // &received_data：接收缓冲区地址
        // 1：最多接收 1 个元素
        // MPI_INT：接收的数据类型要和发送方一致！
        // 0：只接收 tag=0 的消息
        // MPI_COMM_WORLD：同一个通信域
        // &status：输出状态，包含发送方 rank、tag、错误信息
        MPI_Recv(&received_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: received data = %d from rank 0\n", received_data);

        // --------------------------
        // 如果不知道发送了多少元素，可以用 MPI_Get_count 获取实际接收长度
        int count;
        MPI_Get_count(&status, MPI_INT, &count);
        printf("Rank 1: actually received %d elements\n", count);
    }

    // --------------------------
    // 所有通信完成，结束 MPI
    MPI_Finalize();
    return 0;
}
