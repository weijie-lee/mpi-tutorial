// examples/02-core/nonblocking.c
// 演示非阻塞通信：Isend/Irecv + Wait
// 好处：可以让计算和通信重叠，掩盖通信延迟
// 编译：mpicc -O2 -o nonblocking nonblocking.c
// 运行：mpirun -np 2 ./nonblocking

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    int send_data = rank;
    int recv_data;
    MPI_Request request_send, request_recv;
    // MPI_Request 用来跟踪非阻塞操作的状态

    // --------------------------
    // 非阻塞发送和接收：调用立刻返回，不等待完成
    // 接口参数和阻塞版 MPI_Send/MPI_Recv 一样，只是多了个 request 输出
    if (rank == 0) {
        // 启动非阻塞发送
        MPI_Isend(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request_send);
        // 启动非阻塞接收
        MPI_Irecv(&recv_data, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &request_recv);
    } else {
        MPI_Isend(&send_data, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request_send);
        MPI_Irecv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request_recv);
    }

    // --------------------------
    // 现在 send/recv 已经启动，在后台进行
    // 我们可以在这里做不依赖通信结果的计算
    // 这样计算和通信并行执行，掩盖了通信延迟，总时间更快
    // 这里我们用循环模拟计算
    long long sum = 0;
    for (long long i = 0; i < 1000000; i++) {
        sum += i;
    }

    // --------------------------
    // 计算做完了，现在需要通信结果，等待通信完成
    // MPI_Wait 会阻塞直到对应的非阻塞操作完成
    if (rank == 0) {
        MPI_Wait(&request_send, MPI_STATUS_IGNORE);
        MPI_Wait(&request_recv, MPI_STATUS_IGNORE);
        printf("Rank 0: sent %d, received %d from rank 1\n", send_data, recv_data);
    } else {
        MPI_Wait(&request_send, MPI_STATUS_IGNORE);
        MPI_Wait(&request_recv, MPI_STATUS_IGNORE);
        printf("Rank 1: sent %d, received %d from rank 0\n", send_data, recv_data);
    }

    // 如果有多个请求，可以用 MPI_Waitall 一次等待所有完成

    MPI_Finalize();
    return 0;
}
