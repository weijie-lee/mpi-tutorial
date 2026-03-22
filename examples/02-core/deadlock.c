// examples/02-core/deadlock.c
// 演示点对点通信中常见的死锁情况
// 两个进程都要给对方发消息，都先 send 再 recv，可能会死锁
// 编译：mpicc -O2 -o deadlock deadlock.c
// 运行：mpirun -np 2 ./deadlock
// 预期：程序卡住，就是死锁了

#include <mpi.h>

int main(int argc, char** argv) {
    int rank;
    // 初始化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data = rank;  // 每个进程要发的数据就是自己的 rank

    // --------------------------
    // 错误写法：两个都先发再收，容易死锁
    // 如果 MPI_Send 是同步模式，或者缓冲区不够，send 会卡住等 recv
    // 但对方也在等 send 完成，所以互相等，死锁
    if (rank == 0) {
        // 发给 rank 1
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        // 等 rank 1 发给自己
        MPI_Recv(&data, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        // 发给 rank 0
        MPI_Send(&data, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        // 等 rank 0 发给自己
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // --------------------------
    // 正确避免死锁的方法：
    // 方法一：rank 小先发，rank 大先收
    // if (rank == 0) {
    //     MPI_Send(...); MPI_Recv(...);
    // } else {
    //     MPI_Recv(...); MPI_Send(...);
    // }
    //
    // 方法二：用非阻塞通信 Isend/Irecv，立刻返回，然后 Wait
    // 方法三：用 MPI_Sendrecv，原子操作，MPI 保证不死锁

    if (rank == 0) {
        printf("Program completed without deadlock (this is unexpected!)\n");
    }

    MPI_Finalize();
    return 0;
}
