#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 初始化 MPI
    MPI_Init(&argc, &argv);

    // 获取当前进程的 rank（编号），从 0 开始计数
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 获取通信域中的总进程数
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello MPI! I'm rank %d out of %d processes\n", rank, size);

    // 结束 MPI 环境
    MPI_Finalize();
    return 0;
}
