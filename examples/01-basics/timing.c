// examples/01-basics/timing.c
// 演示 MPI_Wtime() 计时器的使用
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI 提供了高精度计时器，返回秒数
    double start = MPI_Wtime();
    
    // 模拟做一些计算工作，这里睡 1 秒
    sleep(1);
    
    double end = MPI_Wtime();
    
    // 只有 root 进程打印结果
    if (rank == 0) {
        printf("Total elapsed time: %.3f seconds\n", end - start);
    }

    MPI_Finalize();
    return 0;
}
