// examples/01-basics/timing.c
// 演示 MPI_Wtime() 计时器的使用，测量程序运行时间
// 编译：mpicc -O2 -o timing timing.c
// 运行：mpirun -np 4 ./timing

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
    // 初始化 MPI 环境，所有 MPI 程序都必须从这开始
    MPI_Init(&argc, &argv);
    
    int rank, size;
    // 获取当前进程编号
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 获取总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // MPI 提供了高精度计时器：MPI_Wtime()
    // 返回值是 double，表示从某个固定时间点到现在的秒数
    // 精度一般可以到微秒级，足够测量并行程序运行时间
    double start = MPI_Wtime();
    
    // --------------------------
    // 这里模拟做一些计算工作，我们用 sleep 模拟耗时 1 秒
    // 实际应用中这里就是你的计算/通信逻辑
    sleep(1);
    
    // 计时结束，获取结束时间
    double end = MPI_Wtime();
    
    // --------------------------
    // 只有 rank 0 打印结果，避免每个进程都输出混乱
    // 注意：MPI_Wtime() 是每个进程本地的时钟
    // 如果要测量整个程序从开始到结束的时间，一般让 rank 0 测量就行
    if (rank == 0) {
        printf("Total elapsed time: %.3f seconds\n", end - start);
        printf("This should be roughly 1 second (sleep(1))\n");
    }

    // 结束 MPI 环境
    MPI_Finalize();
    return 0;
}
