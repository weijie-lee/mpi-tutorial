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
    // MPI_Wtime() 是每个进程自己的时钟，分辨率由 MPI_Wtick() 给出（秒）。
    // 比较不同进程的耗时时，应先归约，而不是只看某一行输出。
    double elapsed = end - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int failed = 0;
    if (rank == 0) {
        double tick = MPI_Wtick();
        printf("MPI_Wtick resolution: %.3e seconds\n", tick);
        printf("Max elapsed time across ranks: %.3f seconds\n", max_elapsed);
        printf("This should be roughly 1 second (sleep(1))\n");
        if (!(tick > 0.0 && tick < 0.1)) {
            fprintf(stderr, "unexpected MPI_Wtick %g\n", tick);
            failed = 1;
        }
        if (!(max_elapsed >= 0.5 && max_elapsed < 5.0)) {
            fprintf(stderr, "elapsed time %.3f is outside [0.5, 5.0]\n", max_elapsed);
            failed = 1;
        }
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }

    // 结束 MPI 环境
    MPI_Finalize();
    return failed ? 1 : 0;
}
