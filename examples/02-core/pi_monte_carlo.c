// examples/02-core/pi_monte_carlo.c
// 使用 Monte Carlo 方法计算 π，演示集合通信 MPI_Reduce 的实际应用
// 原理：在单位正方形内随机投点，统计落在单位圆内的数目
// π ≈ 4 * (inside / total)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 默认每个进程投 1000000 个点，可以从命令行改
    long long points_per_proc = 1000000;
    if (argc > 1) {
        points_per_proc = atoll(argv[1]);
    }

    // 每个进程用不同的随机种子
    srand(time(NULL) + rank);

    long long local_inside = 0;
    // 投点
    for (long long i = 0; i < points_per_proc; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        // 判断是否在单位圆内
        if (x*x + y*y <= 1.0) {
            local_inside++;
        }
    }

    // 所有进程的结果归约求和到 rank 0
    long long global_inside;
    MPI_Reduce(&local_inside, &global_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // rank 0 计算最终 π
    if (rank == 0) {
        long long total_points = points_per_proc * size;
        double pi = 4.0 * (double)global_inside / (double)total_points;
        printf("Total points: %lld\n", total_points);
        printf("Estimated pi = %.10f\n", pi);
        printf("Real    pi = 3.1415926535...\n");
        printf("Error = %.10f\n", pi - 3.1415926535);
    }

    MPI_Finalize();
    return 0;
}
