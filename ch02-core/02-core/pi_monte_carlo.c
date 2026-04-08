// examples/02-core/pi_monte_carlo.c
// 使用 Monte Carlo 方法计算 π，演示 MPI_Reduce 实际应用
// 原理：单位正方形内投点，统计落在单位圆内的数目，π ≈ 4 * inside / total
// 每个进程并行投自己那部分点，最后 Reduce 求和得到总数
// 编译：mpicc -O2 -o pi_monte_carlo pi_monte_carlo.c
// 运行：mpirun -np 4 ./pi_monte_carlo [points_per_proc]

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // 默认每个进程投 1e6 个点，可以从命令行改
    long long points_per_proc = 1000000;
    if (argc > 1) {
        points_per_proc = atoll(argv[1]);
    }

    // --------------------------
    // 每个进程用不同的随机种子，避免所有进程投一样的点
    srand(time(NULL) + rank);

    // 每个进程统计自己这块有多少点落在圆内
    long long local_inside = 0;
    for (long long i = 0; i < points_per_proc; i++) {
        // 生成 [0, 1) 之间均匀随机数
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        // 判断 x^2 + y^2 <= 1 就是落在单位圆内
        if (x*x + y*y <= 1.0) {
            local_inside++;
        }
    }

    // --------------------------
    // 所有进程把 local_inside 求和，结果放到 rank 0
    // MPI_Reduce 做归约求和，这就是集合通信的便利
    // 不用自己写点对点发送收集，MPI帮你做好还更快
    long long global_inside;
    MPI_Reduce(&local_inside, &global_inside, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // --------------------------
    // rank 0 计算最终 π 并输出结果
    if (rank == 0) {
        long long total_points = points_per_proc * size;
        double pi = 4.0 * (double)global_inside / (double)total_points;
        printf("Total points: %lld\n", total_points);
        printf("Estimated pi = %.10f\n", pi);
        printf("Real    pi = 3.1415926535...\n");
        printf("Error   = %.10f\n", pi - 3.1415926535);
    }

    MPI_Finalize();
    return 0;
}
