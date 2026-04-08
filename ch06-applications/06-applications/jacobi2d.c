/*
 * examples/06-applications/jacobi2d.c
 * 二维Jacobi迭代求解泊松方程，MPI并行完整应用示例
 * 方法：区域分解，每个进程负责一块子网格，每次迭代交换边界（ghost exchange）
 * 编译：mpicc -O2 -o jacobi2d jacobi2d.c -lm
 * 运行：mpirun -np 4 ./jacobi2d
 * 说明：这是一个经典的 MPI 并行应用，演示了区域分解 + 边界交换 + 集合通信
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 整体网格大小 N x N
#define N 100   
// 最多迭代次数
#define MAX_ITER 1000
// 收敛阈值
#define TOL 1e-6

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // 区域分解：一维划分，每个进程分 nlocal 行
    // 实际应用可以二维划分，但一维划分简单演示
    int nlocal = N / size;
    int nx = N;                   // x方向还是整体大小
    int ny = nlocal + 2;         // y方向每个进程多分配两行，放ghost边界（上下各一层）

    // --------------------------
    // 分配二维数组
    // u: 当前迭代，u_new: 下一次迭代
    double **u = (double**)malloc(ny * sizeof(double*));
    double **u_new = (double**)malloc(ny * sizeof(double*));
    for (int i = 0; i < ny; i++) {
        u[i] = (double*)malloc(nx * sizeof(double));
        u_new[i] = (double*)malloc(nx * sizeof(double));
        // 初始化为 0
        for (int j = 0; j < nx; j++) {
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
        }
    }

    // --------------------------
    // 初始化边界条件
    // 这里简单设置：右边界 Dirichlet 条件为 1，其他边界 0
    for (int i = 0; i < ny; i++) {
        u[i][nx-1] = 1.0;
        u_new[i][nx-1] = 1.0;
    }

    // --------------------------
    // 邻居 rank
    int up_rank = rank - 1;   // 上面进程
    int down_rank = rank + 1; // 下面进程
    MPI_Status status;

    // --------------------------
    // 迭代求解
    double start = MPI_Wtime();
    double global_diff;
    int iter;

    for (iter = 0; iter < MAX_ITER; iter++) {
        // --------------------------
        // Step 1: 交换 ghost 边界
        // 每个进程把自己最上面一行发给上面进程，放到上面进程的ghost下边界
        // 每个进程把自己最下面一行发给下面进程，放到下面进程的ghost上边界
        if (up_rank >= 0) {
            // 我的第一行（真实数据第一行）发给邻居up
            MPI_Send(u[1], nx, MPI_DOUBLE, up_rank, 0, MPI_COMM_WORLD);
            // 从up邻居接收它的最下面一行，放到我的第0行ghost
            MPI_Recv(u[0], nx, MPI_DOUBLE, up_rank, 0, MPI_COMM_WORLD, &status);
        }
        if (down_rank < size) {
            // 我的倒数第二行（真实数据最后一行）发给邻居down
            MPI_Send(u[ny-2], nx, MPI_DOUBLE, down_rank, 0, MPI_COMM_WORLD);
            // 从down邻居接收它的第一行，放到我的最后一行ghost
            MPI_Recv(u[ny-1], nx, MPI_DOUBLE, down_rank, 0, MPI_COMM_WORLD, &status);
        }

        // --------------------------
        // Step 2: Jacobi 迭代
        // 对每个内部格点做 Jacobi 更新：u_new = 平均四邻居
        double local_diff = 0.0;
        for (int i = 1; i <= ny-2; i++) {
            for (int j = 1; j < nx-1; j++) {
                // Jacobi 公式：u_new[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) / 4
                u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
                // 累加差分，判断收敛
                local_diff += (u_new[i][j] - u[i][j]) * (u_new[i][j] - u[i][j]);
            }
        }

        // --------------------------
        // Step 3: 交换 u 和 u_new 指针，不用拷贝
        double **tmp = u;
        u = u_new;
        u_new = tmp;

        // --------------------------
        // Step 4: 全局归约，所有进程差分求和，rank 0 判断收敛
        // MPI_Allreduce 所有进程都得到 global_diff，方便都能判断收敛
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_diff = sqrt(global_diff);

        // 每 100 次迭代打印一次
        if (rank == 0 && (iter % 100 == 0)) {
            printf("Iter %d: diff = %.8e\n", iter, global_diff);
        }

        // 判断收敛
        if (global_diff < TOL) {
            if (rank == 0) {
                printf("✅ Converged after %d iterations, diff = %.8e < %.8e\n", iter+1, global_diff, TOL);
            }
            break;
        }
    }

    // 计时结束
    double end = MPI_Wtime();
    if (rank == 0) {
        printf("Total elapsed time: %.4f seconds\n", end - start);
        printf("Grid size: %d x %d, %d processes\n", N, N, size);
    }

    // --------------------------
    // 清理内存
    for (int i = 0; i < ny; i++) {
        free(u[i]);
        free(u_new[i]);
    }
    free(u);
    free(u_new);

    MPI_Finalize();
    return 0;
}

/*
 * 这个示例演示了：
 * 1. 区域分解：把大问题分成小问题分给多个进程
 * 2. Ghost 交换：每个进程需要邻居的边界数据才能计算，迭代前交换边界
 * 3. 集合通信：Allreduce 收集差分，全局判断收敛
 * 4. 基本点对点通信：发送接收边界
 *
 * 练习：
 * 1. 试试不同进程数，看时间怎么变化（强扩展性）
 * 2. 试试改成二维区域分解
 * 3. 用非阻塞通信交换边界，让计算和通信重叠，看看能不能更快
 */
