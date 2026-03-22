/*
 * 二维Jacobi迭代求解泊松方程，MPI并行版本
 * 方法：区域分解，每个进程负责一块子网格，迭代中交换边界
 * 编译：mpicc -O2 -o jacobi2d jacobi2d.c
 * 运行：mpirun -np 4 ./jacobi2d
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100   /* 整体网格大小 */
#define MAX_ITER 1000
#define TOL 1e-6

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 简单一维划分，每个进程分nlocal行
    int nlocal = N / size;
    int nx = N;
    int ny = nlocal + 2; /* 上下各加一层 ghost 边界 */

    double **u = malloc(ny * sizeof(double*));
    double **u_new = malloc(ny * sizeof(double*));
    for (int i = 0; i < ny; i++) {
        u[i] = malloc(nx * sizeof(double));
        u_new[i] = malloc(nx * sizeof(double));
        for (int j = 0; j < nx; j++) {
            u[i][j] = 0.0;
            u_new[i][j] = 0.0;
        }
    }

    // 初始化边界：这里简单设右边界为1
    for (int i = 0; i < ny; i++) {
        u[i][nx-1] = 1.0;
        u_new[i][nx-1] = 1.0;
    }

    int up = rank - 1;
    int down = rank + 1;
    MPI_Status status;

    double start = MPI_Wtime();
    double diff;
    int iter;

    for (iter = 0; iter < MAX_ITER; iter++) {
        // 交换ghost边界
        // 发送给上，接收从上
        if (up >= 0) {
            MPI_Send(u[1], nx, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
            MPI_Recv(u[0], nx, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &status);
        }
        // 发送给下，接收从下
        if (down < size) {
            MPI_Send(u[ny-2], nx, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
            MPI_Recv(u[ny-1], nx, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &status);
        }

        // Jacobi迭代
        diff = 0.0;
        for (int i = 1; i <= ny-2; i++) {
            for (int j = 1; j < nx-1; j++) {
                u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
                diff += (u_new[i][j] - u[i][j]) * (u_new[i][j] - u[i][j]);
            }
        }

        // 交换u和u_new
        double **tmp = u;
        u = u_new;
        u_new = tmp;

        // 全局归约求和diff
        double global_diff;
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_diff = sqrt(global_diff);

        if (rank == 0 && (iter % 100 == 0)) {
            printf("Iter %d: diff = %e\n", iter, global_diff);
        }

        if (global_diff < TOL) {
            if (rank == 0) {
                printf("Converged after %d iterations, diff = %e < %e\n", iter+1, global_diff, TOL);
            }
            break;
        }
    }

    double end = MPI_Wtime();
    if (rank == 0) {
        printf("Total time: %.4f seconds\n", end - start);
    }

    // 清理
    for (int i = 0; i < ny; i++) {
        free(u[i]);
        free(u_new[i]);
    }
    free(u);
    free(u_new);

    MPI_Finalize();
    return 0;
}
