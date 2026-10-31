// examples/06-applications/jacobi2d.c
// 二维 Jacobi 迭代。区域按行切开，每轮用 MPI_Sendrecv 交换一行 ghost。
// 右边界为 1，其余边界为 0，源项 f = 0，因此离散方程是拉普拉斯方程。
//
// 两边都先 MPI_Send 再 MPI_Recv 时，消息一旦超过 eager 缓冲就会死锁。
// Sendrecv 把一对发送和接收绑在一起，边界交换不会自己卡住。
//
// 编译：mpicc -O2 -o jacobi2d jacobi2d.c -lm
// 运行：mpirun -np 4 ./jacobi2d 32 2000

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void serial_jacobi(int n, int iters, double *out) {
    int nx = n;
    int ny = n + 2;
    double *a = (double *)calloc((size_t)ny * nx, sizeof(double));
    double *b = (double *)calloc((size_t)ny * nx, sizeof(double));
    for (int i = 0; i < ny; i++) {
        a[(size_t)i * nx + (nx - 1)] = 1.0;
        b[(size_t)i * nx + (nx - 1)] = 1.0;
    }
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j < nx - 1; j++) {
                b[(size_t)i * nx + j] = 0.25 * (
                    a[(size_t)(i - 1) * nx + j] + a[(size_t)(i + 1) * nx + j] +
                    a[(size_t)i * nx + (j - 1)] + a[(size_t)i * nx + (j + 1)]);
            }
        }
        double *swap = a;
        a = b;
        b = swap;
    }
    for (int i = 0; i < n; i++) {
        memcpy(out + (size_t)i * nx, a + (size_t)(i + 1) * nx, (size_t)nx * sizeof(double));
    }
    free(a);
    free(b);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = (argc > 1) ? atoi(argv[1]) : 100;
    int max_iter = (argc > 2) ? atoi(argv[2]) : 1000;
    double tol = (argc > 3) ? atof(argv[3]) : 1e-6;

    if (n < size || n < 3 || max_iter < 1) {
        if (rank == 0) {
            fprintf(stderr, "Need N >= max(3, nprocs) and max_iter >= 1 (N=%d, np=%d)\n", n, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base = n / size;
    int rem = n % size;
    int local_rows = base + (rank < rem ? 1 : 0);
    int nx = n;
    int ny = local_rows + 2;

    double *blocks[2];
    double **rows[2];
    for (int which = 0; which < 2; which++) {
        blocks[which] = (double *)calloc((size_t)ny * nx, sizeof(double));
        rows[which] = (double **)malloc((size_t)ny * sizeof(double *));
        for (int i = 0; i < ny; i++) {
            rows[which][i] = blocks[which] + (size_t)i * nx;
            rows[which][i][nx - 1] = 1.0;
        }
    }

    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    const int tag_up = 10;
    const int tag_down = 11;

    double start = MPI_Wtime();
    int cur = 0;
    int iters_done = 0;
    double global_diff = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        double **u = rows[cur];
        double **u_new = rows[1 - cur];

        if (up != MPI_PROC_NULL) {
            MPI_Sendrecv(u[1], nx, MPI_DOUBLE, up, tag_up,
                         u[0], nx, MPI_DOUBLE, up, tag_down,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (down != MPI_PROC_NULL) {
            MPI_Sendrecv(u[local_rows], nx, MPI_DOUBLE, down, tag_down,
                         u[local_rows + 1], nx, MPI_DOUBLE, down, tag_up,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        double local_diff = 0.0;
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < nx - 1; j++) {
                u_new[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]);
                double delta = u_new[i][j] - u[i][j];
                local_diff += delta * delta;
            }
        }
        cur = 1 - cur;
        iters_done++;

        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_diff = sqrt(global_diff);
        if (rank == 0 && (iter % 100 == 0 || global_diff < tol)) {
            printf("Iter %d: diff = %.8e\n", iter, global_diff);
        }
        if (global_diff < tol) {
            if (rank == 0) {
                printf("Converged after %d iterations, diff = %.8e\n", iters_done, global_diff);
            }
            break;
        }
    }

    double elapsed = MPI_Wtime() - start;

    int local_count = local_rows * nx;
    double *mine = (double *)malloc((size_t)local_count * sizeof(double));
    for (int i = 0; i < local_rows; i++) {
        memcpy(mine + (size_t)i * nx, rows[cur][i + 1], (size_t)nx * sizeof(double));
    }

    int *counts = NULL;
    int *displs = NULL;
    double *all = NULL;
    if (rank == 0) {
        counts = (int *)calloc((size_t)size, sizeof(int));
        displs = (int *)calloc((size_t)size, sizeof(int));
        all = (double *)malloc((size_t)n * nx * sizeof(double));
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }
    MPI_Gatherv(mine, local_count, MPI_DOUBLE, all, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int failed = 0;
    if (rank == 0) {
        double *serial = (double *)malloc((size_t)n * nx * sizeof(double));
        serial_jacobi(n, iters_done, serial);
        double max_abs = 0.0;
        for (int i = 0; i < n * nx; i++) {
            double diff = fabs(all[i] - serial[i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
        }
        printf("Total elapsed time: %.4f seconds\n", elapsed);
        printf("Grid size: %d x %d, %d processes, iterations: %d\n", n, n, size, iters_done);
        printf("Max abs diff vs serial Jacobi: %.3e\n", max_abs);
        if (!(max_abs == 0.0) && max_abs > 1e-12) {
            fprintf(stderr, "parallel solution does not match serial reference\n");
            failed = 1;
        }
        if (all[nx - 1] != 1.0 || all[0] != 0.0) {
            fprintf(stderr, "boundary values were not preserved\n");
            failed = 1;
        }
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
        free(serial);
        free(all);
        free(counts);
        free(displs);
    }

    free(mine);
    for (int which = 0; which < 2; which++) {
        free(blocks[which]);
        free(rows[which]);
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
