// examples/03-advanced/vector_indexed.c
// 向量类型取矩阵的一列，索引类型取不连续元素。
// MPI_Type_vector(count, blocklength, stride, oldtype, newtype)
//   count       块的个数
//   blocklength 每块的元素个数
//   stride      相邻块起始位置相差的元素个数（不是字节）
// 列向量是 vector(nrows, 1, ncols)，不是 vector(nrows, ncols, ncols)。后者只是一段连续内存。
// 编译：mpicc -O2 -o vector_indexed vector_indexed.c
// 运行：mpirun -np 2 ./vector_indexed

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 6

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Need at least 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_fail = 0;
    const int column = 2;

    if (rank == 0) {
        int matrix[ROWS][COLS];
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                matrix[r][c] = r * 10 + c;
            }
        }

        MPI_Datatype column_type;
        MPI_Type_vector(ROWS, 1, COLS, MPI_INT, &column_type);
        MPI_Type_commit(&column_type);
        MPI_Send(&matrix[0][column], 1, column_type, 1, 0, MPI_COMM_WORLD);
        MPI_Type_free(&column_type);

        int blocklengths[] = {1, 1, 1};
        int displacements[] = {1, 4, 7};
        MPI_Datatype indexed_type;
        MPI_Type_indexed(3, blocklengths, displacements, MPI_INT, &indexed_type);
        MPI_Type_commit(&indexed_type);
        MPI_Send(&matrix[0][0], 1, indexed_type, 1, 1, MPI_COMM_WORLD);
        MPI_Type_free(&indexed_type);
    } else if (rank == 1) {
        int column_buf[ROWS];
        MPI_Recv(column_buf, ROWS, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Column %d:", column);
        for (int r = 0; r < ROWS; r++) {
            int expect = r * 10 + column;
            printf(" %d", column_buf[r]);
            if (column_buf[r] != expect) {
                local_fail = 1;
            }
        }
        printf("\n");

        int picked[3];
        MPI_Recv(picked, 3, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 位移单位是元素。4x6 行主序里，下标 7 是第 1 行第 1 列，值是 11，不是 7。
        int expect_idx[3] = {1, 4, 11};
        printf("Indexed elements:");
        for (int i = 0; i < 3; i++) {
            printf(" %d", picked[i]);
            if (picked[i] != expect_idx[i]) {
                local_fail = 1;
            }
        }
        printf("\n");
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
