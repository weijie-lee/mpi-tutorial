#include <mpi.h>
#include <stdio.h>

/*
 * MPI_Comm_split 示例：按颜色分裂通信域
 * 运行：mpirun -np 4 ./comm_split
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 按奇偶颜色分组
    int color = rank % 2;
    int key = rank; // key 决定新通信域中的 rank 顺序

    MPI_Comm newcomm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm);

    int new_rank, new_size;
    MPI_Comm_rank(newcomm, &new_rank);
    MPI_Comm_size(newcomm, &new_size);

    printf("Original rank %d: color = %d, new rank = %d, new size = %d\n",
           rank, color, new_rank, new_size);

    // 可以在新通信域做集合操作，只影响本组
    int value = rank;
    int sum;
    MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, newcomm);
    printf("Original rank %d: sum within group = %d\n", rank, sum);

    MPI_Comm_free(&newcomm);
    MPI_Finalize();
    return 0;
}
