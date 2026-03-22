#include <mpi.h>
#include <stdio.h>

/*
 * 笛卡尔拓扑示例：创建二维网格拓扑，获取邻居进程
 * 运行：mpirun -np 4 ./cartesian
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 创建 2x2 二维笛卡尔拓扑
    int dims[2] = {0, 0};  // 0 表示让 MPI 自动分担进程数
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {0, 0}; // 不循环
    int reorder = 1;         // 允许重新排序优化
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    // 获取当前进程在拓扑中的坐标
    int cart_rank, coords[2];
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    printf("Rank %d: coords = (%d, %d), dims = %d x %d\n",
           rank, coords[0], coords[1], dims[0], dims[1]);

    // 获取上下左右邻居
    int left, right, up, down;
    MPI_Cart_shift(cart_comm, 1, 1, &cart_rank, &left);   // x 方向移动
    MPI_Cart_shift(cart_comm, 0, 1, &cart_rank, &down);   // y 方向移动

    printf("Rank %d (%d,%d): left=%d, down=%d\n",
           rank, coords[0], coords[1], left, down);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
