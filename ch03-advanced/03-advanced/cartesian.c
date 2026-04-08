// examples/03-advanced/cartesian.c
// 演示 MPI 笛卡尔虚拟拓扑：创建二维网格拓扑，找邻居
// 常用于物理模拟（CFD、分子动力学等），网格分块
// 编译：mpicc -O2 -o cartesian cartesian.c
// 运行：mpirun -np 4 ./cartesian （最好 np=4，就是 2x2 网格）

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // 创建 2D 笛卡尔拓扑
    // 我们尽量分配成接近方形的网格
    int dims[2] = {0, 0}; // 0 让 MPI 自动计算每个维度多少进程
    MPI_Dims_create(size, 2, dims);
    if (rank == 0) {
        printf("Creating %d x %d 2D grid for %d processes\n", dims[0], dims[1], size);
    }

    int periods[2] = {0, 0}; // 两个维度都不是周期性边界（0=不周期，1=周期）
    int reorder = 1;           // 允许 MPI 重新排序 rank 优化拓扑亲和性
    MPI_Comm cart_comm;       // 新的通信域，对应笛卡尔拓扑
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    // --------------------------
    // 获取当前进程在拓扑中的坐标
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    printf("Rank %d: coords (%d, %d) in grid\n", rank, coords[0], coords[1]);

    // --------------------------
    // 找邻居：shift 找 0 维度（x方向）移动 1 步的左右邻居
    int left_rank, right_rank;
    // MPI_Cart_shift：获取移动偏移后的 rank
    // 参数：通信域，哪个维度，偏移多少，输出源rank，输出目的rank
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
    printf("Rank %d (%d,%d): left neighbor rank=%d, right neighbor rank=%d\n",
           rank, coords[0], coords[1], left_rank, right_rank);

    // 同样可以找 y 方向上下邻居
    int up_rank, down_rank;
    MPI_Cart_shift(cart_comm, 1, 1, &up_rank, &down_rank);
    printf("Rank %d (%d,%d): up neighbor rank=%d, down neighbor rank=%d\n",
           rank, coords[0], coords[1], up_rank, down_rank);

    // --------------------------
    // 用完释放拓扑通信域
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}
