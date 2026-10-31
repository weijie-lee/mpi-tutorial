// examples/03-advanced/cartesian.c
// 二维笛卡尔拓扑。reorder=1 时，新通信域里的 rank 可以和 MPI_COMM_WORLD 不同。
// 坐标、shift、Sendrecv 都必须使用 cart_comm 里的 rank。
// 编译：mpicc -O2 -o cartesian cartesian.c
// 运行：mpirun -np 4 ./cartesian

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int world_rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    if (world_rank == 0) {
        printf("Creating %d x %d 2D grid for %d processes\n", dims[0], dims[1], size);
    }

    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    int back = -1;
    MPI_Cart_rank(cart_comm, coords, &back);

    // 维度 0 是 MPI_Dims_create 的第一维，这里把它当作“行 / 左右”。
    int left_rank, right_rank, up_rank, down_rank;
    MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &up_rank, &down_rank);
    printf("World rank %d -> cart rank %d coords (%d, %d), neighbors L=%d R=%d U=%d D=%d\n",
           world_rank, cart_rank, coords[0], coords[1],
           left_rank, right_rank, up_rank, down_rank);

    int token = cart_rank;
    int from_left = -1;
    // 源或目的是 MPI_PROC_NULL 时，Sendrecv 的这一侧是空操作，缓冲区保持原值。
    MPI_Sendrecv(&token, 1, MPI_INT, right_rank, 0,
                 &from_left, 1, MPI_INT, left_rank, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    int local_fail = 0;
    if (back != cart_rank) {
        local_fail = 1;
    }
    if (left_rank == MPI_PROC_NULL) {
        if (from_left != -1) {
            local_fail = 1;
        }
    } else if (from_left != left_rank) {
        local_fail = 1;
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, cart_comm);
    if (cart_rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return failed ? 1 : 0;
}
