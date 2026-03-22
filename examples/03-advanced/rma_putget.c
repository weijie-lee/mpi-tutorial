#include <mpi.h>
#include <stdio.h>

/*
 * RMA 单侧通信示例：Put 写远程内存，Get 读远程内存
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        printf("Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    const int n = 4;
    double local_buf[n];

    // 初始化本地数据
    for (int i = 0; i < n; i++) {
        local_buf[i] = rank * 100 + i;
    }

    printf("Rank %d initial: ", rank);
    for (int i = 0; i < n; i++) {
        printf("%.0f ", local_buf[i]);
    }
    printf("\n");

    // 创建窗口，暴露本地缓冲区
    MPI_Win win;
    MPI_Win_create(local_buf, n * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    int other = 1 - rank;

    // 第一个栅栏同步，准备好窗口
    MPI_Win_fence(0, win);

    if (rank == 0) {
        double send[2] = {999, 888};
        // rank 0 把数据写到 rank 1 缓冲区的偏移 1 位置
        MPI_Put(send, 2, MPI_DOUBLE, other, 1, 2, MPI_DOUBLE, win);
        printf("Rank 0 put [999, 888] to rank 1 at offset 1\n");
    }

    // 栅栏保证操作完成
    MPI_Win_fence(0, win);

    if (rank == 1) {
        printf("Rank 1 after put: ");
        for (int i = 0; i < n; i++) {
            printf("%.0f ", local_buf[i]);
        }
        printf("\n");
    }

    // 再同步，然后 Get 操作
    MPI_Win_fence(0, win);

    if (rank == 0) {
        double recv[3];
        // rank 0 从 rank 1 读 3 个元素，从偏移 0 开始
        MPI_Get(recv, 3, MPI_DOUBLE, other, 0, 3, MPI_DOUBLE, win);
        MPI_Win_fence(0, win);

        printf("Rank 0 got from rank 1: ");
        for (int i = 0; i < 3; i++) {
            printf("%.0f ", recv[i]);
        }
        printf("\n");
    }

    MPI_Win_fence(0, win);

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
