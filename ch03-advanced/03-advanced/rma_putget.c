// examples/03-advanced/rma_putget.c
// 演示 MPI 单侧通信（RMA - Remote Memory Access）put/get
// 单侧通信：一端可以直接读写另一端内存，不需要另一端主动参与
// 编译：mpicc -O2 -o rma_putget rma_putget.c
// 运行：mpirun -np 2 ./rma_putget

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    // 窗口内存由 MPI_Win_allocate 分配。MPI 可以选择它能够注册给网卡的地址。
    // 把栈上的 int 交给 MPI_Win_create 时，部分 OpenMPI 的 osc 组件会直接返回 MPI_ERR_WIN。
    // Win_free 会一并释放这块内存，不要再对它调用 free。
    MPI_Win win;
    int *data = NULL;
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &data, &win);
    if (rank == 0) {
        *data = 1234;
    } else {
        *data = 0;
    }

    // fence 划定一个 RMA 访问纪元。纪元里发起的 Get/Put 要到下一次 fence 之后才完成。
    // 在此之前读 Get 的本地缓冲是未定义的，所以检查必须放在 fence 之后。
    int recv_data = -1;
    MPI_Win_fence(0, win);
    if (rank == 1) {
        // target_disp 的单位是创建窗口时的 disp_unit（这里是 sizeof(int)），不是字节。
        MPI_Get(&recv_data, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
    }
    MPI_Win_fence(0, win);

    int local_fail = 0;
    if (rank == 1) {
        printf("Rank 1: got data %d from rank 0\n", recv_data);
        if (recv_data != 1234) {
            local_fail = 1;
        }
    }

    MPI_Win_fence(0, win);
    if (rank == 1) {
        int new_data = 5678;
        MPI_Put(&new_data, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
        printf("Rank 1: put new data %d to rank 0\n", new_data);
    }
    MPI_Win_fence(0, win);

    if (rank == 0) {
        printf("Rank 0: data after put from rank 1 is now: %d\n", *data);
        if (*data != 5678) {
            local_fail = 1;
        }
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }

    // --------------------------
    // 释放窗口
    MPI_Win_free(&win);

    MPI_Finalize();
    return failed ? 1 : 0;
}

/*
 * 单侧通信优势：
 * 1. 不需要对方 CPU 参与，适合动态不规则访问
 * 2. 一方可以主动读写另一方内存，握手更少
 * 3. 负载均衡、检查点常用
 */
