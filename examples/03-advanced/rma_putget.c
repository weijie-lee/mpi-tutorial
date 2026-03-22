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

    // --------------------------
    // 单侧通信第一步：每个进程暴露自己一块内存给对方访问
    // 这个叫做 创建窗口（Window）
    int data;
    MPI_Win win; // 窗口句柄

    if (rank == 0) {
        data = 1234; // rank 0 初始化数据
    } else {
        data = 0;
    }

    // --------------------------
    // 创建窗口：暴露本地 data 给远程访问
    // MPI_Win_create(
    //     &data,      // 暴露的内存起始地址
    //     sizeof(int), // 暴露的内存大小（字节）
    //     sizeof(int), // 单位大小
    //     MPI_INFO_NULL, // 信息，默认用 NULL
    //     MPI_COMM_WORLD, // 通信域
    //     &win // 输出窗口句柄
    // );
    MPI_Win_create(&data, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // --------------------------
    //  fence 同步：所有进程进入 RMA 访问阶段
    // 所有进程必须调用 MPI_Win_fence 才能开始访问
    MPI_Win_fence(0, win);

    // --------------------------
    // rank 1 从 rank 0 读数据
    // MPI_Get：把远程 window 的数据读到本地
    if (rank == 1) {
        int recv_data;
        // MPI_Get(
        //     &recv_data, // 本地接收缓冲区
        //     1, MPI_INT, // 接收 1 个 int
        //     0, // 目标进程 rank（我们读 rank 0 的数据）
        //     0, // 目标偏移（字节），从窗口起始第 0 字节开始读
        //     1, MPI_INT, // 读 1 个 int
        //     win, // 窗口
        // );
        MPI_Get(&recv_data, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
        printf("Rank 1: got data %d from rank 0\n", recv_data);

        // 我们再改一下 rank 0 的数据，用 MPI_Put 写过去
        int new_data = 5678;
        MPI_Put(&new_data, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
        printf("Rank 1: put new data %d to rank 0\n", new_data);
    }

    // --------------------------
    //  fence 同步：所有 RMA 访问完成
    MPI_Win_fence(0, win);

    // rank 0 看看数据是不是被改了
    if (rank == 0) {
        printf("Rank 0: data after put from rank 1 is now: %d\n", data);
    }

    // --------------------------
    // 释放窗口
    MPI_Win_free(&win);

    MPI_Finalize();
    return 0;
}

/*
 * 单侧通信优势：
 * 1. 不需要对方 CPU 参与，适合动态不规则访问
 * 2. 一方可以主动读写另一方内存，握手更少
 * 3. 负载均衡、检查点常用
 */
