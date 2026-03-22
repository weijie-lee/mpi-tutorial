// examples/01-basics/hello.c
// 第一个 MPI 程序：Hello World
// 功能：启动多个进程，每个进程打印自己的编号
// 编译：mpicc -O2 -o hello hello.c
// 运行：mpirun -np 4 ./hello

// 必须包含 MPI 头文件，所有 MPI 函数/类型定义都在这里
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // --------------------------
    // 第一步：必须初始化 MPI 环境
    // MPI_Init 必须是程序中第一个调用的 MPI 函数
    // 参数：argc 和 argv 的指针，用来处理 MPI 启动时传给程序的参数
    int rank;   // 存储当前进程的 rank（编号）
    int size;   // 存储通信域中总共有多少进程

    MPI_Init(&argc, &argv);

    // --------------------------
    // 获取当前进程在默认通信域 MPI_COMM_WORLD 中的 rank
    // MPI_COMM_WORLD：默认通信域，包含所有启动的进程，是最常用的
    // rank 从 0 开始编号，每个进程有唯一的 rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // --------------------------
    // 获取 MPI_COMM_WORLD 中总共多少进程
    // 通常等于你 mpirun -np 指定的数目
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // 每个进程打印自己的信息
    // 注意：输出顺序不一定按 rank 顺序，因为进程调度是操作系统决定的
    // 这是正常现象，MPI 不保证输出顺序
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int len;
    // 额外获取运行这个进程的主机名，方便多节点运行时看分布
    MPI_Get_processor_name(hostname, &len);
    printf("Hello MPI! I'm rank %d / %d on host: %s\n", rank, size, hostname);

    // --------------------------
    // 最后一步：必须结束 MPI 环境
    // 清理所有 MPI 分配的资源
    // MPI_Finalize 之后不能再调用任何 MPI 函数
    MPI_Finalize();
    return 0;
}
