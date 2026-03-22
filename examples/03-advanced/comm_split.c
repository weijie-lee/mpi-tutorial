// examples/03-advanced/comm_split.c
// 演示 MPI_Comm_split：把一个大通信域分裂成多个小通信域
// 示例：按奇偶rank分裂，奇数一组，偶数一组
// 编译：mpicc -O2 -o comm_split comm_split.c
// 运行：mpirun -np 4 ./comm_split

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // MPI_Comm_split 把原通信域按 color 分裂
    // color 相同的进程分到同一个新通信域
    // 这里我们按 rank 奇偶分：偶数 color=0，奇数 color=1
    int color = rank % 2;
    int key = rank; // key 用来在新通信域中排序 rank，key 小 rank 小
    MPI_Comm new_comm; // 输出新的通信域
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);

    // --------------------------
    // 在新通信域里获取自己的 rank 和 size
    int new_rank, new_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    printf("Original rank=%d, color=%d → New comm: rank=%d, size=%d\n",
           rank, color, new_rank, new_size);

    // --------------------------
    // 用完释放新通信域
    MPI_Comm_free(&new_comm);

    MPI_Finalize();
    return 0;
}
/*
 * 示例输出（np=4）：
 * Original rank=0, color=0 → New comm: rank=0, size=2
 * Original rank=1, color=1 → New comm: rank=0, size=2
 * Original rank=2, color=0 → New comm: rank=1, size=2
 * Original rank=3, color=1 → New comm: rank=1, size=2
 *
 * 应用场景：
 * 1. 分组计算：不同组做不同任务
 * 2. 分层通信：节点内一组，节点间一组
 * 3. 隔离不同模块的通信，避免消息混淆
 */
