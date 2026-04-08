// examples/02-core/collectives.c
// 演示常用集合通信操作：Bcast / Reduce / Allreduce / Scatter / Gather
// 编译：mpicc -O2 -o collectives collectives.c
// 运行：mpirun -np 4 ./collectives

#include <mpi.h>
#include <stdio.h>

/*
 * 集合通信要点：
 * 1. 通信域中 ALL 进程必须调用同一个集合通信操作
 * 2. MPI 实现会针对拓扑做优化，比自己用点对点实现快很多
 * 3. 参数要所有进程一致，不一致会出错
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------
    // 1. Bcast 示例：root 0 广播数据给所有进程
    int data = 0;
    if (rank == 0) {
        data = 12345;  // root 初始化数据
        printf("Root (rank 0) broadcasting data: %d\n", data);
    }

    // --------------------------
    // MPI_Bcast：所有进程都要调用！
    // 参数：
    // &data：发送（root）/ 接收（其他）缓冲区
    // 1：一个元素
    // MPI_INT：元素类型
    // 0：root 的 rank
    // MPI_COMM_WORLD：通信域
    //
    // 调用完后，所有进程的 data 都等于 root 的值
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: received broadcast data: %d\n", rank, data);

    // --------------------------
    // 2. Reduce 示例：所有进程的局部值归约到 root
    double local = rank * rank;  // 每个进程局部值是 rank^2
    double global_sum;

    // MPI_Reduce：把所有进程的 local 按操作归约，结果放到 root 的 global_sum
    // 参数：
    // &local：输入（每个进程）
    // &global_sum：输出（只有 root 有效）
    // 1：一个元素
    // MPI_DOUBLE：类型
    // MPI_SUM：归约操作：求和
    // 0：结果输出到 root 0
    MPI_Reduce(&local, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nSum of rank^2 from all ranks: %f (expected %f)\n", 
               global_sum, (size-1)*size*(2*size-1)/6.0);
    }

    // --------------------------
    // 3. Allreduce 示例：所有进程都得到归约结果
    // 和 Reduce 的区别就是：所有进程都拿到结果，不只是 root
    double global_sum_all;
    MPI_Allreduce(&local, &global_sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: global sum via Allreduce: %f\n", rank, global_sum_all);

    // --------------------------
    // 4. Scatter 示例：root 把数组分发给所有进程，每个进程一块
    int sendbuf[4];
    int recvval;
    if (rank == 0) {
        // root 初始化，每个进程 i 对应值 i*10
        for (int i = 0; i < size; i++) {
            sendbuf[i] = i * 10;
        }
        printf("\nScattering from root: ");
        for (int i = 0; i < size; i++) printf("%d ", sendbuf[i]);
        printf("\n");
    }

    // MPI_Scatter：
    // root 把 sendbuf 分成 size 块，每个进程一块
    // 每个进程把自己那块放到 recvval
    MPI_Scatter(sendbuf, 1, MPI_INT, &recvval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: scattered value: %d\n", rank, recvval);

    // --------------------------
    // 5. Gather 示例：每个进程把自己的值收集到 root，拼到数组里
    // 正好是 Scatter 反过来
    // 每个进程把 recvval 乘以 2，再 gather 回 root
    recvval *= 2;
    int recvbuf[4];
    MPI_Gather(&recvval, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("After gather (doubled): ");
        for (int i = 0; i < size; i++) printf("%d ", recvbuf[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
