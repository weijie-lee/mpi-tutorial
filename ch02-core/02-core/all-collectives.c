// examples/02-core/all-collectives.c
// 演示所有常用集合通信原语的用法，带详细注释
// 编译：mpicc -O2 -o all-collectives all-collectives.c
// 运行：mpirun -np 4 ./all-collectives

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("===== Rank %d / %d =====\n", rank, size);

    // --------------------------
    // 1. MPI_Barrier：屏障同步
    // 所有进程都到这里才继续往下走
    printf("Rank %d: waiting at barrier\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== After Barrier, everyone continues ===\n\n");
    }

    // --------------------------
    // 2. MPI_Bcast：root 广播给所有进程
    int bcast_data = 0;
    if (rank == 0) {
        bcast_data = 12345;
        printf("Rank 0: Bcasting data %d\n", bcast_data);
    }
    // 所有进程都要调用！root 也要调用
    // 参数：&bcast_data 缓冲区，1 个元素，MPI_INT，root 是 0，通信域
    MPI_Bcast(&bcast_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: received bcast data %d\n", rank, bcast_data);

    // --------------------------
    // 3. MPI_Scatter：root 分发数据给所有进程
    int* sendbuf = NULL;
    int recvval;
    if (rank == 0) {
        sendbuf = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            sendbuf[i] = (i+1) * 10; // rank i 拿到 (i+1)*10
            printf("Rank 0: Scatter sending: ");
            for (int i = 0; i < size; i++) printf("%d ", sendbuf[i]);
            printf("\n");
        }
    }
    // 每个进程收一个 int
    // root 发送：sendbuf 每个进程一个，每个进程收一个放到 recvval
    MPI_Scatter(sendbuf, 1, MPI_INT, &recvval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Rank %d: Scatter received %d\n", rank, recvval);
    if (rank == 0) {
        free(sendbuf);
    }

    // --------------------------
    // 4. MPI_Gather：所有进程把数据收集到 root
    int sendval = recvval; // 刚才 Scatter 拿到的，现在 gather 回去
    int* recvbuf = NULL;
    if (rank == 0) {
        recvbuf = (int*)malloc(size * sizeof(int));
    }
    // 每个进程发一个 int，root 收集起来按 rank 拼
    MPI_Gather(&sendval, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Rank 0: Gather received: ");
        for (int i = 0; i < size; i++) printf("%d ", recvbuf[i]);
        printf("\n");
        free(recvbuf);
    }

    // --------------------------
    // 5. MPI_Allgather：所有进程收集到所有进程数据，每个进程都有完整结果
    int allgather_send = rank + 1;
    int* allgather_recv = (int*)malloc(size * sizeof(int));
    MPI_Allgather(&allgather_send, 1, MPI_INT, allgather_recv, 1, MPI_INT, MPI_COMM_WORLD);
    printf("Rank %d: Allgather received: ", rank);
    for (int i = 0; i < size; i++) printf("%d ", allgather_recv[i]);
    printf("\n");
    free(allgather_recv);

    // --------------------------
    // 6. MPI_Reduce：归约到 root
    double local = (double)(rank + 1);
    double global_sum;
    // 所有进程 local 求和，结果放到 root 的 global_sum
    MPI_Reduce(&local, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Rank 0: Reduce sum result: %.1f, expected: %.1f\n", 
               global_sum, (double)(size * (size + 1)) / 2.0);
    }

    // --------------------------
    // 7. MPI_Allreduce：归约完所有进程都得到结果
    // 这是深度学习最常用的！数据平行训练梯度平均用这个
    double allreduce_local = (double)(rank + 1);
    double allreduce_result;
    MPI_Allreduce(&allreduce_local, &allreduce_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("Rank %d: Allreduce sum result = %.1f, expected %.1f\n", 
           rank, allreduce_result, (double)(size * (size + 1)) / 2.0);

    // --------------------------
    // 8. MPI_Scan：前缀归约，每个 rank i 得到 0..i 的归约结果
    double scan_local = (double)(rank + 1);
    double scan_result;
    MPI_Scan(&scan_local, &scan_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double expected = (double)(rank + 1) * (rank + 2) / 2.0;
    printf("Rank %d: Scan prefix sum = %.1f, expected %.1f\n", 
           rank, scan_result, expected);

    MPI_Finalize();
    return 0;
}
