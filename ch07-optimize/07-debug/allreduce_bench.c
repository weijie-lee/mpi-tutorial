#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 分配内存
    int* sendbuf = (int*)malloc(N * sizeof(int));
    int* recvbuf = (int*)malloc(N * sizeof(int));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        sendbuf[i] = rank + i;
    }
    
    // 同步
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // 测试 Allreduce
    int local_sum = 0;
    for (int i = 0; i < N; i++) {
        local_sum += sendbuf[i];
    }
    
    int global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("=== Allreduce Benchmark ===\n");
        printf("Processes: %d\n", size);
        printf("Elements: %d\n", N);
        printf("Local sum: %d\n", local_sum);
        printf("Global sum: %d\n", global_sum);
        printf("Time: %.6f seconds\n", end_time - start_time);
        printf("========================\n");
    }
    
    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
