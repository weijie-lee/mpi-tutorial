#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int* sendbuf = (int*)malloc(N * sizeof(int));
    int* recvbuf = (int*)malloc(N * sizeof(int));
    
    // 初始化
    for (int i = 0; i < N; i++) {
        sendbuf[i] = rank * N + i;
    }
    
    // ===== 测试阻塞通信 =====
    double t1 = MPI_Wtime();
    if (rank == 0) {
        MPI_Send(sendbuf, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, N, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        MPI_Recv(recvbuf, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(sendbuf, N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    double t2 = MPI_Wtime();
    
    if (rank == 0) {
        printf("=== Blocking Communication ===\n");
        printf("Time: %.6f seconds\n", t2 - t1);
    }
    
    // ===== 测试非阻塞通信 =====
    MPI_Request req[2];
    
    t1 = MPI_Wtime();
    if (rank == 0) {
        MPI_Isend(sendbuf, N, MPI_INT, 1, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(recvbuf, N, MPI_INT, 1, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    } else if (rank == 1) {
        MPI_Irecv(recvbuf, N, MPI_INT, 0, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(sendbuf, N, MPI_INT, 0, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }
    t2 = MPI_Wtime();
    
    if (rank == 0) {
        printf("=== Non-blocking Communication ===\n");
        printf("Time: %.6f seconds\n", t2 - t1);
        printf("==============================\n");
    }
    
    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
