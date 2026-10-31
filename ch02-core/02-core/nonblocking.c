// examples/02-core/nonblocking.c
// 非阻塞通信：先 Irecv 再 Isend，计算和传输重叠，最后 Waitall。
// 编译：mpicc -O2 -o nonblocking nonblocking.c
// 运行：mpirun -np 2 ./nonblocking
//       进程数为任意偶数时两两配对；奇数时最后一个进程不参与。

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Need at least 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_fail = 0;
    int peer = rank ^ 1;
    if (peer < size) {
        int send_data = rank;
        int recv_data = -1;
        MPI_Request reqs[2];

        // 先投递接收，再发送。对端同样如此，就不会出现“谁都还没 Recv”的窗口。
        MPI_Irecv(&recv_data, 1, MPI_INT, peer, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(&send_data, 1, MPI_INT, peer, 0, MPI_COMM_WORLD, &reqs[1]);

        // 请求已经交给 MPI。这里的计算不能读写 send_data / recv_data。
        long long sum = 0;
        for (long long i = 0; i < 1000000; i++) {
            sum += i;
        }
        if (sum < 0) {
            local_fail = 1;
        }

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        printf("Rank %d: sent %d, received %d from rank %d\n",
               rank, send_data, recv_data, peer);
        if (recv_data != peer) {
            local_fail = 1;
        }
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
