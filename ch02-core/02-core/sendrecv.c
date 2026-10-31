// examples/02-core/sendrecv.c
// 点对点通信：阻塞收发，并用 MPI_Probe 接收长度事先未知的消息。
// 编译：mpicc -O2 -o sendrecv sendrecv.c
// 运行：mpirun -np 2 ./sendrecv

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "This example needs at least 2 processes, you have %d\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_fail = 0;

    if (rank == 0) {
        int data = 42;
        printf("Rank 0: sending int data = %d to rank 1\n", data);
        // count 是元素个数，不是字节数。tag 用来和别的消息区分。
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        int variable[5] = {10, 20, 30, 40, 50};
        MPI_Send(variable, 5, MPI_INT, 1, 1, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int received_data = -1;
        MPI_Status status;
        MPI_Recv(&received_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: received data = %d from rank 0\n", received_data);
        if (received_data != 42) {
            local_fail = 1;
        }

        int count = 0;
        MPI_Get_count(&status, MPI_INT, &count);
        printf("Rank 1: actually received %d elements\n", count);
        if (count != 1 || status.MPI_SOURCE != 0 || status.MPI_TAG != 0) {
            local_fail = 1;
        }

        // 长度未知时先 Probe：匹配规则是 (communicator, source, tag)。
        // MPI_ANY_SOURCE / MPI_ANY_TAG 可以放宽来源或标签，status 里再读出实际值。
        MPI_Status probe_status;
        MPI_Probe(0, 1, MPI_COMM_WORLD, &probe_status);
        int n = 0;
        MPI_Get_count(&probe_status, MPI_INT, &n);
        int *buf = (int *)malloc((size_t)n * sizeof(int));
        if (!buf) {
            local_fail = 1;
        } else {
            MPI_Recv(buf, n, MPI_INT, probe_status.MPI_SOURCE, probe_status.MPI_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Rank 1: probed %d elements:", n);
            int expect[5] = {10, 20, 30, 40, 50};
            if (n != 5) {
                local_fail = 1;
            }
            for (int i = 0; i < n; i++) {
                printf(" %d", buf[i]);
                if (i < 5 && buf[i] != expect[i]) {
                    local_fail = 1;
                }
            }
            printf("\n");
            free(buf);
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
