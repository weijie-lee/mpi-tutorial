// examples/02-core/deadlock.c
// 点对点死锁：什么时候会卡住，以及三种不会卡住的写法。
//
// 编译：mpicc -O2 -o deadlock deadlock.c
// 运行：mpirun -np 2 ./deadlock
//       mpirun -np 2 ./deadlock ssend     # 双方都先 MPI_Ssend，这一模式会一直卡住
//
// 小消息的 MPI_Send 常常走 eager 协议：数据被拷进 MPI 内部缓冲后 Send 就能返回，
// 所以“两边都先 Send 再 Recv”不一定死锁。消息大到走 rendezvous，或者改用同步发送
// MPI_Ssend 时，发送完成依赖于对端已经开始接收，两边同时先发送就会死锁。

#include <mpi.h>
#include <stdio.h>
#include <string.h>

static int exchange_ordered(int rank) {
    int send = 100 + rank;
    int recv = -1;
    if (rank == 0) {
        MPI_Send(&send, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&recv, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&recv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&send, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    return recv == 100 + (1 - rank);
}

static int exchange_sendrecv(int rank) {
    int peer = 1 - rank;
    int send = 200 + rank;
    int recv = -1;
    // MPI_Sendrecv 在同一个调用里安排发送和接收，标准保证这一对调用不会自己死锁。
    MPI_Sendrecv(&send, 1, MPI_INT, peer, 1,
                 &recv, 1, MPI_INT, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return recv == 200 + peer;
}

static int exchange_nonblocking(int rank) {
    int peer = 1 - rank;
    int send = 300 + rank;
    int recv = -1;
    MPI_Request reqs[2];
    MPI_Irecv(&recv, 1, MPI_INT, peer, 2, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&send, 1, MPI_INT, peer, 2, MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    return recv == 300 + peer;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Run with exactly 2 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (argc > 1 && strcmp(argv[1], "ssend") == 0) {
        // 同步发送：两边都先 Ssend，后 Recv。用来观察死锁，验证脚本不要走这条路径。
        int send = rank;
        int recv = -1;
        if (rank == 0) {
            printf("Both ranks call MPI_Ssend first; this run is expected to hang.\n");
        }
        MPI_Ssend(&send, 1, MPI_INT, 1 - rank, 9, MPI_COMM_WORLD);
        MPI_Recv(&recv, 1, MPI_INT, 1 - rank, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Finalize();
        return 0;
    }

    int local_fail = 0;
    if (!exchange_ordered(rank)) {
        local_fail++;
    }
    if (!exchange_sendrecv(rank)) {
        local_fail++;
    }
    if (!exchange_nonblocking(rank)) {
        local_fail++;
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Ordered Send/Recv, MPI_Sendrecv and nonblocking exchange checked\n");
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }
    MPI_Finalize();
    return failed ? 1 : 0;
}
