#include <mpi.h>
#include <stdio.h>
#include <stddef.h>

/*
 * 派生数据类型示例：发送自定义结构体
 */

typedef struct {
    int id;
    double x;
    double y;
} Particle;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 定义派生数据类型
    int          count = 3;
    int          blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint     offsets[3];

    offsets[0] = offsetof(Particle, id);
    offsets[1] = offsetof(Particle, x);
    offsets[2] = offsetof(Particle, y);

    MPI_Datatype MPI_Particle;
    MPI_Type_create_struct(count, blocklengths, offsets, types, &MPI_Particle);
    MPI_Type_commit(&MPI_Particle);

    if (rank == 0) {
        Particle p = {1, 1.5, 2.5};
        printf("Rank 0: sending Particle {id=%d, x=%.1f, y=%.1f} to rank 1\n",
               p.id, p.x, p.y);
        MPI_Send(&p, 1, MPI_Particle, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1 && size >= 2) {
        Particle p;
        MPI_Status status;
        MPI_Recv(&p, 1, MPI_Particle, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: received Particle {id=%d, x=%.1f, y=%.1f}\n",
               p.id, p.x, p.y);
    }

    // 释放类型
    MPI_Type_free(&MPI_Particle);

    MPI_Finalize();
    return 0;
}
