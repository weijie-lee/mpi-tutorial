// examples/03-advanced/comm_split.c
// MPI_Comm_split 按 color 分组；MPI_Group_incl 按显式 rank 列表建组。
// 编译：mpicc -O2 -o comm_split comm_split.c
// 运行：mpirun -np 4 ./comm_split

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int color = rank % 2;
    int key = rank;
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);

    int new_rank = -1, new_size = -1;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);
    printf("Original rank=%d, color=%d -> New comm: rank=%d, size=%d\n",
           rank, color, new_rank, new_size);

    int local_fail = 0;
    int expect_size = (color == 0) ? (size + 1) / 2 : size / 2;
    if (new_size != expect_size || new_rank != rank / 2) {
        local_fail = 1;
    }

    int ones = 1;
    int group_members = 0;
    MPI_Allreduce(&ones, &group_members, 1, MPI_INT, MPI_SUM, new_comm);
    if (group_members != new_size) {
        local_fail = 1;
    }

    MPI_Group world_group, even_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int n_even = (size + 1) / 2;
    int *even_ranks = (int *)malloc((size_t)n_even * sizeof(int));
    for (int i = 0; i < n_even; i++) {
        even_ranks[i] = i * 2;
    }
    MPI_Group_incl(world_group, n_even, even_ranks, &even_group);

    int group_rank = -1, group_size = -1;
    MPI_Group_rank(even_group, &group_rank);
    MPI_Group_size(even_group, &group_size);
    if (group_size != n_even) {
        local_fail = 1;
    }
    if (rank % 2 == 0) {
        if (group_rank != rank / 2) {
            local_fail = 1;
        }
    } else if (group_rank != MPI_UNDEFINED) {
        local_fail = 1;
    }

    if (rank == 0) {
        printf("Even-rank group size = %d\n", group_size);
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }

    MPI_Group_free(&even_group);
    MPI_Group_free(&world_group);
    free(even_ranks);
    MPI_Comm_free(&new_comm);
    MPI_Finalize();
    return failed ? 1 : 0;
}
