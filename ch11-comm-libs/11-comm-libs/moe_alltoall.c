/* MoE dispatch / combine，用 MPI_Alltoallv 交换不等长的 token。
 *
 * 每个 rank 有 LOCAL_EXPERTS 个专家。token 0 的两个专家都在本 rank，
 * combine 前先在本地把加权结果加起来，所以回传消息数小于 dispatch 槽位数。
 * 专家函数是把隐藏向量乘以 expert_id+1。权重 0.5 和 0.25，最后加 bias 1。
 * rank 0 的 token 0 手算结果是 [1, 2, 3, 4]。
 *
 * 编译：mpicc -O2 -Wall -Wextra -Werror -std=c11 -o moe_alltoall moe_alltoall.c
 * 运行：mpirun -np 4 ./moe_alltoall
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 4, TOPK = 2, LOCAL_EXPERTS = 2 };

typedef struct {
    int src_rank;
    int token_id;
    int expert_id;
    int k;
    double weight;
    double hidden[HIDDEN];
} Slot;

static int route_expert(int rank, int token, int k, int num_experts) {
    if (token == 0) {
        return rank * LOCAL_EXPERTS + k;
    }
    if (token == 1 && k == 1) {
        return -1;
    }
    int e0 = (rank * 17 + token * 3) % num_experts;
    int expert = (rank * 17 + token * 3 + k * 5) % num_experts;
    if (k == 1 && expert == e0) {
        expert = (expert + 1) % num_experts;
    }
    return expert;
}

static double route_weight(int k) {
    return (k == 0) ? 0.5 : 0.25;
}

static double hidden_value(int rank, int token, int dim) {
    return (double)(rank * 1000 + token * 10 + dim);
}

static double token0_expect(int rank, int dim) {
    double scale = 1.5 * (double)rank + 1.0;
    return scale * (double)(rank * 1000 + dim) + 1.0;
}

static void oracle_token(int rank, int token, int num_experts, double out[HIDDEN]) {
    for (int d = 0; d < HIDDEN; d++) {
        out[d] = 1.0;
    }
    for (int k = 0; k < TOPK; k++) {
        int expert = route_expert(rank, token, k, num_experts);
        if (expert < 0) {
            continue;
        }
        double scale = route_weight(k) * (double)(expert + 1);
        for (int d = 0; d < HIDDEN; d++) {
            out[d] += scale * hidden_value(rank, token, d);
        }
    }
}

static int slot_bytes(int nslots) {
    return nslots * (int)sizeof(Slot);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ntokens = 8;
    if (argc > 1) {
        ntokens = atoi(argv[1]);
    }
    if (ntokens < 2 || ntokens > 100000) {
        if (rank == 0) {
            fprintf(stderr, "tokens must be in [2, 100000]\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int num_experts = size * LOCAL_EXPERTS;
    int local_fail = 0;
    int *send_n = calloc((size_t)size, sizeof(int));
    int *recv_n = calloc((size_t)size, sizeof(int));
    int *sdispls = calloc((size_t)size, sizeof(int));
    int *rdispls = calloc((size_t)size, sizeof(int));
    int *sendcounts = calloc((size_t)size, sizeof(int));
    int *recvcounts = calloc((size_t)size, sizeof(int));
    if (!send_n || !recv_n || !sdispls || !rdispls || !sendcounts || !recvcounts) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int token = 0; token < ntokens; token++) {
        for (int k = 0; k < TOPK; k++) {
            int expert = route_expert(rank, token, k, num_experts);
            if (expert < 0) {
                continue;
            }
            int dest = expert / LOCAL_EXPERTS;
            if (dest < 0 || dest >= size) {
                local_fail = 1;
                continue;
            }
            send_n[dest]++;
        }
    }

    int total_send = 0;
    for (int dest = 0; dest < size; dest++) {
        sdispls[dest] = total_send;
        total_send += send_n[dest];
    }
    Slot *packed = calloc((size_t)(total_send > 0 ? total_send : 1), sizeof(Slot));
    int *cursor = calloc((size_t)size, sizeof(int));
    if (!packed || !cursor) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int token = 0; token < ntokens; token++) {
        for (int k = 0; k < TOPK; k++) {
            int expert = route_expert(rank, token, k, num_experts);
            if (expert < 0) {
                continue;
            }
            int dest = expert / LOCAL_EXPERTS;
            Slot *slot = &packed[sdispls[dest] + cursor[dest]];
            cursor[dest]++;
            slot->src_rank = rank;
            slot->token_id = token;
            slot->expert_id = expert;
            slot->k = k;
            slot->weight = route_weight(k);
            for (int d = 0; d < HIDDEN; d++) {
                slot->hidden[d] = hidden_value(rank, token, d);
            }
        }
    }
    for (int dest = 0; dest < size; dest++) {
        if (cursor[dest] != send_n[dest]) {
            local_fail = 1;
        }
        sendcounts[dest] = slot_bytes(send_n[dest]);
        sdispls[dest] = slot_bytes(sdispls[dest]);
    }

    MPI_Alltoall(send_n, 1, MPI_INT, recv_n, 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int src = 0; src < size; src++) {
        rdispls[src] = total_recv;
        total_recv += recv_n[src];
        recvcounts[src] = slot_bytes(recv_n[src]);
    }
    for (int src = 0; src < size; src++) {
        rdispls[src] = slot_bytes(rdispls[src]);
    }

    int expect_recv = 0;
    int expect_pairs = 0;
    for (int src = 0; src < size; src++) {
        for (int token = 0; token < ntokens; token++) {
            int hit = 0;
            for (int k = 0; k < TOPK; k++) {
                int expert = route_expert(src, token, k, num_experts);
                if (expert < 0) {
                    continue;
                }
                if (expert / LOCAL_EXPERTS == rank) {
                    expect_recv++;
                    hit = 1;
                }
            }
            if (hit) {
                expect_pairs++;
            }
        }
    }
    if (total_recv != expect_recv) {
        local_fail = 1;
        fprintf(stderr, "rank %d recv slots %d expected %d\n", rank, total_recv, expect_recv);
    }

    Slot *inbox = calloc((size_t)(total_recv > 0 ? total_recv : 1), sizeof(Slot));
    if (!inbox) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Alltoallv(packed, sendcounts, sdispls, MPI_BYTE, inbox, recvcounts, rdispls, MPI_BYTE,
                  MPI_COMM_WORLD);

    double *partial = calloc((size_t)size * (size_t)ntokens * HIDDEN, sizeof(double));
    int *hits = calloc((size_t)size * (size_t)ntokens, sizeof(int));
    if (!partial || !hits) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < total_recv; i++) {
        Slot *slot = &inbox[i];
        if (slot->src_rank < 0 || slot->src_rank >= size || slot->token_id < 0 ||
            slot->token_id >= ntokens) {
            local_fail = 1;
            continue;
        }
        if (slot->expert_id / LOCAL_EXPERTS != rank) {
            local_fail = 1;
        }
        int index = slot->src_rank * ntokens + slot->token_id;
        hits[index]++;
        double scale = slot->weight * (double)(slot->expert_id + 1);
        for (int d = 0; d < HIDDEN; d++) {
            partial[(size_t)index * HIDDEN + (size_t)d] += scale * slot->hidden[d];
        }
    }

    int return_msgs = 0;
    for (int i = 0; i < size * ntokens; i++) {
        if (hits[i] > 0) {
            return_msgs++;
        }
    }
    if (return_msgs != expect_pairs) {
        local_fail = 1;
        fprintf(stderr, "rank %d combine msgs %d expected %d\n", rank, return_msgs, expect_pairs);
    }

    int *back_n = calloc((size_t)size, sizeof(int));
    if (!back_n) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int src = 0; src < size; src++) {
        for (int token = 0; token < ntokens; token++) {
            if (hits[src * ntokens + token] > 0) {
                back_n[src]++;
            }
        }
    }

    int total_back = 0;
    int *back_displ = calloc((size_t)size, sizeof(int));
    int *back_counts = calloc((size_t)size, sizeof(int));
    int *back_recv_n = calloc((size_t)size, sizeof(int));
    int *back_rdispl = calloc((size_t)size, sizeof(int));
    int *back_rcounts = calloc((size_t)size, sizeof(int));
    if (!back_displ || !back_counts || !back_recv_n || !back_rdispl || !back_rcounts) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int src = 0; src < size; src++) {
        back_displ[src] = total_back;
        total_back += back_n[src];
    }
    Slot *back_packed = calloc((size_t)(total_back > 0 ? total_back : 1), sizeof(Slot));
    if (!back_packed) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int src = 0; src < size; src++) {
        int fill = 0;
        for (int token = 0; token < ntokens; token++) {
            int index = src * ntokens + token;
            if (hits[index] == 0) {
                continue;
            }
            Slot *slot = &back_packed[back_displ[src] + fill];
            fill++;
            slot->src_rank = src;
            slot->token_id = token;
            slot->expert_id = -1;
            slot->k = -1;
            slot->weight = 1.0;
            for (int d = 0; d < HIDDEN; d++) {
                slot->hidden[d] = partial[(size_t)index * HIDDEN + (size_t)d];
            }
        }
        back_counts[src] = slot_bytes(back_n[src]);
        back_displ[src] = slot_bytes(back_displ[src]);
    }

    MPI_Alltoall(back_n, 1, MPI_INT, back_recv_n, 1, MPI_INT, MPI_COMM_WORLD);
    int total_back_recv = 0;
    for (int src = 0; src < size; src++) {
        back_rdispl[src] = total_back_recv;
        total_back_recv += back_recv_n[src];
    }
    for (int src = 0; src < size; src++) {
        back_rcounts[src] = slot_bytes(back_recv_n[src]);
        back_rdispl[src] = slot_bytes(back_rdispl[src]);
    }
    Slot *combined = calloc((size_t)(total_back_recv > 0 ? total_back_recv : 1), sizeof(Slot));
    if (!combined) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Alltoallv(back_packed, back_counts, back_displ, MPI_BYTE, combined, back_rcounts,
                  back_rdispl, MPI_BYTE, MPI_COMM_WORLD);

    double *result = calloc((size_t)ntokens * HIDDEN, sizeof(double));
    if (!result) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int token = 0; token < ntokens; token++) {
        for (int d = 0; d < HIDDEN; d++) {
            result[(size_t)token * HIDDEN + (size_t)d] = 1.0;
        }
    }
    for (int i = 0; i < total_back_recv; i++) {
        Slot *slot = &combined[i];
        if (slot->src_rank != rank || slot->token_id < 0 || slot->token_id >= ntokens) {
            local_fail = 1;
            continue;
        }
        for (int d = 0; d < HIDDEN; d++) {
            result[(size_t)slot->token_id * HIDDEN + (size_t)d] += slot->hidden[d];
        }
    }

    double local_max = 0.0;
    for (int token = 0; token < ntokens; token++) {
        double expect[HIDDEN];
        oracle_token(rank, token, num_experts, expect);
        for (int d = 0; d < HIDDEN; d++) {
            double diff = result[(size_t)token * HIDDEN + (size_t)d] - expect[d];
            if (diff < 0) {
                diff = -diff;
            }
            if (diff > local_max) {
                local_max = diff;
            }
            if (diff != 0.0) {
                local_fail = 1;
            }
        }
    }
    for (int d = 0; d < HIDDEN; d++) {
        double diff = result[d] - token0_expect(rank, d);
        if (diff < 0) {
            diff = -diff;
        }
        if (diff != 0.0) {
            local_fail = 1;
        }
        if (rank == 0) {
            double hand = (double)d + 1.0;
            double hand_diff = result[d] - hand;
            if (hand_diff < 0) {
                hand_diff = -hand_diff;
            }
            if (hand_diff != 0.0) {
                local_fail = 1;
            }
        }
    }

    int global_slots = 0;
    int global_msgs = 0;
    double global_max = 0.0;
    MPI_Allreduce(&total_send, &global_slots, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&return_msgs, &global_msgs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (global_max != 0.0 || global_msgs >= global_slots) {
        local_fail = 1;
    }

    int failed = 0;
    MPI_Allreduce(&local_fail, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("ranks=%d tokens=%d slots=%d combine_msgs=%d max_abs_diff=%.3e rank0_token0=",
               size, ntokens, global_slots, global_msgs, global_max);
        for (int d = 0; d < HIDDEN; d++) {
            printf("%s%.0f", d ? "," : "", result[d]);
        }
        printf("\n");
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
        fflush(stdout);
    }

    free(send_n);
    free(recv_n);
    free(sdispls);
    free(rdispls);
    free(sendcounts);
    free(recvcounts);
    free(packed);
    free(cursor);
    free(inbox);
    free(partial);
    free(hits);
    free(back_n);
    free(back_displ);
    free(back_counts);
    free(back_recv_n);
    free(back_rdispl);
    free(back_rcounts);
    free(back_packed);
    free(combined);
    free(result);
    MPI_Finalize();
    return failed ? 1 : 0;
}
