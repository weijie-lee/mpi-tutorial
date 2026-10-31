#ifndef COMMON_H
#define COMMON_H

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define BUFFER_SIZE 1024
#define VERBS_TIMEOUT_MS 10000

// TCP 上交换的连接信息。以前靠人手抄 QPN/GID，格式两边还不一致。
struct peer_info {
    uint32_t qpn;
    uint16_t lid;
    uint32_t psn;
    uint32_t rkey;
    uint64_t addr;
    int gid_index;
    union ibv_gid gid;
};

struct verbs_res {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    char *buffer;
    int port;
    uint16_t lid;
    int gid_index;
    union ibv_gid gid;
};

static inline int verbs_read_line(int fd, char *buf, int cap) {
    int n = 0;
    while (n < cap - 1) {
        char c;
        ssize_t r = recv(fd, &c, 1, 0);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (r == 0) {
            break;
        }
        if (c == '\n') {
            break;
        }
        buf[n++] = c;
    }
    buf[n] = '\0';
    return n;
}

static inline int verbs_write_all(int fd, const char *text) {
    size_t len = strlen(text);
    size_t off = 0;
    while (off < len) {
        ssize_t w = send(fd, text + off, len - off, 0);
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}

static inline int verbs_open(struct verbs_res *res) {
    memset(res, 0, sizeof(*res));
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (!list || num <= 0) {
        if (list) {
            ibv_free_device_list(list);
        }
        return -1;
    }
    int found = -1;
    for (int i = 0; i < num && found != 0; i++) {
        res->ctx = ibv_open_device(list[i]);
        if (!res->ctx) {
            continue;
        }
        struct ibv_device_attr dev_attr;
        if (ibv_query_device(res->ctx, &dev_attr)) {
            ibv_close_device(res->ctx);
            res->ctx = NULL;
            continue;
        }
        for (int port = 1; port <= dev_attr.phys_port_cnt; port++) {
            struct ibv_port_attr attr;
            if (ibv_query_port(res->ctx, port, &attr) || attr.state != IBV_PORT_ACTIVE) {
                continue;
            }
            res->port = port;
            res->lid = attr.lid;
            res->gid_index = -1;
            for (int g = 0; g < attr.gid_tbl_len; g++) {
                union ibv_gid gid;
                if (ibv_query_gid(res->ctx, port, g, &gid)) {
                    continue;
                }
                for (int b = 0; b < 16; b++) {
                    if (gid.raw[b]) {
                        res->gid = gid;
                        res->gid_index = g;
                        break;
                    }
                }
                if (res->gid_index >= 0) {
                    break;
                }
            }
            if (res->lid || res->gid_index >= 0) {
                printf("Using device %s port %d\n", ibv_get_device_name(list[i]), port);
                found = 0;
                break;
            }
        }
        if (found != 0) {
            ibv_close_device(res->ctx);
            res->ctx = NULL;
        }
    }
    ibv_free_device_list(list);
    return found;
}

static inline int verbs_create(struct verbs_res *res) {
    res->pd = ibv_alloc_pd(res->ctx);
    res->buffer = (char *)calloc(1, BUFFER_SIZE);
    if (!res->pd || !res->buffer) {
        return -1;
    }
    res->mr = ibv_reg_mr(res->pd, res->buffer, BUFFER_SIZE,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                             IBV_ACCESS_REMOTE_WRITE);
    res->cq = ibv_create_cq(res->ctx, 16, NULL, NULL, 0);
    if (!res->mr || !res->cq) {
        return -1;
    }
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = res->cq;
    attr.recv_cq = res->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = 16;
    attr.cap.max_recv_wr = 16;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    res->qp = ibv_create_qp(res->pd, &attr);
    return res->qp ? 0 : -1;
}

static inline void verbs_format(const struct verbs_res *res, char *out, size_t cap) {
    char gidhex[33];
    for (int i = 0; i < 16; i++) {
        sprintf(gidhex + 2 * i, "%02x", res->gid.raw[i]);
    }
    snprintf(out, cap, "%u %u %u %u %llu %d %s\n", res->qp->qp_num, res->lid, 0,
             res->mr->rkey, (unsigned long long)(uintptr_t)res->buffer, res->gid_index,
             gidhex);
}

static inline int verbs_parse(const char *line, struct peer_info *peer) {
    unsigned qpn, lid, psn, rkey;
    unsigned long long addr;
    int gid_index;
    char gidhex[33];
    if (sscanf(line, "%u %u %u %u %llu %d %32s", &qpn, &lid, &psn, &rkey, &addr,
               &gid_index, gidhex) != 7 ||
        strlen(gidhex) != 32) {
        return -1;
    }
    peer->qpn = qpn;
    peer->lid = (uint16_t)lid;
    peer->psn = psn;
    peer->rkey = rkey;
    peer->addr = addr;
    peer->gid_index = gid_index;
    for (int i = 0; i < 16; i++) {
        unsigned byte = 0;
        sscanf(gidhex + 2 * i, "%2x", &byte);
        peer->gid.raw[i] = (uint8_t)byte;
    }
    return 0;
}

// QP 必须经过 RESET -> INIT -> RTR -> RTS。跳过 INIT 时 modify 到 RTR 会失败。
static inline int verbs_to_rts(struct verbs_res *local, const struct peer_info *remote) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = local->port;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE;
    int rc = ibv_modify_qp(local->qp, &attr,
                           IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                               IBV_QP_ACCESS_FLAGS);
    if (rc) {
        fprintf(stderr, "INIT failed: %s\n", strerror(rc));
        return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote->qpn;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.port_num = local->port;
    if (remote->lid) {
        attr.ah_attr.dlid = remote->lid;
        attr.ah_attr.is_global = 0;
    } else {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = remote->gid;
        attr.ah_attr.grh.sgid_index = local->gid_index;
        attr.ah_attr.grh.hop_limit = 1;
    }
    rc = ibv_modify_qp(local->qp, &attr,
                       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                           IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) {
        fprintf(stderr, "RTR failed: %s\n", strerror(rc));
        return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    rc = ibv_modify_qp(local->qp, &attr,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                           IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc) {
        fprintf(stderr, "RTS failed: %s\n", strerror(rc));
        return -1;
    }
    return 0;
}

static inline int verbs_poll(struct ibv_cq *cq, struct ibv_wc *wc) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (;;) {
        int ne = ibv_poll_cq(cq, 1, wc);
        if (ne < 0) {
            return -1;
        }
        if (ne > 0) {
            return ne;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        long ms = (now.tv_sec - start.tv_sec) * 1000L +
                  (now.tv_nsec - start.tv_nsec) / 1000000L;
        if (ms > VERBS_TIMEOUT_MS) {
            return 0;
        }
        usleep(1000);
    }
}

static inline void verbs_destroy(struct verbs_res *res) {
    if (res->qp) {
        ibv_destroy_qp(res->qp);
    }
    if (res->cq) {
        ibv_destroy_cq(res->cq);
    }
    if (res->mr) {
        ibv_dereg_mr(res->mr);
    }
    if (res->pd) {
        ibv_dealloc_pd(res->pd);
    }
    if (res->ctx) {
        ibv_close_device(res->ctx);
    }
    free(res->buffer);
}

#endif
