#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>

#define BUFFER_SIZE 1024
#define GID_INDEX 0  // 通常用第一个GID

// 交换用的地址信息
struct rdma_addr {
    uint32_t qpn;
    union ibv_gid gid;
};

// 全局RDMA资源
struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    char *buffer;
};

// 注册内存
static inline struct ibv_mr *register_mr(struct ibv_pd *pd, void *addr, size_t size) {
    return ibv_reg_mr(pd, addr, size,
        IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE);
}

// 创建QP并初始化状态
static inline struct ibv_qp *create_qp(struct ibv_pd *pd, struct ibv_cq *cq, int qp_size) {
    struct ibv_qp_init_attr qp_attr = {0};
    qp_attr.send_cq = cq;
    qp_attr.recv_cq = cq;
    qp_attr.qp_type = IBV_QPT_RC;
    qp_attr.cap.max_send_wr = qp_size;
    qp_attr.cap.max_recv_wr = qp_size;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;
    return ibv_create_qp(pd, &qp_attr);
}

// 把QP调到RTR状态
static inline int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, union ibv_gid remote_gid) {
    struct ibv_qp_attr attr = {0};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.is_global = 1;          // GID 路由需要全局路由
    attr.ah_attr.grh.dgid = remote_gid;  // 目标 GID 放在 grh 里
    attr.ah_attr.grh.sgid_index = GID_INDEX;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER |
        IBV_QP_AV);
}

// 把QP调到RTS状态
static inline int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {0};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    return ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

#endif // COMMON_H
