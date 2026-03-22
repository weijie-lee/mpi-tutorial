/*
 * NCCL 原生编程示例：AllReduce 求和
 * 编译：nvcc -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnccl -lmpi nccl_allreduce.cu -o nccl_allreduce
 * 运行：mpirun -np 2 ./nccl_allreduce （需要至少2个进程，每个进程对应一个GPU）
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <stdio.h>
#include <cmath>

int main(int argc, char** argv) {
    // 1. 先用MPI初始化，获取rank和size
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. 每个进程绑定到对应GPU（假设每个进程一个GPU）
    cudaError_t err = cudaSetDevice(rank);
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaSetDevice failed: %s\n",
                rank, cudaGetErrorString(err));
        MPI_Finalize();
        return 1;
    }

    // 3. 在GPU分配数据
    const int n = 1024;
    float *d_send, *d_recv;
    err = cudaMalloc(&d_send, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMalloc failed\n", rank);
        MPI_Finalize();
        return 1;
    }
    err = cudaMalloc(&d_recv, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMalloc failed\n", rank);
        cudaFree(d_send);
        MPI_Finalize();
        return 1;
    }

    // 初始化：每个rank i 所有元素都是 i+1
    float *h_send = new float[n];
    for (int i = 0; i < n; i++) {
        h_send[i] = rank + 1.0f;
    }
    err = cudaMemcpy(d_send, h_send, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_send;
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMemcpy failed\n", rank);
        MPI_Finalize();
        return 1;
    }

    // 4. NCCL 初始化
    // 多节点需要 MPI 广播 unique id
    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) {
        ncclResult_t res = ncclGetUniqueId(&id);
        if (res != ncclSuccess) {
            fprintf(stderr, "ncclGetUniqueId failed: %s\n", ncclGetErrorString(res));
            MPI_Finalize();
            return 1;
        }
    }
    // root 把 unique id 广播给所有进程
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    // 每个进程初始化自己的 rank
    ncclResult_t res = ncclCommInitRank(&comm, size, id, rank);
    if (res != ncclSuccess) {
        fprintf(stderr, "rank %d: ncclCommInitRank failed: %s\n",
                rank, ncclGetErrorString(res));
        MPI_Finalize();
        return 1;
    }

    // 5. 执行 AllReduce 求和
    // ncclSum: 求和，结果放到 d_recv 所有 GPU
    res = ncclAllReduce(d_send, d_recv, n, ncclFloat, ncclSum, comm, cudaStreamDefault);
    if (res != ncclSuccess) {
        fprintf(stderr, "rank %d: ncclAllReduce failed: %s\n",
                rank, ncclGetErrorString(res));
        ncclCommDestroy(comm);
        MPI_Finalize();
        return 1;
    }

    // 等待CUDA完成
    cudaDeviceSynchronize();

    // 6. 验证结果：所有 rank 结果都应该是 sum(1..size)
    float result;
    err = cudaMemcpy(&result, d_recv, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "rank %d: cudaMemcpy failed\n", rank);
        MPI_Finalize();
        return 1;
    }
    float expected = (float)(size * (size + 1)) / 2.0f;
    if (rank == 0) {
        printf("AllReduce result (first element): %.2f, expected: %.2f\n", result, expected);
        if (fabs(result - expected) < 1e-5) {
            printf("✓ Verification PASS\n");
        } else {
            printf("✗ Verification FAIL\n");
        }
    }

    // 7. 清理资源
    ncclCommDestroy(comm);
    cudaFree(d_send);
    cudaFree(d_recv);
    MPI_Finalize();
    return 0;
}
