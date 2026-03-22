// examples/04-hardware/cuda_aware.cu
// 演示 CUDA-aware MPI：直接发送 GPU 设备内存，不需要 CPU 拷贝
// 编译：
//   nvcc -c cuda_aware.cu -o cuda_aware.o
//   mpic++ cuda_aware.o -o cuda_aware -lcuda
// 运行：mpirun -np 2 ./cuda_aware
// 注意：需要你的 MPI 支持 CUDA-aware，否则会 segmentation fault

#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 照常初始化 MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    // --------------------------
    // 每个进程绑定到对应 GPU（假设每个进程一个 GPU）
    // 多进程多GPU，一个进程对应一张卡，这是标准做法
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    int device_id = rank % n_devices;
    cudaSetDevice(device_id);
    printf("Rank %d: using GPU %d\n", rank, device_id);

    // --------------------------
    // 直接在 GPU 上分配内存
    const int N = 1024;
    float *d_buf;
    cudaMalloc(&d_buf, N * sizeof(float));

    // --------------------------
    // rank 0 初始化数据在 GPU，直接发送给 rank 1
    if (rank == 0) {
        // 主机端初始化
        float *h_buf = new float[N];
        for (int i = 0; i < N; i++) {
            h_buf[i] = (float)i;
        }
        // 拷贝到 GPU
        cudaMemcpy(d_buf, h_buf, N * sizeof(float), cudaMemcpyHostToDevice);
        printf("Rank 0: initialized %d floats on GPU, sending to rank 1\n", N);

        // --------------------------
        // CUDA-aware MPI：直接发送 GPU 设备指针！
        // 不需要先拷到 CPU 再发送
        MPI_Send(d_buf, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        delete[] h_buf;
    } else if (rank == 1) {
        // 直接接收放到 GPU 内存
        printf("Rank 1: receiving directly into GPU memory\n");
        MPI_Recv(d_buf, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // --------------------------
        // 拷回 CPU 验证结果
        float *h_buf = new float[N];
        cudaMemcpy(h_buf, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Rank 1: received first 5 elements: %f %f %f %f %f\n",
               h_buf[0], h_buf[1], h_buf[2], h_buf[3], h_buf[4]);

        // 验证正确性
        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (h_buf[i] != (float)i) {
                ok = false;
                break;
            }
        }
        printf("Verification: %s\n", ok ? "PASS" : "FAIL");
        delete[] h_buf;
    }

    // --------------------------
    // 清理
    cudaFree(d_buf);
    MPI_Finalize();
    return 0;
}
