#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

/*
 * CUDA-aware MPI 示例：直接发送GPU设备内存
 * 编译：nvcc -c cuda_aware.cu -o cuda_aware.o && mpic++ cuda_aware.o -o cuda_aware -lcuda
 * 运行：mpirun -np 2 ./cuda_aware
 * 要求MPI已经编译支持CUDA
 */

__global__ void init_kernel(float *buf, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buf[idx] = val + idx;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cout << "Need at least 2 processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const int n = 1024;

    // 分配GPU内存
    float *d_buf;
    cudaError_t err = cudaMalloc(&d_buf, n * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Rank " << rank << " cudaMalloc failed: "
                  << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
    }

    // 初始化
    if (rank == 0) {
        init_kernel<<<(n+255)/256, 256>>>(d_buf, n, 1.0f);
        cudaDeviceSynchronize();
        std::cout << "Rank 0 initialized GPU buffer, sending to rank 1" << std::endl;
    }

    // CUDA-aware MPI 直接发送设备内存
    if (rank == 0) {
        MPI_Send(d_buf, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Status status;
        MPI_Recv(d_buf, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);

        // 拷贝回CPU检查结果
        float *h_buf = new float[n];
        cudaMemcpy(h_buf, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Rank 1 received first 5 elements: ";
        for (int i = 0; i < 5 && i < n; i++) {
            std::cout << h_buf[i] << " ";
        }
        std::cout << "..." << std::endl;
        delete[] h_buf;
    }

    cudaFree(d_buf);
    MPI_Finalize();
    return 0;
}
