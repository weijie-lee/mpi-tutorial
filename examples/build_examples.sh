#!/bin/bash
# 一键编译所有C/C++示例，CUDA示例需要你的MPI支持CUDA

# 基础示例
echo "Building 01-basics..."
cd 01-basics
mpicc -O2 -o hello hello.c
mpicc -O2 -o timing timing.c
mpicc -O2 -o error_handling error_handling.c
cd ..

# 核心示例
echo "Building 02-core..."
cd 02-core
mpicc -O2 -o sendrecv sendrecv.c
mpicc -O2 -o deadlock deadlock.c
mpicc -O2 -o nonblocking nonblocking.c
mpicc -O2 -o collectives collectives.c
mpicc -O2 -o pi_monte_carlo pi_monte_carlo.c
cd ..

# 进阶示例
echo "Building 03-advanced..."
cd 03-advanced
mpicc -O2 -o derived_type derived_type.c
mpicc -O2 -o cartesian cartesian.c
mpicc -O2 -o comm_split comm_split.c
mpicc -O2 -o rma_putget rma_putget.c
cd ..

# GPU/RDMA示例，如果nvcc存在就编译
echo "Building 04-hardware..."
cd 04-hardware
# 编译CUDA-aware MPI示例
if command -v nvcc >/dev/null 2>&1; then
    nvcc -c cuda_aware.cu -o cuda_aware.o
    mpic++ cuda_aware.o -o cuda_aware -lcuda
    echo "Built CUDA example cuda_aware"
else
    echo "nvcc not found, skipping CUDA example"
fi
# 编译RDMA libverbs示例，如果libverbs存在
if [ -f /usr/include/infiniband/verbs.h ]; then
    mpicc -O2 -o rdma_write_server rdma_write_server.c -lrdmacm -libverbs
    mpicc -O2 -o rdma_write_client rdma_write_client.c -lrdmacm -libverbs
    echo "Built RDMA examples rdma_write_server/rdma_write_client"
else
    echo "libverbs not found, skipping RDMA examples"
fi
cd ..

# 完整应用示例
echo "Building 06-applications..."
cd 06-applications
mpicc -O2 -o jacobi2d jacobi2d.c -lm
echo "Built jacobi2d"
cd ..

# RDMA Verbs 示例，如果libverbs存在就编译
echo "Building 08-rdma-verbs..."
cd 08-rdma-verbs
if [ -f /usr/include/infiniband/verbs.h ]; then
    make
    echo "Built RDMA Verbs examples server/client"
else
    echo "libibverbs-dev not found, skipping RDMA Verbs examples"
fi
cd ..

echo "Done. Use mpirun -np <N> <program> to run."
