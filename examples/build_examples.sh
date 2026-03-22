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

# GPU示例，如果nvcc存在就编译
echo "Building 04-hardware..."
cd 04-hardware
if command -v nvcc >/dev/null 2>&1; then
    nvcc -c cuda_aware.cu -o cuda_aware.o
    mpic++ cuda_aware.o -o cuda_aware -lcuda
    echo "Built CUDA example cuda_aware"
else
    echo "nvcc not found, skipping CUDA example"
fi
cd ..

echo "Done. Use mpirun -np <N> <program> to run."
