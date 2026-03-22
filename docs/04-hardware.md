# 四、硬件结合：GPU 与 RDMA 支持

现代大规模并行计算（尤其是深度学习训练）都是多节点多GPU架构，MPI 对 GPU 和 RDMA 的原生支持非常关键。

## 1. MPI 与 GPU

### 为什么需要 GPU-aware MPI？

如果没有 GPU-aware MPI，要发送GPU上的数据必须这么做：
```c
// 不支持 GPU-aware 时代码：需要两次拷贝
cudaMemcpy(cpu_buf, gpu_buf, size, cudaMemcpyDeviceToHost);
MPI_Send(cpu_buf, size, MPI_BYTE, dest, tag, comm);
```
- 把数据从GPU拷贝到CPU
- 再让MPI发送CPU数据
- 接收方反过来：MPI收 → 拷贝到GPU

这额外的两次拷贝非常浪费时间，还浪费PCIe带宽。

**GPU-aware MPI 可以直接发送设备端内存**，不需要CPU拷贝：
```c
// GPU-aware MPI：直接发设备指针
MPI_Send(gpu_buf, size, MPI_BYTE, dest, tag, comm);
```

### 主流实现支持

| MPI 实现 | CUDA 支持 | ROCm 支持 | 启用方式 |
|----------|-----------|-----------|----------|
| OpenMPI | ✅ 原生支持 | ✅ | `--with-cuda=/usr/local/cuda` 编译时指定 |
| Intel MPI | ✅ | ❌ | 环境变量 `I_MPI_CUDA=1` |
| MPICH | ✅ | ✅ | `--enable-g=yes --with-cuda=...` |
| MVAPICH | ✅ | ✅ | 原生带GPU支持 |

### CUDA-aware MPI 示例

```cpp
// examples/04-hardware/cuda_aware.cu
#include <mpi.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 在GPU上分配内存
    float *d_buf;
    cudaMalloc(&d_buf, 1024 * sizeof(float));

    if (rank == 0) {
        // 初始化数据...
        // 直接发送设备指针！不需要拷贝到主机
        MPI_Send(d_buf, 1024, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(d_buf, 1024, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cudaFree(d_buf);
    MPI_Finalize();
    return 0;
}
```

编译用 `nvcc` 结合 `mpic++`:
```bash
nvcc -c cuda_aware.cu -o cuda_aware.o
mpic++ cuda_aware.o -o cuda_aware -lcuda
```

运行测试：
```bash
# 需要至少两个进程
mpirun -np 2 ./cuda_aware
```

如果运行成功没有报错，说明你的 MPI 正确支持 CUDA-aware。

### 验证 CUDA-aware 是否工作

可以通过一个简单的测试来验证：
1. 数据分配在GPU上
2. 直接用MPI发送接收
3. 接收方在GPU上读取结果验证正确性

完整代码见 [cuda_aware.cu](../examples/04-hardware/cuda_aware.cu)，示例中 rank 0 初始化GPU数据，发送给 rank 1，rank 1 拷贝回CPU验证结果。

## 2. MPI 与 RDMA

### 什么是 RDMA

**RDMA** = **Remote Direct Memory Access**，远程直接内存访问：
- 传统 TCP/IP 通信：数据要经过内核缓冲区 → 用户空间，多次内存拷贝
- RDMA：网卡硬件直接绕过内核，直接读写远程机器的内存
- **核心优势**：更低延迟、更高带宽、更低CPU占用

### RDMA 硬件类型

| 类型 | 说明 | 应用场景 |
|------|------|----------|
| **InfiniBand (IB)** | 专用高性能网络，传统HPC主流 | 超级计算机、AI训练集群 |
| **RoCE v2** | RDMA over Converged Ethernet，以太网上跑RDMA | 现代数据中心、云厂商AI集群 |
| **iWARP** | 标准以太网RDMA | 广域RDMA场景 |

### MPI 如何使用 RDMA

主流 MPI 都支持通过 `libverbs` 接口使用 RDMA：

| MPI 实现 | RDMA 支持 | 启用方式 |
|----------|-----------|----------|
| OpenMPI | ✅ | 编译时 `--enable-verbs`，运行时自动检测 |
| MVAPICH | ✅ | 原生针对InfiniBand/RDMA优化，默认启用 |
| Intel MPI | ✅ | 自动检测RDMA设备，默认启用 |
| MPICH | ✅ | 需要编译时启用verbs支持 |

只要MPI库编译时支持了RDMA，**不需要修改你的应用代码**，MPI会自动选择最优的通信路径。

### 如何检查 MPI 是否支持 RDMA

```bash
# OpenMPI 检查
ompi_info --parsable | grep -i verbs
# 如果输出有 MCA verbs: available = 1 就是支持

# 检查可用的 BTL (Byte Transfer Layer)
ompi_info --parsable | grep btl:
# 看到 `openib` 或 `verbs` 说明支持RDMA
```

### 强制使用 RDMA 网络

OpenMPI 可以通过参数指定使用 IB/RDMA 接口：

```bash
# 只使用 openib (IB/RDMA) 模块
mpirun --mca btl openib,self -np 8 ./your_app

# 优先使用 openib，不行再用 tcp
mpirun --mca btl ^tcp -np 8 ./your_app
```

### RDMA 性能对比

| 指标 | TCP/IP (Ethernet) | RDMA (InfiniBand/RoCE) |
|------|-------------------|-------------------------|
| 单消息延迟 | ~20-50 微秒 | **~3-10 微秒** |
| 峰值带宽利用率 | ~70-80% | **~95%+** |
| CPU 占用（通信时） | 高，内核处理消耗 | 低，网卡硬件处理 |
| 多节点扩展性 | 一般，CPU瓶颈明显 | 好，适合大规模训练 |

在多节点多GPU训练中，RDMA 能显著减少通信瓶颈，提升整体训练吞吐量。

## 3. 常见问题排查

### Q: 我的MPI编译时没有CUDA支持怎么办？

**A:** 需要重新编译MPI，编译时指定CUDA路径。以OpenMPI为例：
```bash
./configure --with-cuda=/usr/local/cuda ...
make && make install
```

或者使用你的包管理器安装预编译的支持CUDA的MPI版本（比如`openmpi-cuda`等）。

### Q: 运行CUDA-aware程序报错了怎么办？

常见错误和解决：

1. **"CUDA device pointer not recognized"**  
   → 说明你的MPI不支持CUDA-aware，重新编译MPI或者换支持的版本。

2. **"cudaMemcpy failed after MPI communication"**  
   → 检查是否正确绑定GPU，每个进程绑定到不同的GPU：`cudaSetDevice(rank % n_gpus)`

3. **运行结果不对，数据传错了**  
   → 检查数据类型和计数：`MPI_FLOAT`对应`float*`，`MPI_DOUBLE`对应`double*`，不要错配。

### Q: 系统有RDMA设备但MPI不用怎么办？

检查：
1. MPI是否编译时支持verbs（参考上文检查方法）
2. 是否有防火墙/ACL限制了RDMA端口
3. 尝试强制指定BTL：`--mca btl openib,self`

## 示例代码

### RDMA 是什么

**RDMA** = **Remote Direct Memory Access**，远程直接内存访问：
- 传统 TCP/IP 通信：数据要经过内核 -> 用户拷贝，多次复制
- RDMA：网卡直接绕过内核，直接读写远程机器内存
- **优势**：更低延迟（几微秒 vs 几十微秒）、更高带宽、更低CPU占用

### RDMA 硬件
- **InfiniBand (IB)**：传统高性能计算主流，专用网络
- **RoCE v2**：RDMA over Converged Ethernet，以太网上跑RDMA，现在数据中心主流
- **iWARP**：另一种以太网上的RDMA标准

### MPI 对 RDMA 支持
主流 MPI 都支持通过 `libverbs` 接口用RDMA：
- OpenMPI：`--enable-verbs` 编译，运行时会自动选RDMA可用的端口
- MVAPICH：专门针对InfiniBand优化，原生就是RDMA优先
- Intel MPI：自动检测RDMA设备，默认启用

启用后，MPI会自动用RDMA做节点间通信，不需要改程序代码。

### 性能对比
| 指标 | TCP/IP | RDMA |
|------|--------|------|
| 单消息延迟 ~ | 20-50 μs | 3-10 μs |
| 带宽利用率 | ~70-80% | ~95%+ |
| CPU 占用 | 高（内核处理） | 低（网卡处理） |

在多节点多GPU训练中，RDMA 能显著减少通信瓶颈，提升整体训练吞吐量。

## 如何检查你的MPI是否支持GPU/RDMA

检查GPU支持：
```bash
ompi_info --parsable | grep cuda
# 如果有 cuda:support = 1 就是支持
```

检查RDMA支持：
```bash
ompi_info --parsable | grep verbs
# 有verbs支持就说明支持RDMA
```

## 示例代码

- [cuda_aware.cu](../examples/04-hardware/cuda_aware.cu) - CUDA-aware MPI 基础示例

## 下一步

→ 下一章：[上层栈结合：NCCL 与 PyTorch](05-stack.md)
