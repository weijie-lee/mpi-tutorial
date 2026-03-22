# 六、实现、环境与调试优化

## 1. 主流 MPI 实现对新硬件的支持

| 实现 | 特点 | CUDA/GPU | RDMA | 适用场景 |
|------|------|----------|------|----------|
| **OpenMPI** | 开源，社区活跃，功能全面 | ✅ 原生支持，编译时开启 | ✅ 支持verbs | 通用开发、研究、生产环境都适用 |
| **Intel MPI** | Intel推出，针对Intel硬件优化 | ✅ | ✅ 对Intel Omni-Path优化 | 至强+Intel GPU集群 |
| **MPICH** | 经典开源实现，很多商用版本基于它 | ✅ | ✅ | 学术研究，二次开发 |
| **MVAPICH** | 专门针对InfiniBand/GPU优化 | ✅ | ✅ 极致优化 | IB/RoCE 多GPU超算环境 |

对于大多数用户，**OpenMPI 4.x** 是最好的起点，文档全，社区活跃，开箱即用。

## 2. 编译运行基本方法

### 编译器包装器
MPI 提供了包装好的编译器命令，自动帮你链接正确的库：
- C: `mpicc`
- C++: `mpicxx` / `mpic++`
- Fortran: `mpif90`

基本用法和原生gcc一样：
```bash
mpicc -o hello hello.c -O2
```

如果需要启用GPU/RDMA支持，通常是在编译 MPI 本身的时候配置好，`mpicc` 不用加额外参数。

### 运行：`mpirun` / `mpiexec`
常用参数：
- `-np N`：启动 N 个进程
- `-hostfile hosts`：指定哪些节点运行，一行一个节点
- `--bind-to core`：把进程绑定到核心，提升性能
- `-npernode 4`：每个节点启动4个进程（多节点多GPU常用）

示例：
```bash
# 4个进程在本地运行
mpirun -np 4 ./hello

# 两个节点，node1跑4个，node2跑4个
mpirun -np 8 -H node1:4,node2:4 ./myapp
```

### 集群结合 Slurm 调度
在大多数超算上，用 Slurm 调度，两种方式都可以：
1. **srun 直接启动**（推荐，Slurm 集成更好）：
```bash
srun -N 2 -n 8 ./myapp
```
2. **srun 包装 mpirun**：
```bash
srun -N 2 -n 8 mpirun ./myapp
```

OpenMPI 和 Intel MPI 都原生支持 Slurm，会自动获取节点信息。

## 3. 调试与性能分析

### 基础调试：打印
最简单的方法就是按 rank 过滤打印：
```c
if (rank == 0) {
    printf("Only rank 0 prints this, avoid clutter\n");
}
```
可以把输出重定向到文件，每个rank单独一个文件：
```bash
mpirun -np 4 ./app > output-rank%02d.txt
```

### 并行调试工具
- **TotalView**：商业并行调试器，支持MPI，断点调试非常方便
- **Allinea DDT**：另一个商用主流并行调试器
- **gdb + MPI**：可以调试，但需要 attaching 到各个进程，比较麻烦

用法示例（OpenMPI + gdb）：
```bash
mpirun -np 2 xterm -e gdb ./myapp
```
每个进程会弹出一个xterm窗口，可以分别调试。

### 性能分析工具
- **Intel VTune**：热点分析，通信瓶颈分析
- **NVIDIA Nsight Systems**：分析CPU/GPU/通信时间线，特别适合GPU应用
- **Score-P / Vampir**：开源MPI性能分析，可视化通信行为

### 针对 GPU+RDMA 场景定位瓶颈
1. 用 Nsight Systems 看时间线：是计算占比高还是通信占比高？
2. 通信占比高的话：检查是不是通信太多太小，可以聚合消息
3. GPU 利用率低：看看是不是 GPU 经常在等通信，能不能把计算通信重叠

## 4. 性能优化要点

### 针对 CPU 优化
1. **减少通信次数，合并消息**：发 1 次 1MB 比发 1000 次 1KB 快很多
2. **负载均衡**：尽量让每个进程的计算量差不多，避免快等慢
3. **利用拓扑亲和性**：进程绑定到 CPU 核心/NUMA 节点，减少跨跳
4. **优先用集合通信**：MPI 集合通信比手写点对点更快

### 针对 GPU 优化
1. **尽量用 GPU-aware MPI**：避免显式设备主机拷贝，节省PCIe带宽
2. **让计算和通信重叠**：用非阻塞通信，在等待通信的时候做计算
3. **小消息聚合**：多次小通信合并成一次大通信发送

### 针对 RDMA 优化
1. **使用注册过的内存**：RDMA 需要内存注册，重复使用注册好的缓冲区避免重复注册开销
2. **让 MPI 用 RDMA**：编译的时候确保启用了 verbs 支持，运行时不要强制用TCP
3. **尽量用 GPU 直接通信 + RDMA**：GPU数据直接走RDMA，不需要CPU中转，性能最好

## 编译测试脚本示例

项目根目录提供了 [`build_examples.sh`](../examples/build_examples.sh) 一键编译所有C示例：

```bash
cd examples
chmod +x build_examples.sh
./build_examples.sh
```

## 下一步

→ 下一章：[RDMA Verbs 编程入门](08-rdma-verbs.md)
