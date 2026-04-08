# 第七章：环境与调试优化

## 本章简介

本章讲解 MPI 开发环境的搭建、程序编译运行方法，以及调试和性能优化技巧。

## MPI 环境安装

### 常用 MPI 实现

| 实现 | 安装方式 | 特点 |
|------|----------|------|
| **OpenMPI** | `apt install openmpi` | 默认首选 |
| **MPICH** | `apt install mpich` | 轻量高性能 |
| **Intel MPI** | Intel 官网 | Intel 硬件优化 |

### 环境验证

```bash
# 查看版本
mpirun --version
mpicc --version

# 简单测试
echo "Hello" | mpirun -np 2 hostname

# 检查 OpenMPI 配置
ompi_info
```

## 编译与运行

### 编译器包装器

| 语言 | 编译器 | 典型用法 |
|------|--------|----------|
| C | `mpicc` | `mpicc program.c -o program` |
| C++ | `mpicxx` | `mpicxx program.cpp -o program` |
| Fortran | `mpifort` | `mpifort program.f90 -o program` |

### 常用编译选项

```bash
# 基本编译
mpicc -o hello hello.c

# 带优化
mpicc -O3 -o hello hello.c

# 带调试信息
mpicc -g -o hello hello.c

# 链接数学库
mpicc -lm -o program program.c

# CUDA 程序
mpicxx -o cuda_program cuda_program.cu -lcudart -L/usr/local/cuda/lib64
```

### 运行命令

```bash
# 本地运行（2 个进程）
mpirun -np 2 ./program
mpiexec -np 2 ./program

# 指定主机
mpirun -np 4 -H server1,server2 ./program

# 主机文件
mpirun -np 8 -hostfile hosts.txt ./program

# 绑定 CPU 核心
mpirun -np 4 --bind-to core ./program

# 指定 GPU
mpirun -np 2 -x CUDA_VISIBLE_DEVICES=0,1 ./program
```

### hostfile 格式

```
# hosts.txt
server1 slots=2
server2 slots=2
server3 slots=4
```

## 调试方法

### 1. 打印调试

最简单的方法：

```c
if (rank == 0) {
    printf("Debug: value = %d\n", value);
}
MPI_Barrier(MPI_COMM_WORLD);  // 同步，避免输出乱序
```

### 2. 错误检查

```c
int err;
err = MPI_Send(buf, count, MPI_INT, dest, tag, comm);
if (err != MPI_SUCCESS) {
    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    MPI_Error_string(err, error_string, &length);
    fprintf(stderr, "Error: %s\n", error_string);
}
```

### 3. 验证通信

```c
// 简单验证：所有进程都正确初始化
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

printf("Rank %d of %d\n", rank, size);
MPI_Barrier(MPI_COMM_WORLD);  // 确保按顺序输出
```

### 4. 使用调试器

```bash
# GDB 调试（需要先安装）
mpirun -np 2 xterm -e gdb ./program

# TotalView（商业调试器）
totalview mpirun -a -np 2 ./program
```

## 性能优化

### 通信开销分析

**Amdahl 定律**：

```
加速比 = 1 / (S + P/N)

S = 串行比例
P = 并行比例
N = 进程数
```

### 优化策略

#### 1. 减少通信次数

```c
// ❌ 多次通信
for (i = 0; i < n; i++)
    MPI_Send(data[i], ..., dest, ...);

// ✅ 一次通信
MPI_Send(data, n*sizeof(item), ..., dest, ...);
```

#### 2. 使用非阻塞通信

```c
// 通信与计算重叠
MPI_Request req;
MPI_Isend(send_buf, count, MPI_FLOAT, dest, tag, comm, &req);

// 计算
do_computation();

// 等待完成
MPI_Wait(&req, &status);
```

#### 3. 集合通信优化

```c
// 使用带缓冲的广播（更高效）
MPI_Bcast(buffer, count, MPI_DATATYPE, root, comm);
// 等价于
MPI_Bcast(buffer, count, MPI_DATATYPE, root, comm);
```

#### 4. 拓扑感知

```bash
# 让 MPI 感知网络拓扑
mpirun --map-by node ./program
```

### 性能工具

| 工具 | 用途 |
|------|------|
| `vtune` | Intel 性能分析 |
| `Score-P` | MPI 性能分析 |
| `TAU` | 多语言性能分析 |
| `IPM` | MPI 统计信息 |

```bash
# 使用 IPM
mpirun -np 4 -ipm ./program

# 查看统计
ipm_parse -html log.xml
```

## 常见问题与解决

### 死锁

**症状**：程序卡住不动

**原因**：
- 发送接收顺序不匹配
- 所有进程都在等待

**解决**：
- 使用 `MPI_Sendrecv`
- 检查发送接收匹配

### 内存泄漏

**症状**：内存持续增长

**解决**：
- 检查 `MPI_Type_free`
- 检查 `MPI_Comm_free`

### 通信错误

**症状**：
```
MPI_ERR_TRUNCATE: message truncated
```

**原因**：接收缓冲区太小

**解决**：增大接收缓冲区

## 本章测验

- [ ] 掌握 MPI 程序编译运行
- [ ] 学会基本调试技巧
- [ ] 理解性能优化策略
- [ ] 能排查常见问题

## 下一步

学完本章后，进入 [第八章：RDMA Verbs 编程](./ch08-rdma-verbs/README.md) 学习 RDMA 原生编程。
