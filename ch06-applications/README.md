# 第六章：完整应用实例 - 二维 Jacobi 迭代

## 本章简介

本章通过一个完整的科学计算应用——二维 Jacobi 迭代求解泊松方程，来综合运用前面学到的 MPI 编程技能。

## 问题背景

### 什么是泊松方程？

泊松方程是数学物理中最重要的方程之一：

```
∂²u/∂x² + ∂²u/∂y² = f(x,y)
```

它描述了：
- 热传导：温度分布
- 流体动力学：压力场
- 静电学：电势分布

### 离散化

在网格上近似求解：

```
       u(i,j+1)
          ↑
u(i-1,j) ← u(i,j) → u(i+1,j)
          ↓
       u(i,j-1)

u_new[i,j] = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + h²*f[i,j]) / 4
```

## 域分解

### 为什么要分解？

如果网格是 10000×10000 = 1 亿个点，单核计算太慢。需要**分而治之**。

### 水平分解示例

```
原始网格 (12x8):
+---+---+---+---+---+---+---+---+---+---+---+
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |10|11|
+---+---+---+---+---+---+---+---+---+---+---+

分解为 4 个进程:

P0 (行 0-2):  | 0 | 1 | 2 |...|  |  |  |
P1 (行 3-5):  |  |  |  |...|  |  |  |
P2 (行 6-8):  |  |  |  |...|  |  |  |
P3 (行 9-11): |  |  |  |...|  |11|  |
```

### Ghost Cell（鬼边界）

每个子区域需要**邻居的数据**才能计算：

```
P0 区域:        需要 P1 的数据
        ┌─────────────────────┐
        │  x  x  x  x  x  x  │  ← 内部点
        │  x  ○  ○  ○  ○  x  │  ← 内部点
        │  x  ○  ○  ○  ○  x  │  ← 内部点
        └─────────────────────┘
                        ↑
                   鬼边界 (来自 P1)
```

## 并行实现

### 数据结构

```c
// 每个进程持有的数据
float** u;        // 当前解
float** u_new;    // 下一轮解
float** f;        // 右端项
int local_nrows;  // 本地行数（含鬼边界）
int local_ncols;  // 列数
```

### 通信模式

```c
// 发送上边界到上方进程
if (myrow > 0) {
    MPI_Send(local_u[1], ncols, MPI_FLOAT, 
             myrank - ncols, TAG_UP, MPI_COMM_WORLD);
    MPI_Recv(local_u[0], ncols, MPI_FLOAT,
             myrank - ncols, TAG_DOWN, MPI_COMM_WORLD, &status);
}

// 发送下边界到下方进程
if (myrow < nprocs_row - 1) {
    MPI_Send(local_u[local_nrows-2], ncols, MPI_FLOAT,
             myrank + ncols, TAG_DOWN, MPI_COMM_WORLD);
    MPI_Recv(local_u[local_nrows-1], ncols, MPI_FLOAT,
             myrank + ncols, TAG_UP, MPI_COMM_WORLD, &status);
}
```

### 迭代循环

```c
for (iter = 0; iter < max_iter; iter++) {
    // 1. 交换鬼边界
    exchange_halo(u, ...);
    
    // 2. 本地计算
    for (i = 1; i < local_nrows-1; i++)
        for (j = 1; j < ncols-1; j++)
            u_new[i][j] = (u[i-1][j] + u[i+1][j] + 
                          u[i][j-1] + u[i][j+1] + h*h*f[i][j]) * 0.25;
    
    // 3. 收敛判断（全局归约）
    diff = compute_max_diff(u, u_new);
    MPI_Allreduce(&diff, &global_diff, 1, MPI_FLOAT, MPI_MAX, ...);
    
    if (global_diff < tolerance) break;
    
    // 4. 交换新旧解
    swap(&u, &u_new);
}
```

### 收敛判断

```c
// 计算局部最大差值
float local_max = 0.0;
for (i = 1; i < local_nrows-1; i++)
    for (j = 1; j < ncols-1; j++)
        local_max = fmax(local_max, fabs(u_new[i][j] - u[i][j]));

// 全局最大差值（所有进程都要知道）
MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, 
              MPI_MAX, MPI_COMM_WORLD);
```

## 性能优化

### 通信优化

1. **减少通信次数**：计算与通信重叠
2. **非阻塞通信**：提前开始下一轮通信
3. **合并消息**：一次性发送多行

### 计算优化

1. **内存连续访问**：提高缓存命中率
2. **向量化**：利用 SIMD 指令
3. **减少内存分配**：预分配缓冲区

## 示例代码

本章配套示例在 `06-applications/` 目录：

```bash
cd ch06-applications/06-applications
mpicc -o jacobi2d jacobi2d.c -lm
mpirun -np 4 ./jacobi2d 100 1000
```

参数说明：
- `100` - 网格大小
- `1000` - 最大迭代次数

## 本章测验

- [ ] 理解域分解思想
- [ ] 掌握鬼边界通信模式
- [ ] 实现完整的并行 Jacobi 迭代
- [ ] 理解收敛判断机制

## 下一步

学完本章后，进入 [第七章：调试与优化](./ch07-optimize/README.md) 学习环境搭建和性能调优。
