# 六、完整应用实例：二维Jacobi迭代并行求解

前面我们学了MPI的各个知识点，现在我们来看一个**完整的端到端并行应用**：用Jacobi迭代方法并行求解二维泊松方程。这是PDE（偏微分方程）并行求解中最经典的入门例子，能帮你理解**域分解**和**边界交换**这两个并行计算中最核心的思想。

## 问题描述

我们要求解单位正方形区域 `[0,1] × [0,1]` 上的泊松方程：

```
Δu = 0
```
也就是拉普拉斯方程，边界条件：
- 左、下、上边界：u = 0
- 右边界：u = 1

解析解是线性的：`u(x,y) = x`，我们用Jacobi迭代方法数值求解，看迭代是否收敛到正确解。

## 并行化思路：域分解

对于二维网格问题，最自然的并行化方法是**域分解**：把整个大网格沿着一个维度（这里是y方向）切成多个子网格，每个进程负责计算一个子网格。

比如：整体网格是 `100 × 100`，用 4 个进程：
- 每个进程分到 `100 × 25` 个子网格
- 每个进程自己迭代计算内部网格点
- 每次迭代后，需要和邻居交换**边界层**（ghost cell/halo cell）

### Ghost Cell（鬼边界格）概念

每个进程在自己子网格的上下各多开一层**ghost cell**：
- 这一层数据是邻居进程对应边界的数据
- 每次迭代计算新值之前，先和邻居交换，把对方的最新边界数据拷过来
- 这样每个进程才能正确计算挨着边界的内部点

图示（两个进程，一维划分）：

```
整个网格：
+----+----+
| p0 | p1 |
+----+----+

p0 的本地网格（带 ghost）：
+---+----+
| g | p0 |
+---+----+
   ↑ ghost cell 存 p1 左边界数据

p1 的本地网格（带 ghost）：
+----+---+
| p1 | g |
+----+---+
        ↑ ghost cell 存 p0 右边界数据

每次迭代前交换：
p0 把自己的右边界发给 p1 → p1 放到自己的 ghost 层
p1 把自己的左边界发给 p0 → p0 放到自己的 ghost 层
```

这样每个进程都能拿到邻居的最新边界，计算就正确了。

## 完整代码讲解

完整代码见 [jacobi2d.c](../examples/06-applications/jacobi2d.c)，我们一步步拆解：

### 1. 初始化与网格划分

```c
#define N 100   /* 整体网格大小 */
#define MAX_ITER 1000
#define TOL 1e-6

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 简单一维划分：每个进程分 nlocal 行
    int nlocal = N / size;
    int nx = N;
    int ny = nlocal + 2; /* 上下各加一层 ghost 边界 */
```

- 我们这里做简单的**一维划分**，只沿着 y 方向切，好理解
- `ny = nlocal + 2`：上下各多出来一层放 ghost 边界

### 2. 分配内存并初始化边界

```c
double **u = malloc(ny * sizeof(double*));
double **u_new = malloc(ny * sizeof(double*));
for (int i = 0; i < ny; i++) {
    u[i] = malloc(nx * sizeof(double));
    u_new[i] = malloc(nx * sizeof(double));
    // 初始化全0
    for (int j = 0; j < nx; j++) {
        u[i][j] = 0.0;
        u_new[i][j] = 0.0;
    }
}

// 右边界恒为1（不管哪个进程，所有行的最右点都是1）
for (int i = 0; i < ny; i++) {
    u[i][nx-1] = 1.0;
    u_new[i][nx-1] = 1.0;
}
```

### 3. 确定邻居进程编号

```c
int up = rank - 1;   // 上邻居rank
int down = rank + 1; // 下邻居rank
MPI_Status status;
```

- 如果是 rank 0（最上面的进程），`up` 是负数，表示没有上邻居，不需要交换
- 如果是最后一个 rank（最下面的进程），`down >= size`，表示没有下邻居，不需要交换

### 4. 迭代开始前交换边界

Jacobi 迭代的每一步流程：
1. **交换边界**：把自己的边界发给邻居，从邻居接收对方边界放到 ghost 层
2. **迭代更新**：遍历所有内部点，用周围四个点的平均值更新
3. **计算残差**：统计所有进程的总残差，看是否收敛
4. 如果不收敛，继续下一轮

代码：

```c
double start = MPI_Wtime();
double diff;
int iter;

for (iter = 0; iter < MAX_ITER; iter++) {
    // 第一步：交换ghost边界
    // 发送第一行内部点给上进程，从上进程接收ghost到第0行
    if (up >= 0) {
        MPI_Send(u[1], nx, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
        MPI_Recv(u[0], nx, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &status);
    }
    // 发送最后一行内部点给下进程，从下进程接收ghost到最后一行
    if (down < size) {
        MPI_Send(u[ny-2], nx, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
        MPI_Recv(u[ny-1], nx, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &status);
    }
```

这里收发顺序要注意，避免死锁：
- 如果所有进程都先发给下邻居再收下邻居，会死锁吗？
- 在我们这个一维划分里，不会，因为首尾进程只发一次，所以没问题
- 一般来说，如果你是奇数 rank 先收后发，偶数 rank 先发后收，可以避免死锁，或者用 `MPI_Sendrecv` 更安全

### 5. Jacobi 更新

Jacobi 迭代公式：对于内部点 `(i,j)`，新值是上下左右四个邻居的平均值

```c
// Jacobi迭代
diff = 0.0;
for (int i = 1; i <= ny-2; i++) {
    for (int j = 1; j < nx-1; j++) {
        u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
        diff += (u_new[i][j] - u[i][j]) * (u_new[i][j] - u[i][j]);
    }
}

// 交换 u 和 u_new 指针，不用拷数据
double **tmp = u;
u = u_new;
u_new = tmp;
```

- 只更新内部点（`1 <= i <= ny-2`），ghost 层不更新
- `diff` 是当前进程所有点的误差平方和，后面需要全局归约

### 6. 全局收敛检查

我们需要把所有进程的 `diff` 加起来，开根号，看是不是小于阈值：

```c
// 全局归约求和diff
double global_diff;
MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
global_diff = sqrt(global_diff);

if (rank == 0 && (iter % 100 == 0)) {
    printf("Iter %d: diff = %e\n", iter, global_diff);
}

if (global_diff < TOL) {
    if (rank == 0) {
        printf("Converged after %d iterations, diff = %e < %e\n", iter+1, global_diff, TOL);
    }
    break;
}
```

- `MPI_Allreduce` 是**集合通信**，所有进程参与，每个进程都得到最终求和结果
- 这里用 `MPI_SUM` 做加法，把所有进程的 `diff` 加起来

### 7. 结束计时，清理

```c
double end = MPI_Wtime();
if (rank == 0) {
    printf("Total time: %.4f seconds\n", end - start);
}

// 释放内存
for (int i = 0; i < ny; i++) {
    free(u[i]);
    free(u_new[i]);
}
free(u);
free(u_new);

MPI_Finalize();
```

## 编译和运行

### 编译

因为已经有 `build_examples.sh`，直接跑：

```bash
cd examples
./build_examples.sh
```

手动编译：

```bash
cd examples/06-applications
mpicc -O2 -o jacobi2d jacobi2d.c -lm
```

需要链接 math library (`-lm`)，因为我们用了 `sqrt`。

### 运行

```bash
# 4个进程运行
mpirun -np 4 ./jacobi2d
```

输出类似：

```
Iter 0: diff = 1.665000e-01
Iter 100: diff = 2.118125e-03
Iter 200: diff = 5.810311e-04
Iter 300: diff = 2.620752e-04
...
Converged after 321 iterations, diff = 9.87231e-07 < 1.00000e-06
Total time: 0.1234 seconds
```

## 可以改进的地方

这个例子为了好理解做了简化，实际生产中可以进一步优化：

### 1. 二维划分

我们这里是一维划分（只切y方向），实际二维网格可以做**二维划分**（切x和y两个方向），每个进程分到一个更小的矩形子块：
- 好处：每个进程需要交换的边界数据量是 `O(√(N^2/P)) = O(N/√P)`，比一维划分的 `O(N/P × N) = O(N^2/P)` 更小
- 扩展性更好，进程多的时候更划算

### 2. 用非阻塞通信重叠计算和通信

我们这里是交换完边界再计算，可以用**非阻塞通信**：
1. 立刻发起发送接收请求（不等待完成）
2. 先计算不需要边界的内部点
3. 等内部点算完了，再等待通信完成，计算挨着边界的点
这样**计算和通信重叠**，能掩盖通信延迟，提升速度。

### 3. 用更快的解法

Jacobi 迭代简单但收敛慢，实际用：
- Gauss-Seidel 迭代 + 红黑排序
- 多重网格方法
- Krylov 子空间方法（共轭梯度）
这些方法收敛快很多，但并行化思路（域分解 + 边界交换）是一样的。

## 关键要点总结

| 概念 | 说明 |
|------|------|
| **域分解** | 把大问题切成小块，每个进程一块，这是并行计算最基本的思路 |
| **Ghost Cell / Halo 交换** | 相邻子块需要交换边界数据，这是域分解必须做的一步 |
| **MPI_Allreduce** | 全局归约，这里用来聚合残差，判断收敛，这是迭代方法常用的 |
| **指针交换** | `double **tmp = u; u = u_new; u_new = tmp;` 不用拷贝数据，高效 |

## 思考题

1. 如果我们用二维划分，每个进程需要交换几个方向的边界？
2. 为什么说一维划分在进程多的时候通信量比二维划分大？
3. 我们这里用的是阻塞发送接收，会死锁吗？为什么？
4. 如果要改成用 `MPI_Sendrecv` 交换边界，代码应该怎么改？

## 示例代码

- [jacobi2d.c](../examples/06-applications/jacobi2d.c) - 二维Jacobi迭代并行求解完整代码

## 下一步

→ 下一章：[实现环境与调试优化](07-optimize.md)
