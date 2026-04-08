# 第十章：全链路观测实战

## 本章简介

本章是全栈实战内容，学习如何从 PyTorch 到 NCCL 再到 RDMA 进行端到端的性能观测和优化。

## 全链路架构

### 数据流层次

```
┌────────────────────────────────────────────────────────┐
│  应用层：PyTorch DDP                                   │
│    • 前向传播                                           │
│    • 反向传播 (梯度计算)                                │
├────────────────────────────────────────────────────────┤
│  通信层：NCCL                                           │
│    • AllReduce (梯度同步)                              │
│    • Broadcast (参数同步)                               │
│    • Point-to-point                                    │
├────────────────────────────────────────────────────────┤
│  传输层：OpenMPI + libfabric                           │
│    • RDMA 操作 (Put/Get)                               │
│    • 消息队列管理                                       │
├────────────────────────────────────────────────────────┤
│  物理层：RDMA 网卡 (InfiniBand / RoCE)                 │
│    • 数据传输                                           │
│    • 完成通知                                           │
└────────────────────────────────────────────────────────┘
```

### 训练迭代过程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Forward     │───→│ Backward    │───→│ AllReduce   │
│ (计算输出)   │    │ (计算梯度)   │    │ (同步梯度)   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ↓
更新参数 ←─────────────── 参数梯度 ←──────────┘
```

## 各层观测工具

### PyTorch 层

#### 1. PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

#### 2. DDP 钩子

```python
# 测量梯度同步时间
def hook_fn(module, grad_input, grad_output):
    print(f"Gradient sync took {time.time() - start:.4f}s")

for param in model.parameters():
    param.register_hook(hook_fn)
```

### NCCL 层

#### NCCL Debug 环境变量

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 运行程序
mpirun -np 4 python train.py
```

输出示例：
```
NCCL INFO Channel 0 : 0[1c3c0] -> 1[1c4c1] [Peer 2-3] via NET/IB/0
NCCL INFO Channel 1 : 0[1c3c0] -> 1[1c4c1] [Peer 2-3] via NET/IB/0
NCCL INFO AllReduce : 0 : 1/1/1 [0.506/0.506/0.506] | 1.245 MB/s | 256.000 us
```

#### NCCL Tests

```bash
# 编译 NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make

# 带宽测试
mpirun -np 2 ./build/all_reduce_perf -b 1M -e 1G -f 2
```

### RDMA 层

#### 1. 性能统计

```bash
# 查看网卡统计
perfquery

# 详细统计
perfquery -L
```

#### 2. 带宽测试

```bash
# RDMA 写入带宽
ib_send_bw -d mlx5_0 -a

# RDMA 读取带宽  
ib_send_lat -d mlx5_0
```

#### 3. 网卡状态

```bash
# 查看 IB 设备
ibstat

# 查看端口信息
ibportstate <lid> <port> query
```

## RDMA vs TCP 对比

### 性能差异

| 指标 | TCP | RDMA | 提升 |
|------|-----|------|------|
| 延迟 | 10-20 μs | 1-2 μs | 10x |
| 带宽 | 25 Gbps | 100+ Gbps | 4x+ |
| CPU 开销 | 高 | 极低 | - |

### 测试脚本

```bash
# 对比测试
./rdma_tcp_test.sh
```

输出示例：
```
=== RDMA Bandwidth ===
Send Bandwidth: 95.2 GB/s

=== TCP Bandwidth ===
Send Bandwidth: 3.2 GB/s

=== Speedup ===
RDMA is 29.8x faster than TCP
```

## 性能瓶颈定位

### 常见瓶颈

| 瓶颈 | 症状 | 定位工具 |
|------|------|----------|
| **计算瓶颈** | GPU 利用率低 | nvidia-smi, Profiler |
| **通信瓶颈** | 通信时间占比高 | NCCL Debug |
| **网络瓶颈** | 带宽受限/丢包 | perfquery |
| **同步瓶颈** | 等待时间长 | 打印时间戳 |

### 定位流程

1. **GPU 利用率检查**
   ```bash
   nvidia-smi -l 1
   ```

2. **NCCL 通信检查**
   ```bash
   NCCL_DEBUG=INFO python train.py 2>&1 | grep -E "AllReduce|Broadcast"
   ```

3. **网络检查**
   ```bash
   perfquery
   ```

4. **热点分析**
   ```bash
   # 使用 PyTorch Profiler
   torch.profiler.schedule(...)
   ```

## 优化建议

### 通信优化

1. **减少通信频率**：梯度累积
2. **压缩通信**：梯度量化
3. **异步通信**：计算与通信重叠

### 网络优化

1. **启用 RDMA**：使用 InfiniBand/RoCE
2. **调整 MTU**：使用 4096
3. **中断合并**：调整网卡参数

### 计算优化

1. **混合精度**：FP16/BF16
2. **CUDA 优化**：融合算子
3. **数据加载**：多进程 DataLoader

## 示例代码

本章配套示例在 `10-fullstack-observe/` 目录：

```bash
cd ch10-fullstack-observe/10-fullstack-observe

# NCCL 观测
python observe_nccl.py

# RDMA/TCP 对比
python benchmark_compare.py
```

- `observe_nccl.py` - NCCL 通信观测
- `benchmark_compare.py` - RDMA vs TCP 对比

## 本章测验

- [ ] 理解全链路架构
- [ ] 掌握各层观测工具
- [ ] 能对比 RDMA 和 TCP 性能
- [ ] 学会定位性能瓶颈

## 总结

恭喜完成 MPI 完整教程！🎉

你已掌握：
- ✅ MPI 基础概念和编程
- ✅ 点对点和集合通信
- ✅ GPU/RDMA 高级特性
- ✅ NCCL 与 PyTorch 结合
- ✅ Kubernetes 部署
- ✅ 端到端性能优化

继续加油，在高性能计算的路上前进！🚀
