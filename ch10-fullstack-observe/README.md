# 第十章：全链路观测实战

## 本章简介

本章是全栈实战内容，学习如何从 PyTorch 到 NCCL 再到 RDMA 进行端到端的性能观测和优化。

## 知识点

### 全链路架构
```
应用层：PyTorch DDP
    ↓
通信层：NCCL (GPU间集合通信)
    ↓
传输层：RDMA / TCP
    ↓
物理层：InfiniBand / RoCE 网卡
```

### 各层观测工具

#### PyTorch 层
- **DDP 钩子**：梯度同步时间、训练 step 时间
- **TorchProfiler**：PyTorch Profiler 分析
- **日志**：打印 rank、batch size、forward/backward 时间

#### NCCL 层
- **NCCL DEBUG**：环境变量 `NCCL_DEBUG=INFO`
- **ncclGraph**：NCCL 通信图分析
- **NCCL Tests**：基准测试工具

#### RDMA 层
- **perfstat**：网卡统计（带宽、延迟、错误）
- **ibstat**：IB 设备状态
- **ib_send_bw / ib_recv_bw**：RDMA 带宽测试
- **rdma_cm**：连接状态

### RDMA vs TCP 对比
- **延迟**：RDMA ~1-2μs，TCP ~10-20μs
- **带宽**：RDMA 可达 100+ Gbps
- **CPU 占用**：RDMA 旁路内核，CPU 开销更低

### 性能瓶颈定位
1. **计算瓶颈**：GPU 利用率低
2. **通信瓶颈**：通信时间占比高
3. **网络瓶颈**：带宽受限或丢包
4. **同步瓶颈**：集合通信等待

## 核心概念

### 环境变量配置
```bash
# NCCL 调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# GPU 设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# RDMA 配置
export RDMA_VERBS=1
```

### 观测脚本
```bash
# 网卡统计
perfquery

# RDMA 带宽测试
ib_send_bw -d mlx5_0 -a

# NCCL 基准测试
./build/all_reduce_perf -b 1M -e 1G -f 2
```

## 学习目标

1. 理解分布式训练的全链路架构
2. 掌握各层观测工具的使用
3. 能够对比 RDMA 和 TCP 性能
4. 学会定位性能瓶颈

## 示例代码

本章配套示例在 `10-fullstack-observe/` 目录：
- `rdma_tcp_test.sh` - RDMA/TCP 对比测试脚本
- `observe_nccl.py` - NCCL 观测脚本

## 总结

恭喜完成 MPI 完整教程！学完本章后，你已经掌握了：
- MPI 基础概念和编程
- GPU/RDMA 高级特性
- NCCL 与 PyTorch 结合
- Kubernetes 部署
- 端到端性能优化

继续加油 🚀
