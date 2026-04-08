# 第九章：Kubernetes 上运行 MPI - PyTorchJob 实战

## 本章简介

本章讲解如何在 Kubernetes 环境中运行分布式 MPI 训练任务，使用 PyTorchJob 进行编排。

## 知识点

### 为什么需要 Kubernetes + MPI？
- **弹性扩展**：按需申请和释放计算资源
- **资源隔离**：容器化环境，资源独立
- **自动恢复**：故障自动重试
- **多租户**：支持多用户共享集群

### PyTorchJob 原理
- **Kubeflow**：Kubernetes 上的机器学习工具包
- **PyTorchJob**：Kubeflow 中的 PyTorch 分布式训练 operator
- **Master/Worker**：训练进程角色划分
- **故障恢复**：Worker 失败自动重启

### 架构对比
- **传统模式**：固定机器，手动启动
- **Kubernetes 模式**：声明式配置，自动调度

### 关键配置
- **ReplicaSpec**：指定 Master/Worker 数量
- **GPU 配置**：`nvidia.com/gpu: 1`
- **环境变量**：`WORLD_SIZE`, `RANK`, `MASTER_ADDR`

## 核心概念

### PyTorchJob YAML
```yaml
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "pytorch-distributed-job"
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch:latest
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch:latest
            resources:
              limits:
                nvidia.com/gpu: 1
```

### 训练脚本
```python
import os
import torch.distributed as dist

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
master_addr = os.environ['MASTER_ADDR']

dist.init_process_group(backend='nccl', 
                       init_method='env://',
                       world_size=world_size,
                       rank=rank)
```

## 学习目标

1. 理解 Kubernetes 上运行分布式训练的优势
2. 掌握 PyTorchJob 的配置和使用
3. 学会编写训练脚本和环境变量配置
4. 理解故障恢复机制

## 下一步

学完本章后，进入 [第十章：全链路观测](./ch10-fullstack-observe/README.md) 学习端到端性能观测。
