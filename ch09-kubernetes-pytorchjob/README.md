# 第九章：Kubernetes 上运行 MPI - PyTorchJob 实战

## 本章简介

本章讲解如何在 Kubernetes 环境中运行分布式 MPI 训练任务，使用 PyTorchJob 进行编排。

## 为什么需要 Kubernetes？

### 传统方式的问题

- **手动管理机器**：需要维护机器列表
- **资源利用率低**：机器可能闲置
- **故障处理麻烦**：需要手动重启
- **扩展困难**：增加机器需要改配置

### Kubernetes 优势

| 特性 | 传统方式 | Kubernetes |
|------|----------|------------|
| 资源管理 | 手动 | 自动 |
| 故障恢复 | 手动 | 自动 |
| 扩展性 | 困难 | 简单 |
| 调度 | 手动 | 自动 |

## 基本概念

### Kubernetes (K8s)

容器编排系统：
- **Pod**：最小调度单元（一个或多个容器）
- **Node**：工作节点（服务器）
- **Master**：控制节点
- **Service**：服务发现和负载均衡

### Kubeflow

机器学习工具包，包含：
- **PyTorchJob**：PyTorch 分布式训练 operator
- **TFJob**：TensorFlow 训练 operator
- **Katib**：超参数调优

### PyTorchJob

Kubeflow 中的 PyTorch 分布式训练 CRD：

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
            image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
            resources:
              limits:
                nvidia.com/gpu: 1
```

## 核心机制

### 角色分配

| 角色 | 职责 |
|------|------|
| **Master** | 协调工作，协调训练 |
| **Worker** | 执行实际训练 |

### 环境变量

PyTorchJob 自动设置：

| 变量 | 说明 |
|------|------|
| `WORLD_SIZE` | 总进程数 |
| `RANK` | 当前进程全局 ID |
| `LOCAL_RANK` | 当前节点内 ID |
| `MASTER_ADDR` | Master Pod 地址 |
| `MASTER_PORT` | 通信端口 |

### 训练脚本

```python
import os
import torch
import torch.distributed as dist

# 1. 获取进程信息
rank = int(os.environ['RANK'])
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ['WORLD_SIZE'])
master_addr = os.environ['MASTER_ADDR']
master_port = os.environ.get('MASTER_PORT', '29500')

# 2. 初始化分布式环境
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

dist.init_process_group(backend='nccl')

# 3. 设置 GPU
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

# 4. 创建模型并移动到 GPU
model = MyModel().to(device)
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[local_rank]
)

# 5. 训练
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

## 部署步骤

### 1. 安装 Kubeflow

```bash
# 安装 kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

# 安装 Kubeflow
kustomize build example | kubectl apply -f -
```

### 2. 提交 PyTorchJob

```bash
kubectl apply -f pytorchjob.yaml
```

### 3. 查看状态

```bash
# 查看 Pod 状态
kubectl get pods -l job-name=pytorch-distributed-job

# 查看日志
kubectl logs -l job-name=pytorch-distributed-job -c pytorch

# 查看 PyTorchJob 状态
kubectl get pytorchjob pytorch-distributed-job
```

### 4. 故障排查

```bash
# 查看事件
kubectl describe pytorchjob pytorch-distributed-job

# 查看 worker 日志
kubectl logs pytorch-distributed-job-worker-0 -c pytorch
```

## 多节点训练

### 主机网络模式

```
┌─────────────────┐     ┌─────────────────┐
│   Node 1        │     │   Node 2        │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Worker 0  │──┼─────┼─│ Worker 1  │  │
│  │ (GPU 0)   │  │     │  │ (GPU 0)   │  │
│  └───────────┘  │     │  └───────────┘  │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Worker 2  │──┼─────┼─│ Worker 3  │  │
│  │ (GPU 1)   │  │     │  │ (GPU 1)   │  │
│  └───────────┘  │     │  └───────────┘  │
└─────────────────┘     └─────────────────┘
```

### RDMA 支持

在 K8s 中启用 RDMA：

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: rdma-container
    image: rdma-image
    resources:
      limits:
        rdma/ib mlx5: 1  # 请求 RDMA 设备
```

## 本章测验

- [ ] 理解 Kubernetes 上的分布式训练优势
- [ ] 掌握 PyTorchJob 配置
- [ ] 能编写支持 K8s 的训练脚本
- [ ] 理解故障恢复机制

## 下一步

学完本章后，进入 [第十章：全链路观测](./ch10-fullstack-observe/README.md) 学习端到端性能观测。
