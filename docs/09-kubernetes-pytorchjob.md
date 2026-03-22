# 九、Kubernetes 上运行 MPI：PyTorchJob 实战

在云计算和 AI 平台化的今天，越来越多的集群使用 Kubernetes 进行资源管理和任务调度。本章讲解如何在 Kubernetes 上通过 [PyTorchJob](https://github.com/kubeflow/training-operator/blob/master/examples/pytorch/elastic/README.md) 运行 MPI 分布式训练。

## 1. 基本原理

### 为什么需要 Kubernetes + PyTorchJob？

在传统 HPC 集群上，你用 `mpirun`/`srun` 直接启动 MPI 任务。在 Kubernetes 环境中：

- **Kubernetes** 负责资源调度、节点分配、网络管理
- **PyTorchJob** 是 Kubeflow Training Operator 提供的 CRD，专门用于管理 PyTorch 分布式训练任务
- **MPI 角色**：仍然负责**进程启动**和**通信协调**，底层网络用 Kubernetes CNI 或 RDMA 透传

### 架构对比

| 部署方式 | 进程启动 | 资源调度 | 网络 | 适用场景 |
|----------|----------|----------|------|----------|
| 传统裸机 | `mpirun`/`srun` | SLURM/PBS | IB/RoCE | HPC 集群 |
| Kubernetes + PyTorchJob | Operator 启动 Pod，`mpirun` 启动 MPI | Kubernetes | Kubernetes CNI / RDMA | 云原生 AI 平台 |

### PyTorchJob 工作原理

1. 用户创建一个 `PyTorchJob` YAML，指定 Worker 数量（一般每个 Worker 对应一张 GPU）
2. Training Operator 会自动：
   - 创建指定数量的 Worker Pod
   - 每个 Pod 启动你的容器，启动命令一般是你的训练脚本
   - 通过 `torchrun` 或者 MPI 进行初始化和通信
3. 对于 MPI 模式：**Rank 0** 作为 master，其他 Worker 作为 slave，通过 MPI 通信

## 2. 前置条件

在你的 Kubernetes 集群上需要：

1. **安装了 Kubeflow Training Operator**（支持 PyTorchJob CRD）
   ```bash
   # 检查是否安装
   kubectl get crd pytorchjobs.kubeflow.org
   ```
   如果没有，参考官方文档安装：https://www.kubeflow.org/docs/components/training/

2. **GPU 节点可用**：每个节点能分配到你需要的 GPU（我们默认环境每个节点 8 张 GPU）
3. **私有镜像仓库**（可选）：保存你的训练镜像
4. **共享存储**（可选）：数据集和 checkpoint 需要存在所有 Pod 能访问的地方（比如 PVC、NFS、对象存储）

## 3. 完整示例：单集群 16 GPU (2节点 × 8GPU)

### 步骤 1：准备 Docker 镜像

首先，你需要把你的训练代码和依赖打包成 Docker 镜像。示例 `Dockerfile`:

```dockerfile
# 基础镜像用官方 PyTorch 已经装好 CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 安装 MPI（OpenMPI）
RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 NCCL 和 PyTorch（已经在基础镜像有了，这里可以跳过）
RUN pip install --no-cache-dir torch torchvision

# 把训练代码拷进镜像
WORKDIR /workspace
COPY pytorch_ddp_mpi.py .

# 默认启动命令
CMD ["python", "pytorch_ddp_mpi.py"]
```

构建并推送到你的镜像仓库：
```bash
docker build -t your-registry/mpi-pytorch-example:v1 .
docker push your-registry/mpi-pytorch-example:v1
```

> 💡 我们教程里的 `pytorch_ddp_mpi.py` 可以直接用，代码已经适配了任意数量的 GPU。

### 步骤 2：编写 PyTorchJob YAML

下面是一个完整的例子，**2 节点共 16 GPU**（每节点 8 张）：

```yaml
# mpi-pytorch-job.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: mpi-ddp-example
spec:
  pytorchReplicaSpecs:
    # Master 节点：一般就是 rank 0
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/mpi-pytorch-example:v1
            command: ["mpirun"]
            args:
            - "-np"
            - "16"
            - "-H"
            - "localhost:8,pytorch-ddp-example-worker-0:8"
            - "python"
            - "/workspace/pytorch_ddp_mpi.py"
            resources:
              limits:
                nvidia.com/gpu: 8  # 本节点用 8 张 GPU
            volumeMounts:
            - name: dataset
              mountPath: /data
          volumes:
          - name: dataset
            persistentVolumeClaim:
              claimName: shared-dataset-pvc
    # Worker 节点：剩下的节点
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/mpi-pytorch-example:v1
            # Worker 只需要留着，master 上的 mpirun 会 ssh 过来启动
            command: ["sleep"]
            args: ["infinity"]
            resources:
              limits:
                nvidia.com/gpu: 8  # 本节点用 8 张 GPU
            volumeMounts:
            - name: dataset
              mountPath: /data
          volumes:
          - name: dataset
            persistentVolumeClaim:
              claimName: shared-dataset-pvc
```

> 💡 **为什么 Worker 要 sleep infinity？**
> 因为 `mpirun` 需要在所有节点上启动进程，所以 Master 节点通过 SSH 连接到 Worker 节点来启动进程。Worker 容器保持运行就行。

### 单节点 8GPU 简化版

如果你只跑单节点 8GPU，可以更简单：

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: mpi-ddp-single-node
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/mpi-pytorch-example:v1
            command: ["mpirun"]
            args: ["-np", "8", "python", "/workspace/pytorch_ddp_mpi.py"]
            resources:
              limits:
                nvidia.com/gpu: 8
    Worker:
      replicas: 0
```

### 步骤 3：允许 Master SSH 到 Worker（关键）

`mpirun` 需要在各个节点之间启动进程，默认需要 SSH 访问。有几种方式配置：

**方式一：用 SSH 秘钥认证**

1. 生成 SSH 密钥对：
   ```bash
   ssh-keygen -t rsa -f ssh-key -N ""
   ```

2. 在 Docker 镜像中把公钥加到 `authorized_keys`：
   ```dockerfile
   # 在 Dockerfile 中添加
   COPY ssh-key /root/.ssh/id_rsa
   COPY ssh-key.pub /root/.ssh/id_rsa.pub
   RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys \
       && chmod 600 /root/.ssh/*
   ```

3. 确保 Kubernetes DNS 正常工作，Pod 之间可以通过主机名解析。

**方式二：使用 kubectl 端口转发 + 无密码（开发环境用）**

在开发测试环境，你也可以配置 `.ssh/config` 禁用 StrictHostKeyChecking：

```
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
```

**方式三：使用 MPI 原生启动，不需要 SSH（推荐）**

如果你的 MPI 版本比较新，也可以用 `--enable-static=no` 或者让 Kubernetes 直接每个Pod启动一个进程，然后互相通信。这其实更云原生。

### 更云原生的写法：每个 Pod 一个进程

这是现在更推荐的方式：每个 Pod 只启动一个 MPI 进程，由 Kubernetes 调度，不需要 `mpirun` 通过 SSH 远程启动：

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: mpi-ddp-per-pod
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/mpi-pytorch-example:v1
            command: ["python"]
            args: ["/workspace/pytorch_ddp_mpi.py"]
            resources:
              limits:
                nvidia.com/gpu: 1
            env:
            - name: WORLD_SIZE
              value: "8"
            - name: RANK
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['kubeflow.org/pod-index']
    Worker:
      replicas: 7
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: your-registry/mpi-pytorch-example:v1
            command: ["python"]
            args: ["/workspace/pytorch_ddp_mpi.py"]
            resources:
              limits:
                nvidia.com/gpu: 1
            env:
            - name: WORLD_SIZE
              value: "8"
            - name: RANK
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['kubeflow.org/pod-index']
```

这种方式的优点：
- 每个 GPU 一个 Pod，Kubernetes 可以更灵活的调度
- 不需要 SSH 信任配置，更安全
- 可以跨节点弹性调度

缺点：需要你自己在代码里根据环境变量初始化 MPI rank 和 size。

## 4. RDMA 在 Kubernetes 上的配置

如果你的 Kubernetes 集群有 RDMA/IB 网络，需要配置 RDMA 才能发挥最佳性能：

1. **安装 RDMA CNI**：https://github.com/k8snetworkplumbingwg/rdma-cni
2. **配置网络**: 给 MPI 训练任务使用 RDMA 网络
3. **启用 GPU Direct RDMA**: NCCL 会自动检测到 RDMA 并使用

检查 NCCL 是否用到 RDMA：
```bash
export NCCL_DEBUG=INFO
```
日志看到 `NCCL INFO Using network IB` 说明成功使用 RDMA。

## 5. 常见问题排查

### Q1: `mpirun` 卡在 "connecting to" 不动了

**可能原因**：
1. Master 无法 SSH 到 Worker → 检查 SSH 秘钥和网络连通性
2. 防火墙阻挡了 MPI 端口 → 检查安全组和网络策略
3. DNS 解析不了 Worker 主机名 → 检查 Kubernetes DNS 是否正常

**排查方法**：
```bash
# 进 Master Pod 试试手动 SSH 到 Worker
kubectl exec -it mpi-ddp-example-master-0 -- ssh pytorch-ddp-example-worker-0 hostname
```

### Q2: 某些 GPU 节点初始化失败，NCCL 找不到网卡

**解决方法**：设置 NCCL 网卡，如果 RDMA 不行强制走 TCP：
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

### Q3: 资源分配不到足够的 GPU

**检查**：
```bash
# 看看集群还有多少 GPU 空闲
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

### Q4: PyTorchJob 一直处于 Pending

**原因**：一般是资源不足，检查：
```bash
kubectl describe pytorchjob mpi-ddp-example
```

## 6. 对比：TorchElastic vs MPI 在 Kubernetes

| 特性 | TorchElastic (PyTorchJob 默认) | MPI |
|------|--------------------------------|-----|
| 弹性伸缩 | 支持故障替换和动态伸缩 | 一般固定规模 |
| 依赖 | 不需要 SSH | 需要 SSH 信任 |
| 多节点 | 需要配置 etcd | 原生支持 |
| 性能 | 不错，但是 NCCL 通信还是底层 | MPI 做协调更成熟，和传统习惯一致 |
| 使用场景 | 云原生动态调度 | 传统 MPI 程序迁移过来 |

## 总结

在 Kubernetes 上用 PyTorchJob 跑 MPI 分布式训练主要有两种方式：

1. **传统方式**：一个 Pod 一个节点，Master 用 `mpirun` 通过 SSH 启动所有节点上的进程，适合从传统 HPC 迁移过来的工作流
2. **云原生方式**：一个 Pod 一个 GPU，Kubernetes 调度，每个进程自己初始化，更灵活，不需要 SSH

两种方式都能用，看你的集群环境和使用习惯。我们提供的 PyTorch DDP MPI 示例代码两种方式都支持。

## 下一步

→ [回到目录](../README.md)
