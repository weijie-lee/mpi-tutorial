# 第十一章：通信库、NCCL、DeepEP 与 MoE

这一章把深度学习里几套通信库分开，并核对专家并行（Expert Parallelism，EP）里真正在传的数据。数据并行的梯度用 AllReduce。混合专家（Mixture of Experts，MoE）把 token 送到不同专家，用的是不等长的 all-to-all：先 dispatch，专家算完再 combine。

可在 CPU 上核对的实验会打印 `RESULT: PASS`。本机没有 CUDA，也没有安装 `deep_ep`，NCCL 和 DeepEP 的探针打印 `RESULT: SKIP`。SKIP 的退出码是 0。不要把 SKIP 读成“已经测过 GPU 带宽”。

## 通信库各管哪一段

| 库 | 数据放在哪 | 常用原语 | 典型用途 |
| --- | --- | --- | --- |
| MPI | CPU 缓冲；CUDA-aware 时地址可以在 GPU 上 | `MPI_Send`、`MPI_Allreduce`、`MPI_Alltoallv`、RMA | 科学计算，以及本教程前面的例子 |
| Gloo | CPU | AllReduce、不等长 all-to-all | PyTorch 没有 GPU 时的进程组 |
| NCCL | NVIDIA GPU | AllReduce、AllGather、ReduceScatter、Broadcast、Send/Recv；2.28 起另有等长的 `ncclAlltoAll` | 数据并行、张量并行、FSDP |
| DeepEP | NVIDIA GPU | `EPBuffer.dispatch` / `combine` | MoE 的专家并行 |

RCCL 是 AMD GPU 上对应 NCCL 的集合通信库，oneCCL 是 Intel 的。它们和 NCCL 一样服务于集合通信，接口随各自的运行时走。

一次训练里这几层是叠起来的，不是互相替代：

```text
模型代码（DDP、张量并行、MoE）
    进程组：torch.distributed，backend 选 nccl 或 gloo
        NCCL：GPU 上的集合通信或点对点
            NVLink / PCIe，跨机则是 socket 或 InfiniBand / RoCE
```

`mpirun` 和 `torchrun` 负责把进程拉起来。NCCL 自己搬 GPU 上的数据，不把梯度交给 MPI。CUDA-aware MPI 是另一条路：应用直接调用 `MPI_Allreduce`，缓冲用 GPU 地址。第五章里“MPI 负责跨节点、NCCL 负责卡间”只描述了一种旧的部署。

## 哪种并行用哪种原语

| 并行 | 各 rank 上的数据 | 通信 |
| --- | --- | --- |
| 数据并行 | 模型相同，样本不同，梯度同形 | AllReduce。FSDP / ZeRO 把一次 AllReduce 拆成 ReduceScatter 加 AllGather |
| 张量并行 | 同一层的矩阵切在不同 GPU 上 | 激活的 AllGather，或反向的 ReduceScatter |
| 流水线并行 | 不同 stage 各拿若干层 | 相邻 stage 的 Send / Recv |
| 专家并行 | 每个 rank 只放一部分专家 | 不等长 all-to-all：dispatch 送出 token，combine 把结果加回来 |

AllReduce 要求每个 rank 拿出同一形状的缓冲，结束时每个 rank 都有相同的和。MoE 里 rank A 要送给专家 3 的 token 个数，和送给专家 7 的个数没有理由相等，所以对不上 AllReduce，也对不上“每个目的地长度相同”的 AlltoAll。

总线系数仍然只用于 AllReduce 那种集合通信。消息有 `N` 字节、`p` 个 rank 时，ring AllReduce 每个 rank 在总线上承担的字节数是 `2*(p-1)/p * N`。`p = 3` 时系数是 `4/3`。`collective_contract.py` 会打印这个系数，并核对 AllReduce、AllGather、ReduceScatter 的数值。`p = 2` 时系数是 1，算法带宽和总线带宽相同。

## NCCL

NCCL 面向 GPU。调用和 CUDA stream 绑在一起：kernel 启动之后，host 可以继续往同一个 stream 里排队，完成顺序由 stream 保证。

常用 host API：

| 调用 | 结束时每个 rank 得到什么 |
| --- | --- |
| `ncclAllReduce` | 全体缓冲按元素求和（或其他 op） |
| `ncclBroadcast` | root 的缓冲 |
| `ncclAllGather` | 把每个 rank 的一份拼起来 |
| `ncclReduceScatter` | 先归约，再让每个 rank 留下自己那一段 |
| `ncclSend` / `ncclRecv` | 一段点对点数据 |
| `ncclAlltoAll` | 每个 rank 向每个目的地发送同样多的元素 |

`ncclAlltoAll` 从 NCCL 2.28 进入 host API。它要求 `sendbuff` 至少有 `nranks * count` 个元素，发给 rank `j` 的数据放在偏移 `j * count`。原地操作目前不支持。MoE 的接收长度随路由变化，这个等长接口覆盖不了。

不等长交换用 `ncclGroupStart` / `ncclGroupEnd` 包住一组 `ncclSend` 和 `ncclRecv`。每个 rank 都要进入这一组，发给对端的长度要和对方准备接收的长度一致。这和第二章的匹配规则是同一件事：少一个 rank，其余 rank 就停在这次组调用上。PyTorch 的 `dist.all_to_all_single(..., output_split_sizes, input_split_sizes)` 走的就是这种不等长交换。

小消息和大消息在 NCCL 里走不同算法和协议。Ring 省带宽，延迟随 rank 数变长；Tree 对小的 AllReduce 更合适。Hopper 上还有 NVLS，用 NVLink 的硬件多播做归约。协议名是 LL、LL128 和 Simple。环境变量 `NCCL_ALGO`、`NCCL_PROTO`、`NCCL_IB_DISABLE`、`NCCL_DEBUG` 要在 `import torch` 之前设置，库可能在导入时就读走它们。没有 IB 设备时，NCCL 会退回 socket。日志里出现 TCP 就说明这次走了 socket，不能因为没写 `NCCL_IB_DISABLE` 就把它记成 RDMA。

NCCL 2.28 起另外有一套设备侧 API。Gin（GPU-Initiated Networking）从 2.28.7 起可以在 CUDA kernel 里对已经注册的 window 做 `put`。DeepEP 当前主线的设备核用的是这条路径。它和 host 上的 `ncclAlltoAll` 不是同一个函数。

## MoE：从路由到 combine

一层 MoE 的前向，通信只发生在专家计算的两侧。

```text
token x 在 rank src
  router 给出 (expert a, w_a), (expert b, w_b)，无效槽位的专家下标是 -1
  dispatch：把 x 送到 owner(a) 和 owner(b)
  专家计算：y_e = Expert_e(x)
  乘上 router 权重：z_e = w_e * y_e
  combine：原来的 token 位置得到 z_a + z_b，还可以加上共享专家的 bias
```

专家编号按连续的一段分给 rank。`local_experts` 是每个 rank 上的专家个数时：

```text
owner(expert) = expert / local_experts
local_id      = expert % local_experts
expert        = owner * local_experts + local_id
```

`-1` 表示这个 top-k 槽位没有专家，不发送。一个 token 的两个专家若落在同一个 rank，dispatch 仍然是两行，因为两个专家各算各的。combine 之前先在这个 rank 上把两行加权结果加起来，回传一条。数值等于两行分别送回原 rank 再相加，线上的消息更少。

本教程的参考实现用这个专家函数，方便手算：

```text
Expert_e(h) = h * (e + 1)
```

隐藏维是 4。rank `r` 上 token `t` 的输入是 `h[d] = r * 1000 + t * 10 + d`。权重 `k = 0` 为 0.5，`k = 1` 为 0.25。combine 之后每个元素再加 bias `1`。

路由是固定的，这样每次运行的消息条数都能对上：

- token 0 的两个专家都在本 rank：`r * 2` 和 `r * 2 + 1`
- token 1 只有 `k = 0` 有效，`k = 1` 是 -1
- 其余 token 的两个专家下标不同，可能在别的 rank

### 手算 rank 0 的 token 0

`h[d] = d`。专家 0 的贡献是 `0.5 * h[d] * 1`。专家 1 的贡献是 `0.25 * h[d] * 2 = 0.5 * h[d]`。两项相加是 `h[d]`，再加 bias：

```text
result[d] = d + 1
```

所以这个 token 必须是 `1, 2, 3, 4`。MPI 程序和 gloo 程序都检查这个向量，也检查其余 token 与同一公式算出的参考解逐元素相同。差必须是 0。

任意 rank 的 token 0 也有闭式。两个专家的系数和是 `1.5 * r + 1`：

```text
result[d] = (1.5 * r + 1) * (1000 * r + d) + 1
```

`r = 0` 时它退回到 `d + 1`。

反向把同一对通信对调过来。combine 的反向是把梯度再 dispatch 到各个专家；dispatch 的反向是把专家侧的梯度 combine 回 token。DeepEP 的训练示例就是这样调用的。

一次 dispatch 的载荷按“每个有效 top-k 槽位一行”计算：

```text
dispatch_bytes = slots * hidden * sizeof(dtype)
combine_bytes  = combine_msgs * hidden * sizeof(dtype)
```

`combine_msgs` 是 `(源 rank, token, 目的 rank)` 的个数。token 0 在每个 rank 上把 2 个槽位收成 1 条回传，所以 `combine_msgs < slots`。4 个 rank、每个 rank 8 个 token 时，参考路由的计数是 `slots = 60`、`combine_msgs = 56`，double 元素分别是 240 和 224。最忙的 rank 收到 17 行，平均 15 行。这些是这条路由在 CPU 参考实现里的条数，不是 DeepEP 的带宽。

## DeepEP

DeepEP 是 DeepSeek 开源的专家并行通信库，解决的是 dispatch 和 combine，不是又一个通用 AllReduce。按 2026-11 的主线（V2.5）：

- 高吞吐和低延迟共用 `EPBuffer`。`dispatch` 把 token 按 `topk_idx` 送到专家所在的 rank，`combine` 把专家输出加回原来的 token。
- 节点内走 NVLink，节点间走 RDMA。通信核在运行时用 DeepJIT 编译。
- 后端是 NCCL 的 Gin。`EP_REUSE_NCCL_COMM` 默认打开时，可以复用 PyTorch 进程组里已经有的 NCCL communicator。
- 依赖是 Linux、Python 3.10+、Hopper 或更新的 GPU、CUDA 13.1+、带 CUDA 的 PyTorch 2.10+、NCCL 2.32.3+。安装阶段要能找到这套 NCCL 的头文件和库。
- 主线已经删除 V1，包括当时的 `Buffer`、`low_latency_dispatch` 和 NVSHMEM 依赖。2025 年初的博客还在讲 NVSHMEM 和 IBGDA，对代码时要看仓库版本。

训练示例把 epilogue 推迟，调用方在 `current_stream_wait()` 之后才使用展开后的张量。形状是下面这样，参数以库的 README 为准：

```python
buffer = EPBuffer(group, num_max_tokens_per_rank=capacity,
                  hidden=hidden, num_topk=topk)
event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=num_experts, num_max_tokens_per_rank=capacity,
    do_cpu_sync=True, do_expand=True, defer_epilogue=True,
)
recv_x, recv_topk_idx, recv_topk_weights, handle = event.current_stream_wait()
# 分组 GEMM，然后把 recv_topk_weights 乘到专家输出上
event = buffer.combine(expert_out, handle=handle, defer_epilogue=True)
combined_x, _ = event.current_stream_wait()
```

当前主线的 combine 做不带权重的加法。权重在专家计算之后、调用 combine 之前乘上去。展开后的 `recv_topk_weights` 是一维的，一行专家对应一个权重。`do_cpu_sync=True` 时拿到精确的接收长度，训练和 prefill 用这个。解码可以把 `do_cpu_sync` 设为 False，输出按事先约定的容量分配，有效区间看 handle 里的 `psum_num_recv_tokens_per_expert`。`expert_alignment` 把每个本地专家的行数补齐到分组 GEMM 要求的倍数，空隙填 0，不参与求和。

`num_max_tokens_per_rank` 是所有 rank 事先说好的容量，用来给 buffer 定大小。它小于实际路由时，这次交换没有定义好的结果。参考实现里用“某个 rank 收到的专家行数超过上限”来拒绝这种计划。DeepEP 自己的 `EPBuffer.get_buffer_size_hint` 还依赖 hidden 和 top-k，字节数以那个函数为准，本教程不另造一个字节公式。

2025 年的低延迟 kernel 把权重乘法留在 combine 里面，因为解码时一个 token 在一张 GPU 上通常只会命中一个专家。那个 `low_latency_combine` 已随 V1 删除。

库里还有几块和 dispatch 并列、本教程没有在 CPU 上模拟的功能：

- `BucketBuffer`：给上下文并行和数据并行用的 all-gather、reduce-scatter、all-reduce，标记为实验性质。
- `lb_prefetch_weights` / `lb_reduce_grads`：把冗余专家的权重从 NVLink 拉过来，再把这些副本的 FP32 梯度加回原专家。
- `EngramBuffer` 和 `PPBuffer`：实验性质的远程表查找，以及流水线 stage 之间的 send/recv。

dispatch 和 combine 要占用 GPU 的 SM。主线不提供零 SM 的 RDMA 专家并行。

## 专家容量和丢 token

Switch Transformer 和 GShard 里还有另一种容量：进通信之前就丢掉多余 token。每个专家保留的 token 数是

```text
capacity = ceil(num_tokens / num_experts * topk * capacity_factor)
```

后到的 token 把专家下标改成 -1，于是 dispatch 根本不会发送它。4 个 token、2 个专家、top-k 为 1、容量系数 1 时，容量是 2。路由 `[0, 0, 0, 1]` 保留成 `[0, 0, -1, 1]`。这是训练框架的策略。DeepEP 的 `num_max_tokens_per_rank` 是缓冲区上限，库不会按这个公式替你丢 token。`deepep_layout.py` 把两件事分开检查。

## 实验

代码在 `11-comm-libs/`。

```bash
cd ch11-comm-libs/11-comm-libs

mpicc -O2 -Wall -Wextra -Werror -std=c11 -o moe_alltoall moe_alltoall.c
mpirun -np 4 ./moe_alltoall

python3 deepep_layout.py
python3 deepep_probe.py

export MASTER_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo
export MASTER_PORT=29550
mpirun -np 4 python3 moe_dispatch.py

export MASTER_PORT=29551
mpirun -np 3 python3 collective_contract.py

export MASTER_PORT=29552
mpirun -np 2 python3 nccl_probe.py
```

只有 loopback 时加上 `OMPI_MCA_btl=tcp,self` 和 `OMPI_MCA_btl_tcp_if_include=lo`。一次把本章和前面各章都跑掉：

```bash
bash scripts/verify.sh
```

| 程序 | 在做什么 | 通过标准 |
| --- | --- | --- |
| `moe_alltoall.c` | `MPI_Alltoall` 交换条数，`MPI_Alltoallv` 交换 token，再把部分和送回 | 与参考解差为 0，rank 0 的 token 0 是 `1,2,3,4`，回传条数少于槽位数 |
| `moe_dispatch.py` | 同一路由，`all_to_all_single` 使用不等长 split | 同上。没有 GPU 时 backend 是 `gloo` |
| `deepep_layout.py` | 单进程模拟分块、补齐、容量上限，以及 Switch 的丢 token | 补齐行不改变和；容量减一会被拒绝；`[0,0,0,1]` 丢掉第三个 0 |
| `collective_contract.py` | AllReduce、AllGather、ReduceScatter、含 0 长度的不等长 all-to-all | 和、拼起来的 rank 号、reduce-scatter 的 `world`、按对端填的值都对 |
| `nccl_probe.py` | 有 CUDA 和 NCCL 时做 AllReduce 与等长 all-to-all | 没有 CUDA 构建或没有 GPU 时 `SKIP` |
| `deepep_probe.py` | 尝试 `import deep_ep` | 没安装或没有 GPU 时 `SKIP`，不创建 `EPBuffer`，不编造带宽 |

4 个 rank 的 MPI 和 gloo 都应打印 `slots=60 combine_msgs=56`。3 个 rank 的集合通信合同里，rank 0 的发送长度是 `0,2,3`，AllReduce 的和是 6，总线系数是 `1.3333`。

## 本环境跑不到的部分

没有 NVIDIA GPU 时，NCCL 的集合通信不会执行，DeepEP 的 kernel 也不能编译运行。探针的 SKIP 就是这个结论。有 Hopper GPU、CUDA 13.1+、NCCL 2.32.3+ 和一份 CUDA 版 PyTorch 之后，再跑上游的 `tests/ep/test_ep.py`。那一步才是 DeepEP 自己的正确性测试。本仓库的 CPU 实验只保证：路由、不等长交换、权重、局部求和和 bias 放在一起时，token 和手算一致。

## 下一步

部署到 Kubernetes 的清单在 [第九章](../ch09-kubernetes-pytorchjob/README.md)。全链路里 NCCL 环境变量怎么读，在 [第十章](../ch10-fullstack-observe/README.md)。
