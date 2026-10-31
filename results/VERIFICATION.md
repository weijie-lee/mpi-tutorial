# 验证结果

由 `scripts/verify.sh` 生成。数字来自这次实际运行，不是文档里的数量级示例。

- 时间：2026-11-01 02:15:11 CST
- 主机：Linux 6.12.94+ x86_64
- C 编译器：gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- MPI：mpirun (Open MPI) 4.1.6
- PyTorch：2.15.0+cpu，MPI 后端 False，Gloo True，NCCL False，CUDA False
- GPU：没有 CUDA 设备。NCCL 探针应为 SKIP。
- DeepEP：未安装 deep_ep。探针应为 SKIP。MoE 的 dispatch/combine 在 CPU 上核对。
- RDMA：本机没有 active 的 verbs 设备，内核也不能创建 soft-RoCE。相关程序应打印 SKIP。
- Kubernetes：只检查了清单能被解析，没有提交到集群，也没有构建训练镜像。

## 汇总

失败数：0

| 实验 | 结果 |
| --- | --- |
| hello np=4 | PASS |
| timing np=2 | PASS |
| error_handling np=2 | PASS |
| sendrecv np=2 | PASS |
| deadlock safe np=2 | PASS |
| nonblocking np=3 | PASS |
| collectives np=3 | PASS |
| all-collectives np=5 | PASS |
| pi_monte_carlo np=4 | PASS |
| derived_type np=2 | PASS |
| cartesian np=6 | PASS |
| comm_split np=5 | PASS |
| rma_putget np=2 | PASS |
| vector_indexed np=2 | PASS |
| jacobi2d np=4 N=32 | PASS |
| jacobi2d np=3 N=32 | PASS |
| allreduce_bench np=4 | PASS |
| blocking_vs_nonblocking np=2 | PASS |
| pingpong np=2 | PASS |
| moe_alltoall np=1 | PASS |
| moe_alltoall np=3 | PASS |
| moe_alltoall np=4 | PASS |
| deepep_layout | PASS |
| rdma write ch04 | SKIP |
| verbs send/write/read ch08 | SKIP |
| mpi_basic np=2 | PASS |
| pytorch_ddp np=2 | PASS |
| train.py np=1 | PASS |
| train.py np=2 | PASS |
| benchmark_compare np=2 | PASS |
| nccl_rdma_demo np=2 | PASS |
| moe_dispatch np=1 | PASS |
| moe_dispatch np=4 | PASS |
| collective_contract np=3 | PASS |
| nccl_probe np=2 | SKIP |
| deepep_probe | SKIP |
| pytorchjob yaml | PASS |

## hello np=4

结果：**PASS**，退出码 0。

```text
Hello MPI! I'm rank 0 / 4 on host: cursor
Hello MPI! I'm rank 1 / 4 on host: cursor
Hello MPI! I'm rank 2 / 4 on host: cursor
Hello MPI! I'm rank 3 / 4 on host: cursor
RESULT: PASS
```

## timing np=2

结果：**PASS**，退出码 0。

```text
MPI_Wtick resolution: 1.000e-09 seconds
Max elapsed time across ranks: 1.000 seconds
This should be roughly 1 second (sleep(1))
RESULT: PASS
```

## error_handling np=2

结果：**PASS**，退出码 0。

```text
No error on the success path, rank = 1 / 2
No error on the success path, rank = 0 / 2
MPI_Send to rank == size -> code 6 (MPI_ERR_RANK: invalid rank)
MPI_Comm_rank(MPI_COMM_NULL) -> code 5 (MPI_ERR_COMM: invalid communicator)
RESULT: PASS
```

## sendrecv np=2

结果：**PASS**，退出码 0。

```text
Rank 0: sending int data = 42 to rank 1
Rank 1: received data = 42 from rank 0
Rank 1: actually received 1 elements
Rank 1: probed 5 elements: 10 20 30 40 50
RESULT: PASS
```

## deadlock safe np=2

结果：**PASS**，退出码 0。

```text
Ordered Send/Recv, MPI_Sendrecv and nonblocking exchange checked
RESULT: PASS
```

## nonblocking np=3

结果：**PASS**，退出码 0。

```text
Rank 1: sent 1, received 0 from rank 0
Rank 0: sent 0, received 1 from rank 1
RESULT: PASS
```

## collectives np=3

结果：**PASS**，退出码 0。

```text
Root (rank 0) broadcasting data: 12345
Rank 0: received broadcast data: 12345
Rank 1: received broadcast data: 12345
Rank 1: global sum via Allreduce: 5.000000
Rank 1: scattered value: 10
Rank 2: received broadcast data: 12345
Rank 2: global sum via Allreduce: 5.000000
Rank 2: scattered value: 20

Sum of rank^2 from all ranks: 5.000000 (expected 5.000000)
Rank 0: global sum via Allreduce: 5.000000

Scattering from root: 0 10 20 
Rank 0: scattered value: 0
After gather (doubled): 0 20 40 
RESULT: PASS
```

## all-collectives np=5

结果：**PASS**，退出码 0。

```text
===== Rank 4 / 5 =====
Rank 4: waiting at barrier
===== Rank 1 / 5 =====
Rank 1: waiting at barrier
===== Rank 2 / 5 =====
Rank 2: waiting at barrier
===== Rank 0 / 5 =====
Rank 0: waiting at barrier
===== Rank 3 / 5 =====
Rank 3: waiting at barrier

=== After Barrier, everyone continues ===

Rank 0: Bcasting data 12345
Rank 0: received bcast data 12345
Rank 0: Scatter sending: 10 20 30 40 50
Rank 0: Scatter received 10
Rank 1: received bcast data 12345
Rank 1: Scatter received 20
Rank 3: received bcast data 12345
Rank 2: received bcast data 12345
Rank 4: received bcast data 12345
Rank 4: Scatter received 50
Rank 2: Scatter received 30
Rank 3: Scatter received 40
Rank 0: Gather received: 10 20 30 40 50
Rank 3: Allgather received: 1 2 3 4 5
Rank 1: Allgather received: 1 2 3 4 5
Rank 4: Allgather received: 1 2 3 4 5
Rank 0: Allgather received: 1 2 3 4 5
Rank 2: Allgather received: 1 2 3 4 5
Rank 0: Reduce sum result: 15.0, expected: 15.0
Rank 1: Allreduce sum result = 15.0, expected 15.0
Rank 2: Allreduce sum result = 15.0, expected 15.0
Rank 4: Allreduce sum result = 15.0, expected 15.0
Rank 0: Allreduce sum result = 15.0, expected 15.0
Rank 3: Allreduce sum result = 15.0, expected 15.0
Rank 0: Scan prefix sum = 1.0, expected 1.0
Rank 1: Scan prefix sum = 3.0, expected 3.0
Rank 2: Scan prefix sum = 6.0, expected 6.0
Rank 3: Scan prefix sum = 10.0, expected 10.0
Rank 4: Scan prefix sum = 15.0, expected 15.0
RESULT: PASS
```

## pi_monte_carlo np=4

结果：**PASS**，退出码 0。

```text
Total points: 800000
Estimated pi = 3.1434350000
Real    pi = 3.1415926535...
Error   = 0.0018423464
RESULT: PASS
```

## derived_type np=2

结果：**PASS**，退出码 0。

```text
Rank 0: sending Particle id=42, x=1.500000, y=2.700000
Rank 1: received Particle id=42, x=1.500000, y=2.700000
RESULT: PASS
```

## cartesian np=6

结果：**PASS**，退出码 0。

```text
Creating 3 x 2 2D grid for 6 processes
World rank 1 -> cart rank 1 coords (0, 1), neighbors L=-2 R=3 U=0 D=-2
World rank 2 -> cart rank 2 coords (1, 0), neighbors L=0 R=4 U=-2 D=3
World rank 3 -> cart rank 3 coords (1, 1), neighbors L=1 R=5 U=2 D=-2
World rank 4 -> cart rank 4 coords (2, 0), neighbors L=2 R=-2 U=-2 D=5
World rank 5 -> cart rank 5 coords (2, 1), neighbors L=3 R=-2 U=4 D=-2
World rank 0 -> cart rank 0 coords (0, 0), neighbors L=-2 R=2 U=-2 D=1
RESULT: PASS
```

## comm_split np=5

结果：**PASS**，退出码 0。

```text
Original rank=4, color=0 -> New comm: rank=2, size=3
Original rank=0, color=0 -> New comm: rank=0, size=3
Original rank=1, color=1 -> New comm: rank=0, size=2
Original rank=2, color=0 -> New comm: rank=1, size=3
Original rank=3, color=1 -> New comm: rank=1, size=2
Even-rank group size = 3
RESULT: PASS
```

## rma_putget np=2

结果：**PASS**，退出码 0。

```text
Rank 0: data after put from rank 1 is now: 5678
Rank 1: got data 1234 from rank 0
Rank 1: put new data 5678 to rank 0
RESULT: PASS
```

## vector_indexed np=2

结果：**PASS**，退出码 0。

```text
Column 2: 2 12 22 32
Indexed elements: 1 4 11
RESULT: PASS
```

## jacobi2d np=4 N=32

结果：**PASS**，退出码 0。

```text
Iter 0: diff = 1.41421356e+00
Iter 100: diff = 2.88956709e-02
Iter 200: diff = 1.37054474e-02
Total elapsed time: 0.0085 seconds
Grid size: 32 x 32, 4 processes, iterations: 300
Max abs diff vs serial Jacobi: 0.000e+00
RESULT: PASS
```

## jacobi2d np=3 N=32

结果：**PASS**，退出码 0。

```text
Iter 0: diff = 1.41421356e+00
Iter 100: diff = 2.88956709e-02
Total elapsed time: 0.0046 seconds
Grid size: 32 x 32, 3 processes, iterations: 200
Max abs diff vs serial Jacobi: 0.000e+00
RESULT: PASS
```

## allreduce_bench np=4

结果：**PASS**，退出码 0。

```text
=== Allreduce Benchmark (4 processes) ===
count=1 doubles  iters=100  avg=1.524047e-05 s  alg=0.52 MB/s  bus=0.79 MB/s  check=ok
count=1024 doubles  iters=50  avg=3.407626e-05 s  alg=240.40 MB/s  bus=360.60 MB/s  check=ok
count=65536 doubles  iters=20  avg=2.787754e-04 s  alg=1880.68 MB/s  bus=2821.02 MB/s  check=ok
RESULT: PASS
```

## blocking_vs_nonblocking np=2

结果：**PASS**，退出码 0。

```text
=== Blocking then compute ===
Time: 0.007004 seconds (20 repeats, 262144 ints)
=== Nonblocking overlapped with compute ===
Time: 0.005731 seconds
Overlap is visible only when the compute is long enough to hide the transfer.
RESULT: PASS
```

## pingpong np=2

结果：**PASS**，退出码 0。

```text
=== Ping-pong (alpha + n*beta) ===
bytes=       8  iters= 200  one-way=4.458 us  bandwidth=1.79 MB/s  check=ok
bytes=    4096  iters= 100  one-way=6.184 us  bandwidth=662.31 MB/s  check=ok
bytes=  262144  iters=  40  one-way=45.547 us  bandwidth=5755.41 MB/s  check=ok
bytes= 2097152  iters=  10  one-way=242.530 us  bandwidth=8647.00 MB/s  check=ok
RESULT: PASS
```

## moe_alltoall np=1

结果：**PASS**，退出码 0。

```text
ranks=1 tokens=8 slots=15 combine_msgs=8 max_abs_diff=0.000e+00 rank0_token0=1,2,3,4
RESULT: PASS
```

## moe_alltoall np=3

结果：**PASS**，退出码 0。

```text
ranks=3 tokens=8 slots=45 combine_msgs=33 max_abs_diff=0.000e+00 rank0_token0=1,2,3,4
RESULT: PASS
```

## moe_alltoall np=4

结果：**PASS**，退出码 0。

```text
ranks=4 tokens=8 slots=60 combine_msgs=56 max_abs_diff=0.000e+00 rank0_token0=1,2,3,4
RESULT: PASS
```

## deepep_layout

结果：**PASS**，退出码 0。

```text
layout ranks=4 tokens=8 hidden=4 slots=60 combine_msgs=56 dispatch_doubles=240 combine_doubles=224 max_abs_diff=0.000e+00 inbox_max=17 cap_reject=16
switch_capacity(4,2,1,1)=2 kept=[0, 0, -1, 1]
padding alignment=4 isolated_pad=3 weighted_sum=4
RESULT: PASS
```

## rdma write ch04

结果：**SKIP**，服务端退出码 0，客户端退出码 0，服务端就绪 1。

```text
RESULT: SKIP no active RDMA device
RESULT: SKIP no active RDMA device
```

## verbs send/write/read ch08

结果：**SKIP**，服务端退出码 0，客户端退出码 0，服务端就绪 1。

```text
RESULT: SKIP no active RDMA device
RESULT: SKIP no active RDMA device
```

## mpi_basic np=2

结果：**PASS**，退出码 0。

```text
Hello from rank 0/2 backend=gloo device=cpu
Hello from rank 1/2 backend=gloo device=cpu
Rank 0: before allreduce tensor = 0.0
Rank 1: before allreduce tensor = 1.0
Rank 0: after allreduce tensor = 1.0, expected 1.0
Rank 1: after allreduce tensor = 1.0, expected 1.0
RESULT: PASS
```

## pytorch_ddp np=2

结果：**PASS**，退出码 0。

```text
[rank 1/2] backend=gloo device=cpu
[rank 0/2] backend=gloo device=cpu
[rank 0] first batch loss = 0.6074
[rank 1] weights synchronized
RESULT: PASS
```

## train.py np=1

结果：**PASS**，退出码 0。

```text
Starting training: rank=0 local_rank=0 world_size=1 backend=gloo
Epoch 1/5, Loss: 2.3400
Epoch 2/5, Loss: 2.4376
Epoch 3/5, Loss: 2.4009
Epoch 4/5, Loss: 2.3321
Epoch 5/5, Loss: 2.3594
RESULT: PASS
```

## train.py np=2

结果：**PASS**，退出码 0。

```text
Starting training: rank=1 local_rank=1 world_size=2 backend=gloo
Starting training: rank=0 local_rank=0 world_size=2 backend=gloo
Epoch 1/5, Loss: 2.5592
Epoch 2/5, Loss: 2.5160
Epoch 3/5, Loss: 2.4432
Epoch 4/5, Loss: 2.4604
Epoch 5/5, Loss: 2.2917
RESULT: PASS
```

## benchmark_compare np=2

结果：**PASS**，退出码 0。

```text
Warming up (5 iterations)...
Running benchmark (20 iterations, 4 MB)...

============================================================
  Benchmark Results: cpu-gloo
============================================================
  World Size:         2 processes
  Tensor Size:        4 MB
  Correctness:        True
  Avg Latency:        1.583 ms
  P95 Latency:        3.467 ms
  Alg Bandwidth:      2.796 GB/s
  Bus Bandwidth:      2.796 GB/s
============================================================

Results saved to /workspace/results/cpu_gloo_allreduce.json
RESULT: PASS
```

## nccl_rdma_demo np=2

结果：**PASS**，退出码 0。

```text
mode=cpu-gloo world_size=2 device=cpu
No CUDA device. This run does not exercise NCCL or RDMA.
value=3.0 expected=3.0
avg_latency_ms=0.893 alg_GBps=1.175 bus_GBps=1.175
RESULT: PASS
```

## moe_dispatch np=1

结果：**PASS**，退出码 0。

```text
backend=gloo ranks=1 tokens=8 slots=15 combine_msgs=8 max_abs_diff=0.000e+00 rank0_token0=1,2,3,4
RESULT: PASS
```

## moe_dispatch np=4

结果：**PASS**，退出码 0。

```text
backend=gloo ranks=4 tokens=8 slots=60 combine_msgs=56 max_abs_diff=0.000e+00 rank0_token0=1,2,3,4
RESULT: PASS
```

## collective_contract np=3

结果：**PASS**，退出码 0。

```text
backend=gloo ranks=3 allreduce_sum=6 bus_factor=1.3333 bus_bytes_per_rank=10.7 rank0_send_counts=[0, 2, 3] alltoall_zero_pairs=1
RESULT: PASS
```

## nccl_probe np=2

结果：**SKIP**，退出码 0。

```text
torch=2.15.0+cpu is_nccl_available=False cuda=False device_count=0 world=2
RESULT: SKIP NCCL collectives need a CUDA build of PyTorch and a visible GPU
```

## deepep_probe

结果：**SKIP**，退出码 0。

```text
torch=2.15.0+cpu cuda=False device_count=0
import deep_ep failed: ModuleNotFoundError: No module named 'deep_ep'
DeepEP 主线需要 Hopper 或更新的 GPU、CUDA 13.1+、带 CUDA 的 PyTorch 2.10+、NCCL 2.32.3+。V2.5 使用 EPBuffer 和 NCCL Gin，不再依赖 NVSHMEM。
RESULT: SKIP
```

## pytorchjob yaml

结果：**PASS**，退出码 0。

```text
ch09-kubernetes-pytorchjob/09-pytorchjob/pytorchjob.yaml: kind=PyTorchJob roles=['Master', 'Worker']
ch09-kubernetes-pytorchjob/09-pytorchjob/pytorchjob-cpu.yaml: kind=PyTorchJob roles=['Master', 'Worker']
RESULT: PASS
```

