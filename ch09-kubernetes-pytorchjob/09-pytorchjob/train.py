"""PyTorchJob 里的训练脚本。

每个 Pod 一个进程。Operator 注入 RANK、WORLD_SIZE、LOCAL_RANK、MASTER_ADDR、MASTER_PORT。
一个 Pod 只申请 1 张 GPU 时，容器里的设备号是 cuda:0，不能把全局 RANK 当作设备号。
没有 GPU 的环境用 gloo，方便在提交到集群之前先把脚本跑通。
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def bootstrap_env():
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)


def main():
    bootstrap_env()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if use_cuda:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", local_rank % torch.cuda.device_count())
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    print(f"Starting training: rank={rank} local_rank={local_rank} world_size={world_size} backend={backend}")
    dist.init_process_group(backend=backend, init_method="env://")

    model = SimpleModel().to(device)
    if use_cuda:
        model = DDP(model, device_ids=[device.index])
    else:
        model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = int(os.environ.get("EPOCHS", "5"))
    last_loss = None
    for epoch in range(epochs):
        data = torch.randn(32, 784, device=device)
        target = torch.randint(0, 10, (32,), device=device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {last_loss:.4f}")

    weight = next(model.parameters()).detach().reshape(-1)[0]
    reference = weight.clone()
    dist.broadcast(reference, src=0)
    ok = torch.tensor(
        [1 if last_loss is not None and abs(weight.item() - reference.item()) < 1e-5 else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("RESULT: PASS" if ok.item() == 1 else "RESULT: FAIL")
    dist.destroy_process_group()
    return 0 if ok.item() == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
