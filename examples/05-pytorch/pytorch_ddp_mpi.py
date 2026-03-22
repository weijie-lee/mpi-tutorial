"""
PyTorch DDP 分布式训练示例，使用 MPI 启动 + NCCL 通信
运行示例（单节点4GPU）:
    mpirun -np 4 python pytorch_ddp_mpi.py
多节点（两个节点各4GPU）:
    mpirun -np 8 -H node1:4,node2:4 python pytorch_ddp_mpi.py
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleDataset(Dataset):
    """简单的随机数据集"""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim
        self.data = torch.randn(size, input_dim)
        self.target = (self.data.sum(dim=1) * 0.5).unsqueeze(1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def setup():
    """初始化分布式环境"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    # 每个进程绑定到一张GPU
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
    return rank


def main():
    rank = setup()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    if rank == 0:
        print(f"Starting DDP training with {world_size} processes")

    # 创建数据集和分布式采样器
    dataset = SimpleDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 创建模型并包装为DDP
    model = nn.Linear(10, 1).to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[gpu_id])
    else:
        model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 训练一个epoch
    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()  # DDP 在这里自动调用NCCL做梯度allreduce
        optimizer.step()

        if step % 10 == 0 and rank == 0:
            print(f"Step {step}, loss = {loss.item():.4f}")

    if rank == 0:
        print("Training finished!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
