"""
PyTorch MPI 后端基础示例
运行: mpirun -np 4 python mpi_basic.py
"""
import torch
import torch.distributed as dist


def main():
    # 使用MPI后端初始化分布式环境
    dist.init_process_group(backend='mpi')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Hello from rank {rank}/{world_size} on {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

    # 简单allreduce测试: 每个rank发送rank值，求和
    tensor = torch.tensor([rank], dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank}: allreduce result = {tensor.item()} (expected sum(0..{world_size-1}) = {world_size*(world_size-1)//2})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
