"""
PyTorch 分布式训练脚本
支持 Kubernetes PyTorchJob 环境
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    """简单模型用于测试"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        return self.fc(x)


def train(rank, world_size, epochs=5):
    """训练函数"""
    setup(rank, world_size)
    
    # 创建模型并移到当前 GPU
    model = SimpleModel().cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # 优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 模拟数据
    batch_size = 32
    for epoch in range(epochs):
        # 模拟一个 batch
        data = torch.randn(batch_size, 784).cuda(rank)
        target = torch.randint(0, 10, (batch_size,)).cuda(rank)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    cleanup()


def main():
    # 从环境变量获取进程信息
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"Starting training: Rank={rank}, WorldSize={world_size}")
    
    train(rank, world_size)


if __name__ == '__main__':
    main()
