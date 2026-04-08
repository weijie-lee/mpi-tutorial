# 训练镜像 Dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 安装 MPI 和 NCCL
RUN apt-get update && apt-get install -y \
    openmpi \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制训练脚本
COPY train.py /workspace/

# 设置 MPI 环境
ENV PYTHONUNBUFFERED=1

# 启动命令（由 mpirun 覆盖）
CMD ["python", "/workspace/train.py"]
