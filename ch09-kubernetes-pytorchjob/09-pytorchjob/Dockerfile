# 训练镜像。PyTorchJob 每个 Pod 直接执行 python，不需要在容器里再套 mpirun。
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY train.py /workspace/train.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "/workspace/train.py"]
