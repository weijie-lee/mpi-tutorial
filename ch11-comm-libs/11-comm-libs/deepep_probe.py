#!/usr/bin/env python3
"""检查本机能不能导入 deep_ep。导入失败或没有 GPU 时 SKIP，不编造带宽。"""

import sys

import torch


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    try:
        import deep_ep
    except Exception as exc:
        print(f"import deep_ep failed: {type(exc).__name__}: {exc}")
        print(
            "DeepEP 主线需要 Hopper 或更新的 GPU、CUDA 13.1+、带 CUDA 的 PyTorch 2.10+、"
            "NCCL 2.32.3+。V2.5 使用 EPBuffer 和 NCCL Gin，不再依赖 NVSHMEM。"
        )
        print("RESULT: SKIP")
        return 0

    print(f"deep_ep={getattr(deep_ep, '__file__', deep_ep)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("RESULT: SKIP deep_ep imported but there is no CUDA device, EPBuffer was not created")
        return 0
    print("RESULT: SKIP deep_ep is installed and a GPU is visible; this probe does not launch dispatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
