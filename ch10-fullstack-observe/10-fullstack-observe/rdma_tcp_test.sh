#!/usr/bin/env bash
# 在同一组参数下各跑一次 AllReduce。
# 有 GPU 时第二次设置 NCCL_IB_DISABLE=1，用来和默认 NCCL 路径对比。
# 没有 GPU 时只跑 cpu-gloo，并说明这次没有 RDMA/TCP 网卡对比。
set -euo pipefail

cd "$(dirname "$0")"
OUT_DIR="${1:-./benchmark-out}"
NP="${NP:-2}"
ITERS="${ITERS:-10}"
MB="${MB:-4}"
mkdir -p "$OUT_DIR"

MPI=(mpirun --oversubscribe --bind-to none -np "$NP")
if ! command -v mpirun >/dev/null; then
    echo "mpirun not found" >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29701}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

"${MPI[@]}" python3 benchmark_compare.py \
    --iterations "$ITERS" --tensor_mb "$MB" --warmup 2 \
    --output "$OUT_DIR/default.json"

if python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    export MASTER_PORT="${MASTER_PORT_TCP:-29702}"
    NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0 \
        "${MPI[@]}" python3 benchmark_compare.py \
        --iterations "$ITERS" --tensor_mb "$MB" --warmup 2 \
        --output "$OUT_DIR/tcp.json"
    python3 - <<PY
import json
a=json.load(open("$OUT_DIR/default.json"))
b=json.load(open("$OUT_DIR/tcp.json"))
print(f"default mode={a['mode']} alg={a['avg_alg_bandwidth_GBps']} GB/s")
print(f"forced  mode={b['mode']} alg={b['avg_alg_bandwidth_GBps']} GB/s")
PY
else
    echo "CUDA is not available. Saved $OUT_DIR/default.json as cpu-gloo."
    echo "RDMA and TCP numbers are not measured on this machine."
fi
