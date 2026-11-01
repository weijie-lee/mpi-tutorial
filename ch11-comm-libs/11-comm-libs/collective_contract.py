#!/usr/bin/env python3
"""核对 AllReduce、AllGather、ReduceScatter 和不等长 AlltoAll 的数值约定。

有 CUDA 且 PyTorch 带 NCCL 时走 nccl，否则走 gloo。两边应满足同一组等式。
NCCL 2.28 的 ncclAlltoAll 要求每个目的地的 count 相同；这里的最后一项故意用
不同的长度，对应 grouped ncclSend/ncclRecv 或 MPI_Alltoallv。
"""

import os
import sys

import torch
import torch.distributed as dist


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


def init_distributed():
    bootstrap_env()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0 and dist.is_nccl_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(index)
        dist.init_process_group(backend="nccl", init_method="env://")
        return "nccl", torch.device("cuda", index)
    dist.init_process_group(backend="gloo", init_method="env://")
    return "gloo", torch.device("cpu")


def pair_count(src, dst):
    if (src + dst) % 5 == 0:
        return 0
    return (src + dst) % 3 + 1


def main():
    sys.stdout.reconfigure(line_buffering=True)
    backend, device = init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    ok = True

    values = torch.tensor([float(rank + 1)], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    expect_sum = world * (world + 1) / 2.0
    if values.item() != expect_sum:
        ok = False
        print(f"rank {rank} allreduce {values.item()} expected {expect_sum}")

    shard = torch.tensor([float(rank), float(rank + 10)], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(gathered, shard)
    for src, piece in enumerate(gathered):
        if piece[0].item() != float(src) or piece[1].item() != float(src + 10):
            ok = False
            print(f"rank {rank} allgather src {src} got {piece.tolist()}")

    width = 4
    ones = torch.ones(world * width, dtype=torch.float64, device=device)
    reduced = torch.empty(width, dtype=torch.float64, device=device)
    dist.reduce_scatter_tensor(reduced, ones)
    if not torch.equal(reduced, torch.full((width,), float(world), dtype=torch.float64, device=device)):
        ok = False
        print(f"rank {rank} reduce_scatter {reduced.tolist()}")

    send_counts = [pair_count(rank, dst) for dst in range(world)]
    recv_counts_t = torch.empty(world, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv_counts_t, torch.tensor(send_counts, dtype=torch.int64, device=device))
    recv_counts = [int(v) for v in recv_counts_t.tolist()]
    chunks = []
    for dst, count in enumerate(send_counts):
        chunks.append(torch.full((count,), float(rank * 100 + dst), dtype=torch.float64))
    send = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)
    send = send.to(device)
    recv = torch.empty(sum(recv_counts), dtype=torch.float64, device=device)
    dist.all_to_all_single(recv, send, recv_counts, send_counts)
    recv_cpu = recv.cpu()
    offset = 0
    for src, count in enumerate(recv_counts):
        piece = recv_cpu[offset : offset + count]
        offset += count
        expected_count = pair_count(src, rank)
        expected_value = float(src * 100 + rank)
        if count != expected_count or (count > 0 and not torch.equal(piece, torch.full((count,), expected_value))):
            ok = False
            print(f"rank {rank} alltoall from {src} count {count} values {piece.tolist()}")
    if offset != recv_cpu.numel():
        ok = False

    nbytes = values.element_size() * values.numel()
    bus_factor = (2.0 * (world - 1) / world) if world else 0.0
    if world == 1 and bus_factor != 0.0:
        ok = False
    if world > 1 and not (0.0 < bus_factor < 2.0):
        ok = False

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    failed = flag.item() != 1
    if rank == 0:
        zero_pairs = sum(1 for src in range(world) for dst in range(world) if pair_count(src, dst) == 0)
        print(
            f"backend={backend} ranks={world} allreduce_sum={values.item():.0f} "
            f"bus_factor={bus_factor:.4f} bus_bytes_per_rank={nbytes * bus_factor:.1f} "
            f"rank0_send_counts={send_counts} alltoall_zero_pairs={zero_pairs}"
        )
        print("RESULT: FAIL" if failed else "RESULT: PASS")
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
