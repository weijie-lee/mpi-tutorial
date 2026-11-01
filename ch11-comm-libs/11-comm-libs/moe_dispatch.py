#!/usr/bin/env python3
"""用 torch.distributed 的不等长 all_to_all 做 MoE dispatch 和 combine。

没有 GPU 时走 gloo。数值要和 deepep_layout.oracle 一致，rank 0 的 token 0 是 [1, 2, 3, 4]。
这是通信模式的核对，不是 NCCL 或 DeepEP 的计时。
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepep_layout as layout


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


STRIDE = 5 + layout.HIDDEN


def slots_for(rank, ntokens, world):
    buckets = [[] for _ in range(world)]
    num_experts = world * layout.LOCAL_EXPERTS
    for token in range(ntokens):
        hidden = layout.hidden_of(rank, token)
        for k in range(layout.TOPK):
            expert = layout.expert_of(rank, token, k, num_experts)
            if expert < 0:
                continue
            dest = expert // layout.LOCAL_EXPERTS
            buckets[dest].append([float(rank), float(token), float(expert), float(k), layout.weight_of(k), *hidden])
    return buckets


def pack(buckets):
    counts = [len(rows) for rows in buckets]
    flat = []
    for rows in buckets:
        for row in rows:
            flat.extend(row)
    if not flat:
        flat = [0.0]
    tensor = torch.tensor(flat, dtype=torch.float64)
    if sum(counts) == 0:
        tensor = tensor[:0]
    return counts, tensor


def exchange(send_counts, send_tensor, device):
    count_tensor = torch.tensor(send_counts, dtype=torch.int64, device=device)
    recv_count_tensor = torch.empty_like(count_tensor)
    dist.all_to_all_single(recv_count_tensor, count_tensor)
    recv_counts = [int(v) for v in recv_count_tensor.tolist()]
    send_splits = [c * STRIDE for c in send_counts]
    recv_splits = [c * STRIDE for c in recv_counts]
    recv = torch.empty(sum(recv_splits), dtype=torch.float64, device=device)
    payload = send_tensor.to(device)
    if payload.numel() != sum(send_splits):
        raise RuntimeError(f"send numel {payload.numel()} != {sum(send_splits)}")
    dist.all_to_all_single(recv, payload, recv_splits, send_splits)
    return recv_counts, recv.cpu()


def rows_from(recv_counts, recv):
    rows = []
    offset = 0
    for count in recv_counts:
        for _ in range(count):
            chunk = recv[offset : offset + STRIDE].tolist()
            offset += STRIDE
            rows.append(
                {
                    "src": int(chunk[0]),
                    "token": int(chunk[1]),
                    "expert": int(chunk[2]),
                    "k": int(chunk[3]),
                    "weight": chunk[4],
                    "hidden": chunk[5:],
                }
            )
    if offset != recv.numel():
        raise RuntimeError("payload tail")
    return rows


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ntokens = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if ntokens < 2 or ntokens > 100000:
        print("RESULT: FAIL")
        return 1
    backend, device = init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    try:
        send_counts, send_tensor = pack(slots_for(rank, ntokens, world))
        recv_counts, recv = exchange(send_counts, send_tensor, device)
        incoming = rows_from(recv_counts, recv)
        expected_rows = layout.inbox_for(rank, world, ntokens)
        if len(incoming) != len(expected_rows):
            raise RuntimeError(f"recv {len(incoming)} expected {len(expected_rows)}")

        partial = {}
        for row in incoming:
            if row["expert"] // layout.LOCAL_EXPERTS != rank:
                raise RuntimeError("expert delivered to the wrong rank")
            acc = partial.setdefault((row["src"], row["token"]), [0.0] * layout.HIDDEN)
            scale = row["weight"] * (row["expert"] + 1)
            for d in range(layout.HIDDEN):
                acc[d] += scale * row["hidden"][d]
        expect_pairs = len({(row["src"], row["token"]) for row in expected_rows})
        if len(partial) != expect_pairs:
            raise RuntimeError(f"combine msgs {len(partial)} expected {expect_pairs}")

        back_buckets = [[] for _ in range(world)]
        for (src, token), vec in partial.items():
            back_buckets[src].append(
                [float(src), float(token), -1.0, -1.0, 1.0, *vec]
            )
        back_counts, back_tensor = pack(back_buckets)
        back_recv_counts, back_recv = exchange(back_counts, back_tensor, device)
        returned = rows_from(back_recv_counts, back_recv)

        result = [[layout.BIAS] * layout.HIDDEN for _ in range(ntokens)]
        for row in returned:
            if row["src"] != rank:
                raise RuntimeError("combine returned another rank's token")
            for d in range(layout.HIDDEN):
                result[row["token"]][d] += row["hidden"][d]

        reference = layout.oracle(rank, ntokens, world)
        closed = layout.token0_closed_form(rank)
        max_diff = 0.0
        for token in range(ntokens):
            for d in range(layout.HIDDEN):
                max_diff = max(max_diff, abs(result[token][d] - reference[token][d]))
        for d in range(layout.HIDDEN):
            max_diff = max(max_diff, abs(result[0][d] - closed[d]))
        if rank == 0:
            for d in range(layout.HIDDEN):
                max_diff = max(max_diff, abs(result[0][d] - (float(d) + 1.0)))
        if max_diff != 0.0:
            raise RuntimeError(f"max_abs_diff {max_diff}")

        sent = torch.tensor([float(sum(send_counts)), float(len(partial)), max_diff], dtype=torch.float64, device=device)
        dist.all_reduce(sent, op=dist.ReduceOp.SUM)
        # max_diff was summed; it must still be 0. Recompute a MAX via a second tensor.
        diff_t = torch.tensor([max_diff], dtype=torch.float64, device=device)
        dist.all_reduce(diff_t, op=dist.ReduceOp.MAX)
        slots = int(sent[0].item())
        msgs = int(sent[1].item())
        expect_slots, expect_msgs = layout.global_counts(world, ntokens)
        if slots != expect_slots or msgs != expect_msgs or diff_t.item() != 0.0 or msgs >= slots:
            raise RuntimeError(f"global slots={slots} msgs={msgs} diff={diff_t.item()}")
        ok = True
    except Exception as exc:
        print(f"rank {rank} {exc}")
        ok = False

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    failed = flag.item() != 1
    if rank == 0:
        if failed:
            print(f"backend={backend} ranks={world} tokens={ntokens}")
            print("RESULT: FAIL")
        else:
            shown = ",".join(f"{result[0][d]:.0f}" for d in range(layout.HIDDEN))
            print(
                f"backend={backend} ranks={world} tokens={ntokens} slots={slots} "
                f"combine_msgs={msgs} max_abs_diff={diff_t.item():.3e} rank0_token0={shown}"
            )
            print("RESULT: PASS")
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
