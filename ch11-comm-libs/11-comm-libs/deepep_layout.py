#!/usr/bin/env python3
"""MoE dispatch/combine 的数据移动参考实现。

这里核对的是专家并行里 token 怎么分组、怎么合并，不是 DeepEP 的 CUDA kernel，
也不产生 NCCL 或 RDMA 带宽。DeepEP 在没有 GPU 的机器上由 deepep_probe.py 记为 SKIP。
"""

import math
import sys

HIDDEN = 4
TOPK = 2
LOCAL_EXPERTS = 2
BIAS = 1.0


def expert_of(rank, token, k, num_experts, local_experts=LOCAL_EXPERTS):
    """教程里固定的路由。token 0 的两个专家都在本 rank；token 1 的第二个槽位无效。"""
    if token == 0:
        return rank * local_experts + k
    if token == 1 and k == 1:
        return -1
    e0 = (rank * 17 + token * 3) % num_experts
    expert = (rank * 17 + token * 3 + k * 5) % num_experts
    if k == 1 and expert == e0:
        expert = (expert + 1) % num_experts
    return expert


def weight_of(k):
    return 0.5 if k == 0 else 0.25


def hidden_of(rank, token):
    return [float(rank * 1000 + token * 10 + d) for d in range(HIDDEN)]


def token0_closed_form(rank):
    """token 0 两个专家都在本 rank，加权专家输出有闭式。"""
    scale = 1.5 * rank + 1.0
    return [scale * (rank * 1000 + d) + BIAS for d in range(HIDDEN)]


def oracle(rank, ntokens, world):
    num_experts = world * LOCAL_EXPERTS
    out = [[BIAS] * HIDDEN for _ in range(ntokens)]
    for token in range(ntokens):
        hidden = hidden_of(rank, token)
        for k in range(TOPK):
            expert = expert_of(rank, token, k, num_experts)
            if expert < 0:
                continue
            scale = weight_of(k) * (expert + 1)
            for d in range(HIDDEN):
                out[token][d] += scale * hidden[d]
    return out


def iter_slots(world, ntokens):
    num_experts = world * LOCAL_EXPERTS
    for src in range(world):
        for token in range(ntokens):
            hidden = hidden_of(src, token)
            for k in range(TOPK):
                expert = expert_of(src, token, k, num_experts)
                if expert < 0:
                    continue
                dest = expert // LOCAL_EXPERTS
                yield src, token, k, expert, dest, weight_of(k), hidden


def global_counts(world, ntokens):
    slots = 0
    pairs = set()
    for src, token, _k, _expert, dest, _weight, _hidden in iter_slots(world, ntokens):
        slots += 1
        pairs.add((src, token, dest))
    return slots, len(pairs)


def inbox_for(dest, world, ntokens):
    rows = []
    for src, token, k, expert, owner, weight, hidden in iter_slots(world, ntokens):
        if owner == dest:
            rows.append(
                {
                    "src": src,
                    "token": token,
                    "k": k,
                    "expert": expert,
                    "weight": weight,
                    "hidden": hidden,
                }
            )
    return rows


def exchange_reference(world, ntokens):
    """单进程模拟各 rank 的接收、本地加权求和，以及 combine 后的 token。"""
    inboxes = [inbox_for(dest, world, ntokens) for dest in range(world)]
    outputs = [[[BIAS] * HIDDEN for _ in range(ntokens)] for _ in range(world)]
    return_msgs = 0
    slots = 0
    for inbox in inboxes:
        slots += len(inbox)
        partial = {}
        for row in inbox:
            acc = partial.setdefault((row["src"], row["token"]), [0.0] * HIDDEN)
            scale = row["weight"] * (row["expert"] + 1)
            for d in range(HIDDEN):
                acc[d] += scale * row["hidden"][d]
        return_msgs += len(partial)
        for (src, token), vec in partial.items():
            for d in range(HIDDEN):
                outputs[src][token][d] += vec[d]
    return outputs, slots, return_msgs


def align_up(count, alignment):
    if alignment <= 1:
        return count
    return (count + alignment - 1) // alignment * alignment


def reduce_rows(rows, alignment):
    """按本地专家分块并补齐。补齐行是 0，不进入 combine。"""
    groups = [[] for _ in range(LOCAL_EXPERTS)]
    for row in rows:
        groups[row["expert"] % LOCAL_EXPERTS].append(row)
    partial = {}
    pad = 0
    for group in groups:
        padded = align_up(len(group), alignment)
        pad += padded - len(group)
        for row in group:
            acc = partial.setdefault((row["src"], row["token"]), [0.0] * HIDDEN)
            scale = row["weight"] * (row["expert"] + 1)
            for d in range(HIDDEN):
                acc[d] += scale * row["hidden"][d]
    return partial, pad


def switch_capacity(num_tokens, num_experts, topk, factor):
    """Switch / GShard 常用的每个专家容量。这是进 dispatch 之前的丢 token 策略。"""
    return math.ceil(num_tokens / num_experts * topk * factor)


def apply_capacity(indices, num_experts, capacity):
    """每个专家保留先到的 capacity 个 token，其余下标改成 -1。"""
    used = [0] * num_experts
    kept = []
    for expert in indices:
        if expert < 0 or used[expert] >= capacity:
            kept.append(-1)
            continue
        used[expert] += 1
        kept.append(expert)
    return kept


def _close(a, b):
    return abs(a - b) == 0.0


def check_layout(world, ntokens):
    outputs, slots, return_msgs = exchange_reference(world, ntokens)
    expect_slots, expect_msgs = global_counts(world, ntokens)
    if slots != expect_slots or return_msgs != expect_msgs:
        raise AssertionError(f"counts {slots},{return_msgs} != {expect_slots},{expect_msgs}")
    if return_msgs >= slots:
        raise AssertionError("token 0 的两个专家在同一 rank，combine 消息数应小于 dispatch 槽位数")
    max_diff = 0.0
    for rank in range(world):
        closed = token0_closed_form(rank)
        reference = oracle(rank, ntokens, world)
        for d in range(HIDDEN):
            max_diff = max(max_diff, abs(outputs[rank][0][d] - closed[d]))
            if not _close(outputs[rank][0][d], closed[d]):
                raise AssertionError(f"rank {rank} token0 dim {d}")
            if not _close(reference[0][d], closed[d]):
                raise AssertionError(f"oracle token0 rank {rank} dim {d}")
        for token in range(ntokens):
            for d in range(HIDDEN):
                diff = abs(outputs[rank][token][d] - reference[token][d])
                max_diff = max(max_diff, diff)
                if diff != 0.0:
                    raise AssertionError(f"mismatch rank {rank} token {token} dim {d}")
    if world > 0 and not _close(outputs[0][0][0], 1.0):
        raise AssertionError("rank 0 token 0 dim 0 手算结果是 1")
    return slots, return_msgs, max_diff


def check_padding():
    row = {
        "src": 1,
        "token": 2,
        "k": 0,
        "expert": 3,
        "weight": 0.5,
        "hidden": [2.0, 0.0, 0.0, 0.0],
    }
    partial, pad = reduce_rows([row], alignment=4)
    if pad != 3:
        raise AssertionError(f"pad {pad}")
    # 0.5 * (3+1) * 2 = 4，补齐的三行必须是 0
    if partial[(1, 2)][0] != 4.0:
        raise AssertionError(partial[(1, 2)])
    partial_exact, pad0 = reduce_rows([row], alignment=1)
    if pad0 != 0 or partial_exact[(1, 2)][0] != 4.0:
        raise AssertionError("alignment 1 changed the sum")


def check_capacity(world, ntokens):
    lengths = [len(inbox_for(dest, world, ntokens)) for dest in range(world)]
    limit = max(lengths)
    if limit <= 0 or any(n > limit for n in lengths):
        raise AssertionError(lengths)
    if not any(n > limit - 1 for n in lengths):
        raise AssertionError("cap-1 should reject the fullest rank")
    return limit


def check_switch():
    capacity = switch_capacity(4, 2, 1, 1.0)
    dropped = apply_capacity([0, 0, 0, 1], 2, capacity)
    if capacity != 2 or dropped != [0, 0, -1, 1]:
        raise AssertionError((capacity, dropped))
    if switch_capacity(4, 2, 1, 1.5) != 3:
        raise AssertionError("factor 1.5")
    if apply_capacity([0, 0, 0, 1], 2, 1) != [0, -1, -1, 1]:
        raise AssertionError("capacity 1")
    return capacity, dropped


def main():
    sys.stdout.reconfigure(line_buffering=True)
    try:
        slots, msgs, diff = check_layout(4, 8)
        for world in (1, 3):
            check_layout(world, 8)
        check_padding()
        limit = check_capacity(4, 8)
        capacity, dropped = check_switch()
    except AssertionError as exc:
        print(f"layout check failed: {exc}")
        print("RESULT: FAIL")
        return 1
    print(
        f"layout ranks=4 tokens=8 hidden={HIDDEN} slots={slots} combine_msgs={msgs} "
        f"dispatch_doubles={slots * HIDDEN} combine_doubles={msgs * HIDDEN} "
        f"max_abs_diff={diff:.3e} inbox_max={limit} cap_reject={limit - 1}"
    )
    print(f"switch_capacity(4,2,1,1)={capacity} kept={dropped}")
    print("padding alignment=4 isolated_pad=3 weighted_sum=4")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
