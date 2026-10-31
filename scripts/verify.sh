#!/usr/bin/env bash
# 编译并运行不依赖 GPU、RDMA 网卡和 Kubernetes 的实验，把输出写到 results/VERIFICATION.md。
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/build"
mkdir -p "$BUILD" "$ROOT/results"
REPORT="$ROOT/results/VERIFICATION.md"
LOG="$(mktemp)"
FAILS=0
ROWS=""

export OMPI_MCA_btl="${OMPI_MCA_btl:-tcp,self}"
export OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_btl_tcp_if_include:-lo}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
MPIRUN=(mpirun --oversubscribe --bind-to none)

cleanup() {
    rm -f "$LOG"
}
trap cleanup EXIT

note() {
    printf '%s\n' "$1" >> "$REPORT"
}

run_logged() {
    local name="$1"
    shift
    "$@" >"$LOG" 2>&1
    local status=$?
    local result="PASS"
    if grep -q "RESULT: FAIL" "$LOG"; then
        result="FAIL"
    elif grep -q "RESULT: SKIP" "$LOG"; then
        result="SKIP"
    elif [[ "$status" -ne 0 ]]; then
        result="FAIL"
    elif ! grep -q "RESULT: PASS" "$LOG"; then
        result="FAIL"
    fi
    if [[ "$result" == "FAIL" ]]; then
        FAILS=$((FAILS + 1))
    fi
    ROWS="${ROWS}| ${name} | ${result} |"$'\n'
    {
        echo "## ${name}"
        echo
        echo "结果：**${result}**，退出码 ${status}。"
        echo
        echo '```text'
        cat "$LOG"
        echo '```'
        echo
    } >> "$REPORT"
    printf '%s %s\n' "$name" "$result"
}

compile_c() {
    local name="$1"
    shift
    mpicc -O2 -Wall -Wextra -Werror -std=c11 -o "$BUILD/$name" "$@" >"$LOG" 2>&1
    local status=$?
    if [[ "$status" -ne 0 ]]; then
        FAILS=$((FAILS + 1))
        ROWS="${ROWS}| compile ${name} | FAIL |"$'\n'
        {
            echo "## compile ${name}"
            echo
            echo "结果：**FAIL**，退出码 ${status}。"
            echo
            echo '```text'
            cat "$LOG"
            echo '```'
            echo
        } >> "$REPORT"
        printf 'compile %s FAIL\n' "$name"
        return 1
    fi
    return 0
}

compile_gcc() {
    local name="$1"
    shift
    gcc -O2 -Wall -Wextra -Werror -std=c11 -o "$BUILD/$name" "$@" >"$LOG" 2>&1
    local status=$?
    if [[ "$status" -ne 0 ]]; then
        FAILS=$((FAILS + 1))
        ROWS="${ROWS}| compile ${name} | FAIL |"$'\n'
        {
            echo "## compile ${name}"
            echo
            echo "结果：**FAIL**，退出码 ${status}。"
            echo
            echo '```text'
            cat "$LOG"
            echo '```'
            echo
        } >> "$REPORT"
        printf 'compile %s FAIL\n' "$name"
        return 1
    fi
    return 0
}

{
    echo "# 验证结果"
    echo
    echo "由 \`scripts/verify.sh\` 生成。数字来自这次实际运行，不是文档里的数量级示例。"
    echo
    echo "- 时间：$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- 主机：$(uname -srm)"
    echo "- C 编译器：$(gcc --version | head -1)"
    echo "- MPI：$(mpirun --version | head -1)"
    python3 - << 'PY'
import torch
print(f"- PyTorch：{torch.__version__}，MPI 后端 {torch.distributed.is_mpi_available()}，CUDA {torch.cuda.is_available()}")
PY
    echo "- GPU：无可用 CUDA 设备，NCCL 示例未运行。"
    echo "- RDMA：本机没有 active 的 verbs 设备，内核也不能创建 soft-RoCE。相关程序应打印 SKIP。"
    echo "- Kubernetes：只检查了清单能被解析，没有提交到集群，也没有构建训练镜像。"
    echo
    echo "## 汇总"
    echo
    echo "| 实验 | 结果 |"
    echo "| --- | --- |"
} > "$REPORT"

compile_c hello ch01-intro/01-basics/hello.c || true
compile_c timing ch01-intro/01-basics/timing.c || true
compile_c error_handling ch01-intro/01-basics/error_handling.c || true
compile_c sendrecv ch02-core/02-core/sendrecv.c || true
compile_c deadlock ch02-core/02-core/deadlock.c || true
compile_c nonblocking ch02-core/02-core/nonblocking.c || true
compile_c collectives ch02-core/02-core/collectives.c || true
compile_c all-collectives ch02-core/02-core/all-collectives.c || true
compile_c pi_monte_carlo ch02-core/02-core/pi_monte_carlo.c || true
compile_c derived_type ch03-advanced/03-advanced/derived_type.c || true
compile_c cartesian ch03-advanced/03-advanced/cartesian.c || true
compile_c comm_split ch03-advanced/03-advanced/comm_split.c || true
compile_c rma_putget ch03-advanced/03-advanced/rma_putget.c || true
compile_c vector_indexed ch03-advanced/03-advanced/vector_indexed.c || true
compile_c jacobi2d ch06-applications/06-applications/jacobi2d.c -lm || true
compile_c allreduce_bench ch07-optimize/07-debug/allreduce_bench.c || true
compile_c blocking_vs_nonblocking ch07-optimize/07-debug/blocking_vs_nonblocking.c -lm || true
compile_c pingpong ch07-optimize/07-debug/pingpong.c || true
compile_gcc rdma_write_server ch04-hardware/04-hardware/rdma_write_server.c -libverbs || true
compile_gcc rdma_write_client ch04-hardware/04-hardware/rdma_write_client.c -libverbs || true
compile_gcc verbs_server ch08-rdma-verbs/08-rdma-verbs/server.c -libverbs || true
compile_gcc verbs_client ch08-rdma-verbs/08-rdma-verbs/client.c -libverbs || true

run_logged "hello np=4" "${MPIRUN[@]}" -np 4 "$BUILD/hello"
run_logged "timing np=2" "${MPIRUN[@]}" -np 2 "$BUILD/timing"
run_logged "error_handling np=2" "${MPIRUN[@]}" -np 2 "$BUILD/error_handling"
run_logged "sendrecv np=2" "${MPIRUN[@]}" -np 2 "$BUILD/sendrecv"
run_logged "deadlock safe np=2" "${MPIRUN[@]}" -np 2 "$BUILD/deadlock"
run_logged "nonblocking np=3" "${MPIRUN[@]}" -np 3 "$BUILD/nonblocking"
run_logged "collectives np=3" "${MPIRUN[@]}" -np 3 "$BUILD/collectives"
run_logged "all-collectives np=5" "${MPIRUN[@]}" -np 5 "$BUILD/all-collectives"
run_logged "pi_monte_carlo np=4" "${MPIRUN[@]}" -np 4 "$BUILD/pi_monte_carlo" 200000
run_logged "derived_type np=2" "${MPIRUN[@]}" -np 2 "$BUILD/derived_type"
run_logged "cartesian np=6" "${MPIRUN[@]}" -np 6 "$BUILD/cartesian"
run_logged "comm_split np=5" "${MPIRUN[@]}" -np 5 "$BUILD/comm_split"
run_logged "rma_putget np=2" "${MPIRUN[@]}" -np 2 "$BUILD/rma_putget"
run_logged "vector_indexed np=2" "${MPIRUN[@]}" -np 2 "$BUILD/vector_indexed"
run_logged "jacobi2d np=4 N=32" "${MPIRUN[@]}" -np 4 "$BUILD/jacobi2d" 32 300
run_logged "jacobi2d np=3 N=32" "${MPIRUN[@]}" -np 3 "$BUILD/jacobi2d" 32 200
run_logged "allreduce_bench np=4" "${MPIRUN[@]}" -np 4 "$BUILD/allreduce_bench"
run_logged "blocking_vs_nonblocking np=2" "${MPIRUN[@]}" -np 2 "$BUILD/blocking_vs_nonblocking"
run_logged "pingpong np=2" "${MPIRUN[@]}" -np 2 "$BUILD/pingpong"

rdma_pair() {
    local name="$1"
    local server="$2"
    local client="$3"
    local port="$4"
    : >"$LOG"
    set +e
    "$server" "$port" >"$BUILD/server.log" 2>&1 &
    local spid=$!
    local ready=0
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if grep -q -E "LISTENING|RESULT:" "$BUILD/server.log"; then
            ready=1
            break
        fi
        sleep 0.1
    done
    "$client" 127.0.0.1 "$port" >"$BUILD/client.log" 2>&1
    local client_status=$?
    wait "$spid"
    local server_status=$?
    cat "$BUILD/server.log" "$BUILD/client.log" >"$LOG"
    local result="PASS"
    if grep -q "RESULT: FAIL" "$LOG"; then
        result="FAIL"
    elif grep -q "RESULT: SKIP" "$BUILD/server.log" && grep -q "RESULT: SKIP" "$BUILD/client.log"; then
        result="SKIP"
    elif grep -q "RESULT: PASS" "$BUILD/server.log" && grep -q "RESULT: PASS" "$BUILD/client.log"; then
        result="PASS"
    else
        result="FAIL"
    fi
    if [[ "$result" == "FAIL" ]]; then
        FAILS=$((FAILS + 1))
    fi
    ROWS="${ROWS}| ${name} | ${result} |"$'\n'
    {
        echo "## ${name}"
        echo
        echo "结果：**${result}**，服务端退出码 ${server_status}，客户端退出码 ${client_status}，服务端就绪 ${ready}。"
        echo
        echo '```text'
        cat "$LOG"
        echo '```'
        echo
    } >> "$REPORT"
    printf '%s %s\n' "$name" "$result"
}

rdma_pair "rdma write ch04" "$BUILD/rdma_write_server" "$BUILD/rdma_write_client" 18615
rdma_pair "verbs send/write/read ch08" "$BUILD/verbs_server" "$BUILD/verbs_client" 18616

export MASTER_PORT=29531
run_logged "mpi_basic np=2" "${MPIRUN[@]}" -np 2 python3 ch05-stack/05-pytorch/mpi_basic.py
export MASTER_PORT=29532
run_logged "pytorch_ddp np=2" "${MPIRUN[@]}" -np 2 python3 ch05-stack/05-pytorch/pytorch_ddp_mpi.py
export MASTER_PORT=29533
run_logged "train.py np=1" python3 ch09-kubernetes-pytorchjob/09-pytorchjob/train.py
export MASTER_PORT=29534
run_logged "train.py np=2" "${MPIRUN[@]}" -np 2 python3 ch09-kubernetes-pytorchjob/09-pytorchjob/train.py
export MASTER_PORT=29535
run_logged "benchmark_compare np=2" "${MPIRUN[@]}" -np 2 python3 ch10-fullstack-observe/10-fullstack-observe/benchmark_compare.py \
    --iterations 20 --tensor_mb 4 --warmup 5 --output "$ROOT/results/cpu_gloo_allreduce.json"
export MASTER_PORT=29536
run_logged "nccl_rdma_demo np=2" "${MPIRUN[@]}" -np 2 python3 ch10-fullstack-observe/10-fullstack-observe/pytorch_nccl_rdma_demo.py \
    --tensor_mb 1 --iterations 4

set +e
python3 - << 'PY' >"$LOG" 2>&1
import pathlib
import yaml
paths = [
    "ch09-kubernetes-pytorchjob/09-pytorchjob/pytorchjob.yaml",
    "ch09-kubernetes-pytorchjob/09-pytorchjob/pytorchjob-cpu.yaml",
]
for path in paths:
    data = yaml.safe_load(pathlib.Path(path).read_text())
    reps = data["spec"]["pytorchReplicaSpecs"]
    print(f"{path}: kind={data['kind']} roles={list(reps)}")
    for role, spec in reps.items():
        containers = spec["template"]["spec"]["containers"]
        assert containers[0]["name"] == "pytorch"
        assert "volumeMounts" not in containers[0] or isinstance(containers[0]["volumeMounts"], list)
print("RESULT: PASS")
PY
yaml_status=$?
if [[ "$yaml_status" -ne 0 ]] || ! grep -q "RESULT: PASS" "$LOG"; then
    FAILS=$((FAILS + 1))
    yaml_result="FAIL"
else
    yaml_result="PASS"
fi
ROWS="${ROWS}| pytorchjob yaml | ${yaml_result} |"$'\n'
{
    echo "## pytorchjob yaml"
    echo
    echo "结果：**${yaml_result}**，退出码 ${yaml_status}。"
    echo
    echo '```text'
    cat "$LOG"
    echo '```'
    echo
} >> "$REPORT"

# 把汇总表插到占位标题后面。先写到临时文件再替换。
python3 - "$REPORT" "$ROWS" "$FAILS" << 'PY'
import sys
from pathlib import Path
path, rows, fails = sys.argv[1:]
text = Path(path).read_text()
table = "".join(line + "\n" for line in rows.splitlines() if line.strip())
summary = f"失败数：{fails}\n\n| 实验 | 结果 |\n| --- | --- |\n{table}\n"
text = text.replace("| 实验 | 结果 |\n| --- | --- |\n", summary, 1)
Path(path).write_text(text)
PY

echo "Wrote $REPORT"
echo "Failures: $FAILS"
exit "$FAILS"
