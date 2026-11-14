#!/usr/bin/env bash
# Correctness gate for the GPU (OpenMP target offload) variant.
#
# Real GPU run (Vanda):
#   module load nvhpc-hpcx-cuda12/26.1
#   make gpu MPICC=mpicc GPUFLAGS=-mp=gpu
#   bash scripts/test_gpu.sh
#
# Logic-only run on a machine without a target device (Atlas, CPU-only):
#   make gpu GPUFLAGS='-fopenmp -foffload=disable'
#   GOL_ALLOW_HOST_TARGET=1 bash scripts/test_gpu.sh
#   ^ verifies kernel/wrap/halo logic only. It is NOT a GPU measurement,
#     and the variant still reports "gpu" in JSON -- keep the two apart.
#
# Every case must match member A's serial kernel checksum, and the JSON
# must report variant=gpu with threads=1.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}
GPU_BIN=${GPU_BIN:-$ROOT/build/gpu}
SERIAL=${SERIAL:-$A_DIR/build/serial}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_OPTS=(${MPIRUN_OPTS:---oversubscribe})
CASES=${CASES:-"97x53x100x42 64x64x50x1 5x13x77x99 3x3x20x7 1x1x10x1"}
RANKS=${RANKS:-"1 2 4 8"}
COMMS=${COMMS:-"blocking nonblocking overlap rma"}

[ -x "$GPU_BIN" ] || { echo "missing $GPU_BIN (run: make gpu)" >&2; exit 1; }
[ -x "$SERIAL" ] || { echo "missing $SERIAL (run: make -C $A_DIR build/serial)" >&2; exit 1; }

field() {  # json on stdin -> field
    python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

pass=0 fail=0
for c in $CASES; do
    IFS=x read -r R C T S <<<"$c"
    ref=$("$SERIAL" --rows "$R" --cols "$C" --steps "$T" --seed "$S" | tail -n 1)
    refhash=$(printf '%s' "$ref" | field checksum_fnv1a64)
    for NP in $RANKS; do
        [ "$NP" -le "$R" ] || continue   # 1d needs at least one row per rank
        for M in $COMMS; do
            out=$("$MPIRUN" -np "$NP" "${MPIRUN_OPTS[@]}" "$GPU_BIN" \
                --rows "$R" --cols "$C" --steps "$T" --seed "$S" \
                --comm "$M" 2>/dev/null | tail -n 1)
            if [ -z "$out" ]; then
                echo "FAIL ${R}x${C} steps=$T seed=$S np=$NP comm=$M (no output)"
                fail=$((fail + 1)); continue
            fi
            h=$(printf '%s' "$out" | field checksum_fnv1a64)
            v=$(printf '%s' "$out" | field variant)
            th=$(printf '%s' "$out" | field threads)
            if [ "$h" = "$refhash" ] && [ "$v" = "gpu" ] && [ "$th" = "1" ]; then
                pass=$((pass + 1))
            else
                echo "FAIL ${R}x${C} steps=$T seed=$S np=$NP comm=$M" \
                     "(got $h/$v/$th want $refhash/gpu/1)"
                fail=$((fail + 1))
            fi
        done
    done
done
echo "GPU correctness: PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]
