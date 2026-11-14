#!/usr/bin/env bash
# GPU (OpenMP target offload) benchmark: same CSV contract as bench_atlas.sh,
# with the gpu variant as mode="gpu". Every row is checksum-gated against the
# serial kernel, so a wrong mapping is logged as a failure, never as a time.
#
# Usage (inside a GPU job, Vanda):
#   GRIDS="2048 4096" REPEAT=3 STEPS=300 bash scripts/bench_gpu.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}

GPU_BIN=${GPU_BIN:-$ROOT/build/gpu}
HYBRID_BIN=${HYBRID_BIN:-$ROOT/build/hybrid}
SERIAL=${SERIAL:-$A_DIR/build/serial}
OUT=${OUT:-$ROOT/results/gpu_results.csv}
REPEAT=${REPEAT:-3}
STEPS=${STEPS:-300}
SEED=${SEED:-20261026}
DENSITY=${DENSITY:-500000}
GRIDS=${GRIDS:-"2048 4096"}
RANKS=${RANKS:-"1"}
COMM=${COMM:-blocking}
CPU_THREADS=${CPU_THREADS:-0}   # >0 also measures the hybrid variant on CPU

MPIRUN=${MPIRUN:-mpirun}
MPIRUN_OPTS=(${MPIRUN_OPTS:---oversubscribe})
JOB=${PBS_JOBID:-local}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"
echo "mode,W,H,steps,ranks,threads,repeat,kernel_seconds,total_seconds,checksum,job_id,decomp,comm" > "$OUT"

capture() {  # json_path command...
    local json=$1
    shift
    "$@" 2>>"$TMP/stderr.log" | tail -n 1 >"$json"
}

emit() {  # json mode W H ranks threads repeat decomp comm
    python3 - "$@" "$JOB" >>"$OUT" <<'PY'
import json, sys
j, mode, w, h, ranks, threads, rep, decomp, comm, job = sys.argv[1:11]
d = json.load(open(j))
print(f'{mode},{w},{h},{d["steps"]},{ranks},{threads},{rep},'
      f'{d["kernel_seconds"]:.9f},{d["total_seconds"]:.9f},'
      f'{d["checksum_fnv1a64"]},{job},{decomp},{comm}')
PY
}

reference() {  # grid steps -> serial checksum
    local g=$1
    capture "$TMP/serial.json" "$SERIAL" --rows "$g" --cols "$g" \
        --steps "$STEPS" --seed "$SEED" --density-ppm "$DENSITY"
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum_fnv1a64"])' \
        "$TMP/serial.json"
}

for G in $GRIDS; do
    ref=$(reference "$G")
    # Report the device so the CSV can be traced to hardware.
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 >"$TMP/gpu.txt"
        echo "device: $(cat "$TMP/gpu.txt")" >&2
    fi
    for NP in $RANKS; do
        for rep in $(seq 1 "$REPEAT"); do
            capture "$TMP/gpu.json" env OMP_TARGET_OFFLOAD=MANDATORY \
                "$MPIRUN" -np "$NP" "${MPIRUN_OPTS[@]}" "$GPU_BIN" \
                --rows "$G" --cols "$G" --steps "$STEPS" \
                --seed "$SEED" --density-ppm "$DENSITY" --comm "$COMM"
            got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum_fnv1a64"])' \
                "$TMP/gpu.json")
            if [ "$got" != "$ref" ]; then
                echo "FAIL ${G}^2 x$NP checksum $got != $ref" >&2
                rm -f "$TMP/gpu.json"
                continue
            fi
            emit "$TMP/gpu.json" gpu "$G" "$G" "$NP" 1 "$rep" 1d "$COMM"
        done
    done
    if [ "$CPU_THREADS" -gt 0 ]; then
        for rep in $(seq 1 "$REPEAT"); do
            capture "$TMP/cpu.json" env OMP_NUM_THREADS="$CPU_THREADS" \
                "$MPIRUN" -np 1 "${MPIRUN_OPTS[@]}" "$HYBRID_BIN" \
                --rows "$G" --cols "$G" --steps "$STEPS" \
                --seed "$SEED" --density-ppm "$DENSITY" --comm "$COMM"
            emit "$TMP/cpu.json" hybrid-cpu "$G" "$G" 1 "$CPU_THREADS" "$rep" 1d "$COMM"
        done
    fi
done

echo "wrote $OUT ($(($(wc -l <"$OUT") - 1)) runs)"
