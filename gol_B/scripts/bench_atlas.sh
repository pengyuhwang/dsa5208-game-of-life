#!/usr/bin/env bash
# Atlas experiment batch for the shared interface (P1 PBM + JSON stdout).
# Usage: GRIDS="1024 2048" RANKS="1 2 4 8" DECOMPS="1d 2d" \
#        COMMS="blocking nonblocking overlap rma" REPEAT=3 bash scripts/bench_atlas.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}

BUILD=${BUILD:-$ROOT/build}
SERIAL=${SERIAL:-$A_DIR/build/serial}
OUT=${OUT:-$ROOT/results/raw_results.csv}
REPEAT=${REPEAT:-3}
STEPS=${STEPS:-500}
SEED=${SEED:-20261026}   # same as A's config/experiment_plan.csv
DENSITY=${DENSITY:-500000}
GRIDS=${GRIDS:-"1024 2048"}                   # square, or WxH for non-square
RANKS=${RANKS:-"1 2 4 8"}
DECOMPS=${DECOMPS:-1d}                       # 1d 2d
COMMS=${COMMS:-blocking}                     # blocking nonblocking overlap rma
SKIP_SERIAL=${SKIP_SERIAL:-0}                # 1 = weak-scaling grids without serial baseline
HYBRID_PAIRS=${HYBRID_PAIRS-1x8 2x4 4x2}      # fixed 8-core budget; "" disables hybrid
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_OPTS=${MPIRUN_OPTS:---oversubscribe}
JOB=${PBS_JOBID:-local}
WARMUP=${WARMUP:-1}          # 1 = one unrecorded launch before any measurement
GUARD=${GUARD:-1}            # 1 = abort on a checksum disagreement

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"
# APPEND=1 keeps an existing CSV (weak scaling runs one grid level per call)
if [ "${APPEND:-0}" = 1 ] && [ -f "$OUT" ]; then :; else
    echo "mode,W,H,steps,ranks,threads,repeat,kernel_seconds,total_seconds,live_cells,checksum,job_id,decomp,comm" > "$OUT"
fi

[ -x "$SERIAL" ] || { echo "missing $SERIAL (run make -C $A_DIR build/serial)" >&2; exit 1; }
[ -x "$BUILD/mpi" ] || { echo "missing $BUILD/mpi (run make -B build/mpi)" >&2; exit 1; }

emit() {  # json mode W H ranks threads repeat decomp comm
    python3 - "$@" "$JOB" >> "$OUT" <<'PY'
import json, sys
j, mode, w, h, ranks, threads, rep, dc, cm, job = sys.argv[1:11]
d = json.load(open(j))
print(f'{mode},{w},{h},{d["steps"]},{ranks},{d["threads"]},{rep},'
      f'{d["kernel_seconds"]:.9f},{d["total_seconds"]:.9f},'
      f'{d["live_cells"]},{d["checksum_fnv1a64"]},{job},{dc},{cm}')
PY
}

# One checksum per grid: serial, every rank count, every decomposition and
# every backend must agree, otherwise the row is meaningless.
declare -A REF_CK
guard() {  # json grid label
    [ "$GUARD" = 1 ] || return 0
    local ck ref
    ck=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum_fnv1a64"])' "$1" 2>/dev/null || true)
    [ -n "$ck" ] || { echo "cannot read checksum from $1" >&2; exit 1; }
    ref=${REF_CK[$2]:-}
    if [ -z "$ref" ]; then REF_CK[$2]=$ck; return 0; fi
    if [ "$ck" != "$ref" ]; then
        echo "checksum mismatch at $3: got $ck, expected $ref" >&2
        exit 1
    fi
}

capture() {  # json_path command...
    local json=$1
    shift
    # timeout guards against launcher hangs (Vanda cpuset binding once
    # wedged mpirun); a healthy run is seconds, 900s only trips on bugs.
    if ! timeout "${RUN_TIMEOUT:-900}" "$@" 2>>"$TMP/stderr.log" \
            | tail -n 1 >"$json"; then
        echo "run failed: $*" >&2
        tail -n 10 "$TMP/stderr.log" >&2
        exit 1
    fi
}

# One unrecorded launch keeps the first measured run free of start-up cost
# (mpirun prologue, page cache, MPI library warm-up).
if [ "$WARMUP" = 1 ]; then
    capture "$TMP/warm.json" env OMP_NUM_THREADS=1 \
        "$MPIRUN" -np 2 $MPIRUN_OPTS "$BUILD/mpi" \
        --rows 64 --cols 64 --steps 20 --seed "$SEED" --density-ppm "$DENSITY"
    echo "warm-up: np=2 64x64 steps=20"
fi

for G in $GRIDS; do
    case $G in
        *x*) W=${G%x*} H=${G#*x} ;;
        *)   W=$G H=$G ;;
    esac
    GK="${W}x${H}"
    if [ "$SKIP_SERIAL" != 1 ]; then
        for rep in $(seq 1 "$REPEAT"); do
            capture "$TMP/serial.json" "$SERIAL" --rows "$H" --cols "$W" \
                --steps "$STEPS" --seed "$SEED" --density-ppm "$DENSITY"
            guard "$TMP/serial.json" "$GK" "serial $GK"
            emit "$TMP/serial.json" serial "$W" "$H" 1 1 "$rep" na na
        done
    fi

    for NP in $RANKS; do
        for D in $DECOMPS; do
            if [ "$D" = 1d ]; then
                [ "$NP" -le "$H" ] || continue
            else
                { [ "$NP" -le "$H" ] && [ "$NP" -le "$W" ]; } || continue
            fi
            for M in $COMMS; do
                for rep in $(seq 1 "$REPEAT"); do
                    capture "$TMP/mpi.json" env OMP_NUM_THREADS=1 \
                        "$MPIRUN" -np "$NP" $MPIRUN_OPTS "$BUILD/mpi" \
                        --rows "$H" --cols "$W" --steps "$STEPS" \
                        --seed "$SEED" --density-ppm "$DENSITY" \
                        --decomp "$D" --comm "$M"
                    guard "$TMP/mpi.json" "$GK" "mpi np=$NP $D/$M"
                    emit "$TMP/mpi.json" mpi "$W" "$H" "$NP" 1 "$rep" "$D" "$M"
                done
            done
        done
    done

    for pair in $HYBRID_PAIRS; do
        NP=${pair%x*}
        T=${pair#*x}
        [ "$NP" -le "$H" ] || continue   # hybrid runs 1d decomposition
        for rep in $(seq 1 "$REPEAT"); do
            capture "$TMP/hybrid.json" env OMP_NUM_THREADS="$T" \
                "$MPIRUN" -np "$NP" $MPIRUN_OPTS "$BUILD/hybrid" \
                --rows "$H" --cols "$W" --steps "$STEPS" \
                --seed "$SEED" --density-ppm "$DENSITY" \
                --decomp 1d --comm blocking
            guard "$TMP/hybrid.json" "$GK" "hybrid np=$NP x$T"
            emit "$TMP/hybrid.json" hybrid "$W" "$H" "$NP" "$T" "$rep" 1d blocking
        done
    done
done

echo "wrote $OUT ($(($(wc -l <"$OUT") - 1)) runs)"
