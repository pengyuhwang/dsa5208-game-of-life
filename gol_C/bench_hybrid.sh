#!/bin/bash
# Hybrid MPI+OpenMP at a fixed core budget (E4, E5). Run from a PBS job after
# job_setup.sh, on whole nodes (2 packages x 36 cores). Needs BIN, LOGS and:
#   NODES  nodes in the job          PAIRS  "ranksxthreads ...", ranks x threads = NODES x 64
# Optional: GRID (16384) STEPS (200) REPEAT (3) DECOMP (2d) OUT.
#
# Placement: 32 cores of each package, 64 per node, so a rank with up to 32
# threads never spans two packages (with 64 of 72 cores, PBS split them 36/28 and
# OpenMPI refused 4x16: job 1529183). Threads per rank must divide 32; the one
# exception is 1x64 on one node, where the rank is left unbound and OpenMP pins
# its threads to the same 32+32 cores.
#
# 1. Check: every pair on a 512^2 grid must match the serial checksum and report
#    the requested thread count; the bindings are saved per pair. Any failure
#    stops the job before the expensive part. This also serves as the warm-up.
# 2. Serial baseline on the full grid, 3 times.
# 3. Every pair, 3 times; repeats are the outer loop so drift hits all pairs alike.
set -euo pipefail
GRID=${GRID:-16384}
STEPS=${STEPS:-200}
REPEAT=${REPEAT:-3}
DECOMP=${DECOMP:-2d}
OUT=${OUT:?set OUT}
JOBID=${PBS_JOBID:-local}
SEED=(--seed 20261026 --density-ppm 500000)

HOSTS=($(sort -u "$PBS_NODEFILE"))
[ "${#HOSTS[@]}" -eq "$NODES" ] || { echo "FATAL: expected $NODES nodes, got ${#HOSTS[@]}"; exit 1; }
for pair in $PAIRS; do
    np=${pair%x*} t=${pair#*x}
    [ $((np * t)) -eq $((NODES * 64)) ] ||
        { echo "FATAL: $pair does not use $NODES nodes x 64 cores"; exit 1; }
    if [ "$t" -le 32 ]; then
        [ $((32 % t)) -eq 0 ] || { echo "FATAL: $t threads do not divide 32"; exit 1; }
    else
        [ "$np" -eq 1 ] && [ "$NODES" -eq 1 ] || { echo "FATAL: $pair spans packages"; exit 1; }
    fi
done

# Allowed cores per package on this node, and the first 32 of each as OpenMP
# places for the 1x64 case. One CPU per physical core if hyperthreads exist.
topology() {
    [ -n "${FAKE_TOPOLOGY:-}" ] && { printf '%b\n' "$FAKE_TOPOLOGY"; return; }
    python3 - <<'PY'
import os
pk = {}
for c in sorted(os.sched_getaffinity(0)):
    t = f'/sys/devices/system/cpu/cpu{c}/topology/'
    sib = open(t + 'thread_siblings_list').read().strip().replace('-', ',').split(',')
    if c != min(int(x) for x in sib):
        continue
    pk.setdefault(int(open(t + 'physical_package_id').read()), []).append(c)
print(' '.join(f'package{p}={len(v)}' for p, v in sorted(pk.items())))
print(','.join('{%d}' % c for p in sorted(pk) for c in pk[p][:32]))
PY
}
TOPO=$(topology)
echo "cores allowed on $(hostname): $(echo "$TOPO" | head -n 1)"
WIDE_PLACES=$(echo "$TOPO" | tail -n 1)
python3 -c 'import sys; c = [int(x.split("=")[1]) for x in sys.argv[1].split()]; sys.exit(not (len(c) == 2 and min(c) >= 32))' \
    "$(echo "$TOPO" | head -n 1)" ||
    { echo "FATAL: need 2 packages with at least 32 free cores each; request whole nodes"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "mode,W,H,steps,ranks,threads,repeat,kernel_seconds,total_seconds,live_cells,checksum,job_id,decomp,comm" > "$OUT"

field() {  # json key
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

emit() {  # json mode ranks threads repeat decomp
    python3 - "$@" "$GRID" "$JOBID" >> "$OUT" <<'PY'
import json, sys
j, mode, ranks, threads, rep, dc, g, job = sys.argv[1:9]
d = json.load(open(j))
print(f'{mode},{g},{g},{d["steps"]},{ranks},{d["threads"]},{rep},'
      f'{d["kernel_seconds"]:.9f},{d["total_seconds"]:.9f},'
      f'{d["live_cells"]},{d["checksum_fnv1a64"]},{job},{dc},{"na" if dc == "na" else "blocking"}')
PY
}

serial() {  # json size steps
    timeout 900 "$BIN/serial" --rows "$2" --cols "$2" --steps "$3" "${SEED[@]}" \
        2>>"$LOGS/stderr.log" | tail -n 1 > "$1" ||
        { echo "FATAL: serial run failed"; tail -n 10 "$LOGS/stderr.log"; exit 1; }
}

hybrid() {  # json np threads size steps stderr-file [extra mpirun options]
    local json=$1 np=$2 t=$3 size=$4 steps=$5 err=$6
    shift 6
    # 32/t ranks per package of t cores each = 64 cores per node.
    local hf=$TMP/hosts_$np places=cores map
    : > "$hf"
    for h in "${HOSTS[@]}"; do echo "$h slots=$((np / NODES))" >> "$hf"; done
    if [ "$t" -le 32 ]; then
        map=(--map-by "ppr:$((32 / t)):package:PE=$t" --bind-to core)
    else
        map=(--bind-to none)
        places=$WIDE_PLACES
    fi
    OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=$places timeout 900 \
        mpirun -np "$np" --hostfile "$hf" "${map[@]}" \
        -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES "$@" \
        "$BIN/hybrid" --rows "$size" --cols "$size" --steps "$steps" "${SEED[@]}" \
        --decomp "$DECOMP" --comm blocking 2>>"$err" | tail -n 1 > "$json" ||
        { echo "FATAL: hybrid ${np}x$t failed"; tail -n 15 "$err"; exit 1; }
}

expect() {  # json checksum threads label
    local got th
    got=$(field "$1" checksum_fnv1a64)
    th=$(field "$1" threads)
    if [ "$got" != "$2" ] || [ "$th" != "$3" ]; then
        echo "FATAL: $4 gave checksum $got threads $th, want $2 threads $3"
        exit 1
    fi
}

echo "=== check on 512^2, 20 steps ($DECOMP)"
serial "$TMP/s.json" 512 20
ref=$(field "$TMP/s.json" checksum_fnv1a64)
for pair in $PAIRS; do
    np=${pair%x*} t=${pair#*x}
    bind=$LOGS/bind_${np}x$t.log
    : > "$bind"
    OMP_DISPLAY_AFFINITY=TRUE hybrid "$TMP/h.json" "$np" "$t" 512 20 "$bind" \
        --report-bindings -x OMP_DISPLAY_AFFINITY
    expect "$TMP/h.json" "$ref" "$t" "check ${np}x$t"
    echo "ok ${np}x$t, $(grep -ac 'bound' "$bind") binding lines, first: $(grep -a -m 1 'bound' "$bind" | cut -c1-150)"
done

echo "=== serial baseline ${GRID}^2, $STEPS steps"
for rep in $(seq 1 "$REPEAT"); do
    serial "$TMP/s.json" "$GRID" "$STEPS"
    emit "$TMP/s.json" serial 1 1 "$rep" na
    echo "serial rep $rep: $(field "$TMP/s.json" kernel_seconds) s"
done
ref=$(field "$TMP/s.json" checksum_fnv1a64)

echo "=== hybrid ${GRID}^2, $STEPS steps, $DECOMP blocking"
for rep in $(seq 1 "$REPEAT"); do
    for pair in $PAIRS; do
        np=${pair%x*} t=${pair#*x}
        hybrid "$TMP/h.json" "$np" "$t" "$GRID" "$STEPS" "$LOGS/stderr.log"
        expect "$TMP/h.json" "$ref" "$t" "${np}x$t rep $rep"
        emit "$TMP/h.json" hybrid "$np" "$t" "$rep" "$DECOMP"
        echo "${np}x$t rep $rep: $(field "$TMP/h.json" kernel_seconds) s"
    done
done
echo "wrote $OUT ($(($(wc -l < "$OUT") - 1)) rows)"
