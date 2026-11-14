#!/usr/bin/env bash
# Halo-time accounting batch (GOL_HALO_STATS=1): one measured halo_seconds per
# backend/decomp/rank-count at production scale. Runs the full backend matrix
# so the report can attribute communication cost per transport method.
# Output: results/halo_atlas.csv (interface schema + halo_seconds column).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}
BUILD=${BUILD:-$ROOT/build}
OUT=${OUT:-$ROOT/results/halo_atlas.csv}
STEPS=${STEPS:-200}
SEED=${SEED:-20261026}
DENSITY=${DENSITY:-500000}
GRIDS=${GRIDS:-16384}                          # square, or WxH
RANKS=${RANKS:-"16 64 256"}
DECOMPS=${DECOMPS:-"1d 2d"}
COMMS=${COMMS:-"blocking nonblocking overlap rma persistent sendrecv2"}
REPEAT=${REPEAT:-1}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_OPTS=${MPIRUN_OPTS:-}
JOB=${PBS_JOBID:-local}

export GOL_HALO_STATS=1
mkdir -p "$(dirname "$OUT")"
echo "mode,W,H,steps,ranks,threads,repeat,kernel_seconds,total_seconds,live_cells,checksum,job_id,decomp,comm,halo_seconds" > "$OUT"

[ -x "$BUILD/mpi" ] || { echo "missing $BUILD/mpi (run make -B)" >&2; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

emit() {  # json W H ranks repeat decomp comm
    python3 - "$@" "$JOB" >> "$OUT" <<'PY'
import json, sys
j, w, h, ranks, rep, dc, cm, job = sys.argv[1:9]
d = json.load(open(j))
print(f'mpi,{w},{h},{d["steps"]},{ranks},{d["threads"]},{rep},'
      f'{d["kernel_seconds"]:.9f},{d["total_seconds"]:.9f},'
      f'{d["live_cells"]},{d["checksum_fnv1a64"]},{job},{dc},{cm},'
      f'{d.get("halo_seconds", 0):.9f}')
PY
}

# Every row of one grid must reproduce the same checksum, otherwise the
# halo_seconds column sits next to a wrong result.
declare -A REF_CK
guard() {  # json grid-key label
    local ck ref
    ck=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum_fnv1a64"])' "$1") || {
        echo "cannot read checksum from $1 (python rc=$?)" >&2
        echo "--- file start ---" >&2; head -c 400 "$1" >&2; echo >&2
        echo "--- err.log ---" >&2; head -c 400 "${1%out.json}err.log" >&2
        exit 1
    }
    ref=${REF_CK[$2]:-}
    if [ -z "$ref" ]; then REF_CK[$2]=$ck; return 0; fi
    [ "$ck" = "$ref" ] || { echo "checksum mismatch at $3: got $ck, expected $ref" >&2; exit 1; }
}

for G in $GRIDS; do
    case $G in
        *x*) W=${G%x*} H=${G#*x} ;;
        *)   W=$G H=$G ;;
    esac
    for NP in $RANKS; do
        for D in $DECOMPS; do
            if [ "$D" = 1d ]; then [ "$NP" -le "$H" ] || continue
            else { [ "$NP" -le "$H" ] && [ "$NP" -le "$W" ]; } || continue; fi
            for M in $COMMS; do
                for rep in $(seq 1 "$REPEAT"); do
                    label="${W}x${H} np=$NP $D/$M"
                    # FAB (set by the PBS wrapper) pins the easybuild
                    # libfabric onto mpirun only: libmpi's DT_RPATH hits the
                    # stale /opt/amazon/efa copy, and a global LD_PRELOAD
                    # would also pollute the python guards.
                    if ! timeout "${RUN_TIMEOUT:-900}" \
                            env OMP_NUM_THREADS=1 ${FAB:+LD_PRELOAD=$FAB} \
                            "$MPIRUN" -np "$NP" $MPIRUN_OPTS "$BUILD/mpi" \
                            --rows "$H" --cols "$W" --steps "$STEPS" \
                            --seed "$SEED" --density-ppm "$DENSITY" \
                            --decomp "$D" --comm "$M" \
                            > "$TMP/out.json" 2>"$TMP/err.log"; then
                        echo "run failed: $label" >&2
                        tail -n 10 "$TMP/err.log" >&2
                        exit 1
                    fi
                    guard "$TMP/out.json" "${W}x${H}" "$label"
                    emit "$TMP/out.json" "$W" "$H" "$NP" "$rep" "$D" "$M"
                    echo "done $label"
                done
            done
        done
    done
done

echo "wrote $OUT ($(($(wc -l <"$OUT") - 1)) runs)"
