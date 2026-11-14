#!/usr/bin/env bash
# Serial runtime vs problem size (lecture requirement: 4-5 points, log axes).
# Free to run on Atlas; the numbers feed the O(N^2) scaling argument in the
# report and reuse the same kernel_seconds field as the parallel rows.
#
#   GRIDS="512 1024 2048 4096 8192" STEPS=100 bash scripts/serial_scaling.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}
SERIAL=${SERIAL:-$A_DIR/build/serial}
OUT=${OUT:-$ROOT/results/serial_scaling.csv}
GRIDS=${GRIDS:-"512 1024 2048 4096 8192"}
STEPS=${STEPS:-100}
REPEAT=${REPEAT:-3}
SEED=${SEED:-20261026}
DENSITY=${DENSITY:-500000}
JOB=${PBS_JOBID:-local}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"
echo "mode,W,H,steps,ranks,threads,repeat,kernel_seconds,total_seconds,checksum,job_id" > "$OUT"

[ -x "$SERIAL" ] || { echo "missing $SERIAL" >&2; exit 1; }

for G in $GRIDS; do
    for rep in $(seq 1 "$REPEAT"); do
        "$SERIAL" --rows "$G" --cols "$G" --steps "$STEPS" \
            --seed "$SEED" --density-ppm "$DENSITY" 2>>"$TMP/stderr.log" |
            tail -n 1 >"$TMP/serial.json"
        python3 - "$TMP/serial.json" "$G" "$rep" "$JOB" >>"$OUT" <<'PY'
import json, sys
j, g, rep, job = sys.argv[1:5]
d = json.load(open(j))
print(f'serial,{g},{g},{d["steps"]},1,1,{rep},'
      f'{d["kernel_seconds"]:.9f},{d["total_seconds"]:.9f},'
      f'{d["checksum_fnv1a64"]},{job}')
PY
    done
done

echo "wrote $OUT ($(($(wc -l <"$OUT") - 1)) runs)"
