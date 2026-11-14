#!/usr/bin/env bash
# Correctness matrix for --decomp (1d/2d) x --comm
# (blocking/nonblocking/overlap/rma/persistent/sendrecv2) across rank counts.
# Ground truth: A's serial kernel. Every parallel run must reproduce the
# serial checksum and live-cell count exactly, echo the requested decomp/comm,
# and carry every interface 1.0 JSON field. A --input run per backend checks
# the Scatterv/Gatherv path with a non-uniform initial pattern.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}
BUILD=${BUILD:-$ROOT/build}
SERIAL=${SERIAL:-$A_DIR/build/serial}
MPI=${MPI:-$BUILD/mpi}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_OPTS=${MPIRUN_OPTS:---oversubscribe}
STEPS=${STEPS:-37}
SEED=${SEED:-20261026}
DENSITY=${DENSITY:-500000}
GRIDS=${GRIDS:-"64x64 97x53 5x13"}   # WxH; 97x53 exercises uneven blocks,
                                     # 5x13 forces single-column blocks at np=8
                                     # (2x4 process grid, col widths 2 1 1 1)
RANKS=${RANKS:-"1 2 3 4 8"}
DECOMPS=${DECOMPS:-"1d 2d"}
COMMS=${COMMS:-"blocking nonblocking overlap rma persistent sendrecv2"}

[ -x "$SERIAL" ] || { echo "missing $SERIAL (make -C $A_DIR build/serial)" >&2; exit 1; }
[ -x "$MPI" ] || { echo "missing $MPI (make -C $ROOT)" >&2; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0 fail=0

check() {  # label json want_checksum want_live want_decomp want_comm
    python3 - "$@" <<'PY'
import json, sys
label, path, cksum, live, want_d, want_c = sys.argv[1:7]
d = json.load(open(path))
required = ["interface_version", "variant", "rows", "cols", "steps", "init",
            "seed", "density_ppm", "ranks", "threads", "live_cells",
            "checksum_fnv1a64", "kernel_seconds", "total_seconds"]
missing = [k for k in required if k not in d]
if missing:
    print(f"FAIL {label}: missing JSON fields {missing}"); sys.exit(1)
if d["checksum_fnv1a64"] != cksum or str(d["live_cells"]) != live:
    print(f"FAIL {label}: checksum/live mismatch"); sys.exit(1)
if d["decomp"] != want_d or d["comm"] != want_c:
    print(f"FAIL {label}: decomp/comm echo mismatch"); sys.exit(1)
PY
}

for G in $GRIDS; do
    W=${G%x*} H=${G#*x}
    "$SERIAL" --rows "$H" --cols "$W" --steps "$STEPS" --seed "$SEED" \
        --density-ppm "$DENSITY" > "$TMP/ref.json"
    read -r CK LIVE < <(python3 -c \
        "import json,sys; d=json.load(open(sys.argv[1])); print(d['checksum_fnv1a64'], d['live_cells'])" \
        "$TMP/ref.json")

    for NP in $RANKS; do
        for D in $DECOMPS; do
            for M in $COMMS; do
                label="${W}x${H} np=$NP $D/$M"
                skip=0
                [ "$D" = 1d ] && [ "$NP" -gt "$H" ] && skip=1
                [ "$D" = 2d ] && { [ "$NP" -gt "$H" ] || [ "$NP" -gt "$W" ]; } && skip=1
                if [ "$skip" = 1 ]; then echo "skip $label"; continue; fi
                if "$MPIRUN" -np "$NP" $MPIRUN_OPTS "$MPI" \
                        --rows "$H" --cols "$W" --steps "$STEPS" \
                        --seed "$SEED" --density-ppm "$DENSITY" \
                        --decomp "$D" --comm "$M" \
                        > "$TMP/out.json" 2>"$TMP/err.log" \
                   && check "$label" "$TMP/out.json" "$CK" "$LIVE" "$D" "$M"; then
                    echo "pass $label"; pass=$((pass + 1))
                else
                    echo "fail $label"; sed -n '1,5p' "$TMP/err.log" >&2
                    fail=$((fail + 1))
                fi
            done
        done
    done
done

# --input path: same matrix over one non-uniform pattern at np=4.
python3 - "$TMP/init.pbm" <<'PY'
import sys
w, h = 23, 17
lines = ["P1", f"{w} {h}"]
for r in range(h):
    lines.append(" ".join("1" if (r * w + c) % 7 < 3 else "0" for c in range(w)))
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
PY
"$SERIAL" --input "$TMP/init.pbm" --steps "$STEPS" > "$TMP/ref.json"
read -r CK LIVE < <(python3 -c \
    "import json,sys; d=json.load(open(sys.argv[1])); print(d['checksum_fnv1a64'], d['live_cells'])" \
    "$TMP/ref.json")
for D in $DECOMPS; do
    for M in $COMMS; do
        label="input 23x17 np=4 $D/$M"
        if "$MPIRUN" -np 4 $MPIRUN_OPTS "$MPI" --input "$TMP/init.pbm" \
                --steps "$STEPS" --decomp "$D" --comm "$M" \
                > "$TMP/out.json" 2>"$TMP/err.log" \
           && check "$label" "$TMP/out.json" "$CK" "$LIVE" "$D" "$M"; then
            echo "pass $label"; pass=$((pass + 1))
        else
            echo "fail $label"; sed -n '1,5p' "$TMP/err.log" >&2
            fail=$((fail + 1))
        fi
    done
done

echo "PASS=$pass FAIL=$fail"
[ "$fail" = 0 ]
