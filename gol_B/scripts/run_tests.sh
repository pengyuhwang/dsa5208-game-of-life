#!/usr/bin/env bash
# Correctness and cross-version verification through member A's oracle harness.
# The two external binaries are installed into ../Game_of_Life_A/build so that
# tools/verify.py finds ./build/mpi and ./build/hybrid from its project root.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A_DIR=${A_DIR:-$ROOT/../Game_of_Life_A}

make -C "$ROOT" install-a
make -C "$A_DIR" build/serial
cd "$A_DIR"
exec python3 tools/verify.py --require-all "$@"
