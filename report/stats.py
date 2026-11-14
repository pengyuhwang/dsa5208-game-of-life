"""Load the recorded runs and compute the statistics used in the report.

Everything is read from the CSV files committed with the code, so the tables and
figures can be regenerated without access to the cluster.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["blocking", "nonblocking", "overlap", "rma", "persistent", "sendrecv2"]
LATER_METHODS = ("persistent", "sendrecv2")   # measured in a later job


def _rows(path):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _typed(row, experiment=None):
    out = dict(row)
    if experiment:
        out["experiment"] = experiment
    for key in ("W", "H", "steps", "ranks", "threads"):
        out[key] = int(row[key])
    for key in ("kernel_seconds", "total_seconds"):
        out[key] = float(row[key])
    if row.get("halo_seconds"):
        out["halo_seconds"] = float(row["halo_seconds"])
    out["job"] = row["job_id"].split(".")[0]
    return out


def all_runs():
    """Every formal run merged by gol_C/merge_results.py."""
    return [_typed(r) for r in _rows("gol_C/results/all_runs.csv")]


def later_method_runs():
    """persistent and sendrecv2: 16 and 64 ranks (job 1535552), 256 ranks (1535553)."""
    rows = _rows("gol_B/results/e4_newcomm_p1.csv") + _rows("gol_B/results/e4_newcomm_p2.csv")
    return [_typed(r, "later_methods") for r in rows]


def halo_runs():
    """Atlas runs with the halo timer on: one run per configuration."""
    return [_typed(r, "halo_atlas") for r in _rows("gol_B/results/halo_atlas.csv")]


def select(rows, **where):
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def grouped(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def summary(rows, field="kernel_seconds"):
    """Median, minimum and maximum of one field over repeated runs."""
    values = [r[field] for r in rows]
    return statistics.median(values), min(values), max(values)


def core_ns_per_cell(seconds, row):
    """Core time spent per cell update, in nanoseconds."""
    cores = row["ranks"] * row["threads"]
    return 1e9 * seconds * cores / (row["W"] * row["H"] * row["steps"])


def serial_median(rows, experiment, size=16384, steps=200):
    base = [r for r in rows if r["experiment"] == experiment and r["mode"] == "serial"
            and r["W"] == size and r["steps"] == steps]
    return statistics.median(r["kernel_seconds"] for r in base)


def checksums_consistent(rows):
    """True when every run of the same grid and step count has one checksum."""
    seen = defaultdict(set)
    for r in rows:
        seen[(r["W"], r["H"], r["steps"])].add(r["checksum"])
    return all(len(v) == 1 for v in seen.values()), len(seen)
