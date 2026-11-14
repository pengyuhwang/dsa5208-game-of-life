#!/usr/bin/env python3
"""Summarise raw_results.csv: mean kernel time, speedup and efficiency.

Speedup uses kernel_seconds only, the scope defined by interface 1.0: all
iterations including halo exchange, excluding input, gathering and file I/O.
Efficiency divides speedup by ranks*threads.
"""
import collections
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "raw_results.csv"
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit(f"{path}: no data")

serial = collections.defaultdict(list)
groups = collections.defaultdict(list)
for r in rows:
    key = (int(r["W"]), int(r["H"]), int(r["steps"]))
    if r["mode"] == "serial":
        serial[key].append(float(r["kernel_seconds"]))
    else:
        # decomp/comm columns are new; older CSVs fall back to the defaults
        groups[(r["mode"],) + key + (int(r["ranks"]), int(r["threads"]),
                r.get("decomp", "1d"), r.get("comm", "blocking"))].append(
            float(r["kernel_seconds"]))

print("| mode | grid | steps | ranks | threads | decomp | comm | runs | median s | mean s | min s | max s | speedup | efficiency |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(groups):
    mode, w, h, steps, ranks, threads, dc, cm = key
    times = groups[key]
    base = serial.get((w, h, steps))
    if not base:
        continue
    speedup = statistics.median(base) / statistics.median(times)
    print(f"| {mode} | {w}x{h} | {steps} | {ranks} | {threads} | {dc} | {cm} | "
          f"{len(times)} | {statistics.median(times):.3f} | {statistics.mean(times):.3f} | "
          f"{min(times):.3f} | {max(times):.3f} | {speedup:.2f} | "
          f"{speedup / (ranks * threads):.2f} |")

print()
for grid, times in sorted(serial.items()):
    print(f"serial {grid[0]}x{grid[1]} steps={grid[2]}: "
          f"median {statistics.median(times):.3f} s, mean {statistics.mean(times):.3f} s "
          f"over {len(times)} runs")
