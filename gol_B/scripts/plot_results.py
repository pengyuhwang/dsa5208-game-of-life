#!/usr/bin/env python3
"""Render the final Vanda figures from the 256-rank experiment CSVs.

Inputs (results/): vanda_e2_weak_256.csv (weak, 1024^2 cells/rank),
vanda_e3_backends_256.csv (strong 16384^2 + backend matrix),
serial_scaling.csv (single-core scaling).
Outputs (report/figures/): strong_scaling.png, weak_scaling.png,
backends.png, serial_scaling.png.

Lecture 12 requires log axes for time / problem size / core count. Reported
value is the per-group median; the min-max span is drawn as a band/whisker.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "report", "figures")

DECOMPS = ("1d", "2d")
COLORS = {"1d": "tab:blue", "2d": "tab:red"}
MARKS = {"1d": "o", "2d": "s"}
COMMS = ("blocking", "nonblocking", "overlap", "rma")


def load(name):
    rows = []
    with open(os.path.join(RES, name)) as f:
        for r in csv.DictReader(f):
            r["kernel"] = float(r["kernel_seconds"])
            r["ranks"] = int(r["ranks"])
            rows.append(r)
    return rows


def agg(rows, key):
    """key -> (min, median, max, mean)."""
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r["kernel"])
    out = {}
    for k, v in g.items():
        v.sort()
        out[k] = (v[0], v[len(v) // 2], v[-1], sum(v) / len(v))
    return out


def plot_strong(rows):
    serial = agg((r for r in rows if r["mode"] == "serial"), lambda r: 0)[0][1]
    print(f"strong: serial median {serial:.3f} s", file=sys.stderr)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ticks = []
    for d in DECOMPS:
        a = agg((r for r in rows
                 if r["decomp"] == d and r["comm"] == "blocking"), lambda r: r["ranks"])
        ranks = sorted(a)
        sp = [serial / a[p][1] for p in ranks]
        lo = [serial / a[p][2] for p in ranks]
        hi = [serial / a[p][0] for p in ranks]
        ax.fill_between(ranks, lo, hi, color=COLORS[d], alpha=0.15)
        ax.plot(ranks, sp, marker=MARKS[d], color=COLORS[d], label=f"{d} (median)")
        ticks += ranks
        print(f"  {d}: " + ", ".join(
            f"{p}->{serial / a[p][1]:.1f}x" for p in ranks), file=sys.stderr)
    ticks = sorted(set(ticks))
    ax.plot(ticks, ticks, "k--", lw=1, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("MPI ranks (log)")
    ax.set_ylabel("speedup over serial (log)")
    ax.set_title("Strong scaling, 16384$^2$ x 200 steps, blocking")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "strong_scaling.png"), dpi=150)
    plt.close(fig)


def plot_weak(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    base = {}
    for d in DECOMPS:
        a = agg((r for r in rows if r["decomp"] == d), lambda r: r["ranks"])
        ranks = sorted(a)
        base[d] = a[ranks[0]][1]
        med = [a[p][1] for p in ranks]
        ax1.fill_between(ranks, [a[p][0] for p in ranks],
                         [a[p][2] for p in ranks], color=COLORS[d], alpha=0.15)
        ax1.plot(ranks, med, marker=MARKS[d], color=COLORS[d],
                 label=f"{d} (median)")
        ax2.plot(ranks, [base[d] / a[p][1] for p in ranks],
                 marker=MARKS[d], color=COLORS[d], label=f"{d}")
        print(f"weak {d}: " + ", ".join(f"{p}:{a[p][1]:.2f}s" for p in ranks),
              file=sys.stderr)
        print(f"weak {d} efficiency at {ranks[-1]}: "
              f"{base[d] / a[ranks[-1]][1]:.1%}", file=sys.stderr)
    tick = sorted({p for d in DECOMPS
                   for p in agg((r for r in rows if r["decomp"] == d),
                                lambda r: r["ranks"])})
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(tick)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("MPI ranks (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    ax1.set_yscale("log")
    ax1.set_ylabel("kernel seconds (log)")
    ax1.set_title("Weak scaling, 1024$^2$ cells/rank x 500 steps")
    ax2.set_ylim(0.4, 1.05)
    ax2.set_ylabel("weak efficiency (T1 / Tp)")
    ax2.set_title("Weak scaling efficiency")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "weak_scaling.png"), dpi=150)
    plt.close(fig)


def plot_backends(rows):
    combos = [(d, c) for d in DECOMPS for c in COMMS]
    levels = sorted({r["ranks"] for r in rows if r["mode"] != "serial"})
    fig, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 4.2),
                             sharey=False)
    for ax, np_ in zip(axes, levels):
        a = agg((r for r in rows if r["ranks"] == np_),
                lambda r: (r["decomp"], r["comm"]))
        med = [a[k][1] for k in combos]
        err = [[a[k][1] - a[k][0] for k in combos],
               [a[k][2] - a[k][1] for k in combos]]
        ax.bar(range(len(combos)), med, yerr=err, capsize=3,
               color=[COLORS[d] for d, _ in combos])
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"{d}\n{c[:4]}" for d, c in combos], fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("kernel seconds (log, median)")
        ax.set_title(f"{np_} ranks, 16384$^2$ x 200 steps")
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(med):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        print(f"backends {np_}: " + ", ".join(
            f"{d}/{c[:4]}={a[(d, c)][1]:.3f}" for d, c in combos),
            file=sys.stderr)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "backends.png"), dpi=150)
    plt.close(fig)


def plot_serial():
    per = defaultdict(list)
    with open(os.path.join(RES, "serial_scaling.csv")) as f:
        for r in csv.DictReader(f):
            per[int(r["W"]) * int(r["H"])].append(float(r["kernel_seconds"]))
    sc = {c: (min(v), sorted(v)[len(v) // 2], max(v)) for c, v in per.items()}
    cells = sorted(sc)
    t = [sc[c][1] for c in cells]
    lx = [math.log10(c) for c in cells]
    ly = [math.log10(v) for v in t]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = (sum((x - mx) * (y - my) for x, y in zip(lx, ly)) /
             sum((x - mx) ** 2 for x in lx))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.fill_between(cells, [sc[c][0] for c in cells],
                    [sc[c][2] for c in cells], color="tab:blue", alpha=0.15)
    ax.plot(cells, t, "o-", color="tab:blue", label="measured (median of 3)")
    ax.plot(cells, [t[0] * (c / cells[0]) for c in cells], "k--", lw=1,
            label="ideal O(cells)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("grid cells (log)")
    ax.set_ylabel("serial kernel seconds (log)")
    ax.set_title("Serial scaling, 100 steps, single core")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    for c in cells:
        print(f"serial {c} cells: median {sc[c][1]:.3f} s "
              f"({c / sc[c][1] / 1e6:.1f} M cells/s)", file=sys.stderr)
    print(f"log-log slope = {slope:.3f} (1.0 = linear in cells)", file=sys.stderr)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "serial_scaling.png"), dpi=150)
    plt.close(fig)


def plot_gpu():
    """GPU (OpenMP target, 1 rank) vs CPU reference (1 rank x 36 threads)."""
    rows = load("vanda_gpu_paid.csv")
    a = agg(rows, lambda r: (r["mode"], r["W"]))
    modes = [m for m in ("gpu", "hybrid-cpu") if (m, rows[0]["W"]) in a]
    grids = sorted({int(r["W"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4.2))
    width = 0.35
    for i, m in enumerate(modes):
        med = [a[(m, str(g))][1] for g in grids]
        err = [[a[(m, str(g))][1] - a[(m, str(g))][0] for g in grids],
               [a[(m, str(g))][2] - a[(m, str(g))][1] for g in grids]]
        xs = [j + (i - 0.5) * width for j in range(len(grids))]
        ax.bar(xs, med, width, yerr=err, capsize=3,
               label=("1 GPU (OpenMP target)" if m == "gpu"
                      else "1 rank x 36 CPU threads"))
        for x, v in zip(xs, med):
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        print(f"{m}: " + ", ".join(f"{g}^2={a[(m, str(g))][1]:.4f}s"
                                   for g in grids), file=sys.stderr)
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([f"{g}$^2$" for g in grids])
    ax.set_yscale("log")
    ax.set_ylabel("kernel seconds (log, median of 3)")
    ax.set_title("GPU vs CPU, 300 steps, 1D blocking (Vanda A40)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpu_cpu.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    plot_strong(load("vanda_e3_backends_256.csv"))
    plot_weak(load("vanda_e2_weak_256.csv"))
    plot_backends(load("vanda_e3_backends_256.csv"))
    plot_serial()
    plot_gpu()
    print("figures written to", FIG, file=sys.stderr)


if __name__ == "__main__":
    main()
