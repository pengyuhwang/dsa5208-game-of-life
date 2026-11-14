"""Draw the two figures of the report that are not member C's originals.

fig_comm_methods.png   six halo-exchange methods at 16, 64 and 256 processes
fig_core_time.png      core time per cell update against the number of processes

Usage: python3 report/make_figures.py   (needs matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

import stats

OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # 1D, 2D: same colours as the other figures
INK, AXIS, GRID, BAND = "#52514e", "#8a8984", "#e6e5e1", "#eeeeeb"
STYLE = {"1d": dict(color=BLUE, marker="o"), "2d": dict(color=ORANGE, marker="D")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": "#0b0b0b",
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.9, "legend.frameon": False})


def plain(value, _pos=None):
    return "%g" % value


def log_axis(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(plain))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def legend_handles():
    return [Line2D([], [], lw=2, ms=6, mec="white", mew=1, label=name.upper(), **STYLE[name])
            for name in ("1d", "2d")]


def comm_methods():
    runs = [r for r in stats.all_runs() if r["experiment"] == "E3" and r["mode"] == "mpi"]
    runs += [r for r in stats.later_method_runs() if r["mode"] == "mpi"]
    groups = stats.grouped(runs, "ranks", "decomp", "comm")
    panels = [(16, "16 processes (1 node)", [8.2, 8.6, 9, 9.5], (8.15, 9.7)),
              (64, "64 processes (1 node)", [2.4, 2.6, 2.8, 3, 3.4], (2.33, 3.5)),
              (256, "256 processes (4 nodes)", [0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4], (0.6, 1.42))]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.85), sharey=True)
    for ax, (ranks, title, ticks, limits) in zip(axes, panels):
        for row, method in enumerate(stats.METHODS):
            for decomp, shift in (("1d", -0.17), ("2d", 0.17)):
                med, low, high = stats.summary(groups[(ranks, decomp, method)])
                ax.errorbar(med, row + shift, xerr=[[med - low], [high - med]], fmt="none",
                            ecolor=STYLE[decomp]["color"], elinewidth=1.3, capsize=2.5)
                ax.plot(med, row + shift, ls="none", ms=6, mec="white", mew=1, **STYLE[decomp])
        ax.set_xscale("log")
        log_axis(ax.xaxis, ticks)
        ax.set_xlim(*limits)
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    labels = [m + (" †" if m in stats.LATER_METHODS else "") for m in stats.METHODS]
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels) - 0.45, -0.55)
    axes[1].set_xlabel("kernel time (s), 16384² × 200 steps, log scale")
    fig.legend(handles=legend_handles(), loc="lower right", ncol=2, handlelength=1.6,
               bbox_to_anchor=(0.995, -0.005), columnspacing=1.2)
    fig.text(0.008, 0.012, "† measured in a later job (1535552 / 1535553); the other four in job 1527956",
             fontsize=7.2, color=INK)
    fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=1.2)
    fig.savefig(OUT / "fig_comm_methods.png", dpi=220)
    plt.close(fig)


def core_time():
    runs = stats.all_runs()
    strong = [r for r in runs if r["experiment"] == "E1"]
    weak = [r for r in runs if r["experiment"] == "E2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    panels = [(axes[0], strong, "(a) Strong scaling, 16384² × 200 steps"),
              (axes[1], weak, "(b) Weak scaling, 1024² cells per process")]
    for ax, rows, title in panels:
        ax.axvspan(90, 330, color=BAND, lw=0)
        ax.text(96, 6.35, "2–4 nodes", ha="left", va="top", fontsize=7.5, color=INK)
        single = [r for r in rows if r["ranks"] == 1]
        base = stats.core_ns_per_cell(stats.summary(single)[0], single[0])
        ax.axhline(base, color=AXIS, lw=1.2)
        ax.text(1.05, base * 0.985, "one process: %.2f ns" % base, va="top", fontsize=7.5, color=INK)
        mpi = stats.grouped([r for r in rows if r["mode"] == "mpi"], "decomp", "ranks")
        for decomp in ("1d", "2d"):
            xs = sorted(p for d, p in mpi if d == decomp and (p > 1 or rows is weak))
            points = []
            for p in xs:
                sample = mpi[(decomp, p)]
                points.append([stats.core_ns_per_cell(v, sample[0]) for v in stats.summary(sample)])
            med = [v[0] for v in points]
            ax.errorbar(xs, med, yerr=[[m - v[1] for m, v in zip(med, points)],
                                       [v[2] - m for m, v in zip(med, points)]],
                        lw=2, ms=6, mec="white", mew=1, elinewidth=1.2, capsize=2.5,
                        label=decomp.upper(), **STYLE[decomp])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        log_axis(ax.xaxis, [1, 2, 4, 8, 16, 32, 64, 128, 256])
        log_axis(ax.yaxis, [2, 2.5, 3, 4, 5, 6])
        ax.set_xlim(0.85, 330)
        ax.set_ylim(1.95, 6.6)
        ax.set_title(title)
        ax.set_xlabel("MPI processes (cores)")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("core time per cell update (ns)")
    axes[0].legend(loc="upper left", handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_core_time.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    comm_methods()
    core_time()
    print("wrote fig_comm_methods.png and fig_core_time.png in", OUT)
