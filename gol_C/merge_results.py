#!/usr/bin/env python3
"""Merge every formal result CSV of the project into one file, check it, and
write the summary tables. Standard library only. Run from anywhere:

    python3 gol_C/merge_results.py

Reads the raw CSVs of A, B and C (paths below, relative to the project folder)
and writes next to C's raw data in gol_C/results/:
    all_runs.csv     every formal run, one row each, with experiment and job
    tables.md        checks and summary tables (median, min-max, speedup, efficiency)
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "gol_C" / "results"

# experiment, file, expected rows, repeats per group, nodes, what it is
SOURCES = [
    ("serial_A", "Game_of_Life_A/results/serial_scaling/raw_results.csv", 21, 3,
     "IN-002 (Xeon Gold 6438N)", "A: serial, 256^2-16384^2, 500 steps"),
    ("serial_B", "gol_B/results/serial_scaling.csv", 15, 3,
     "not recorded", "B: serial, 512^2-8192^2, 100 steps"),
    ("E1", "gol_C/results/e1_strong.csv", 51, 3,
     "4 x gn-a40 (046/054/057/067)", "strong scaling, 16384^2 x 200, 2-256 ranks"),
    ("E1_1node", "gol_C/results/e1_1node.csv", 39, 3,
     "1 x gn-a40-057", "E1 backup, 2-64 ranks"),
    ("E2", "gol_B/results/vanda_e2_weak_256.csv", 54, 3,
     "4 nodes, names not recorded here", "weak scaling, 1024^2 per rank x 500"),
    ("E3", "gol_B/results/vanda_e3_backends_256.csv", 75, 3,
     "4 x cn (087/093/105/107)", "4 backends, 16384^2 x 200, 16/64/256 ranks"),
    ("E4", "gol_C/results/e4_hybrid.csv", 21, 3,
     "4 x cn (004/120/132/134)", "hybrid, 256 cores, 16384^2 x 200, 2d"),
    ("E5", "gol_C/results/e5_hybrid.csv", 15, 3,
     "1 x cn-144", "hybrid, 64 cores, 16384^2 x 200, 2d"),
    ("E6", "gol_C/results/e6_gpu_1530468.csv", 40, 5,
     "1 x gn-a40-020 (A40 + 36 cores)", "GPU vs 36-thread CPU, 2048^2-16384^2 x 300"),
    ("recheck", "gol_C/results/e2d_recheck.csv", 15, 3,
     "4 x cn (083/090/134/138), 3 switches", "2D recheck, 64/256 ranks"),
]
JOBS = {
    "serial_A": "1513317", "serial_B": "1524532", "E1": "1529185", "E1_1node": "1530869",
    "E2": "1527955", "E3": "1527956", "E4": "1530362", "E5": "1530361",
    "E6": "1530468", "recheck": "1532914",
}
FIELDS = ["experiment", "mode", "W", "H", "steps", "ranks", "threads", "decomp", "comm",
          "repeat", "kernel_seconds", "total_seconds", "checksum", "job_id", "nodes", "source"]


def load(exp, rel, nodes):
    rows = []
    with open(ROOT / rel, newline="") as f:
        for r in csv.DictReader(f):
            if "environment" in r:  # A's serial format
                r = {"mode": "serial", "W": r["cols"], "H": r["rows"], "steps": r["steps"],
                     "ranks": "1", "threads": "1", "repeat": r["repeat"],
                     "kernel_seconds": r["kernel_seconds"], "total_seconds": r["total_seconds"],
                     "checksum": r["checksum_fnv1a64"], "job_id": r["job_id"]}
            rows.append({
                "experiment": exp, "mode": r["mode"], "W": int(r["W"]), "H": int(r["H"]),
                "steps": int(r["steps"]), "ranks": int(r["ranks"]), "threads": int(r["threads"]),
                "decomp": r.get("decomp", "na"), "comm": r.get("comm", "na"),
                "repeat": int(r["repeat"]), "kernel_seconds": float(r["kernel_seconds"]),
                "total_seconds": float(r["total_seconds"]), "checksum": r["checksum"],
                "job_id": r["job_id"].split(".")[0], "nodes": nodes, "source": rel})
    return rows


def stats(v):
    return st.median(v), min(v), max(v)


def table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out) + "\n"


def f3(x):
    return f"{x:.3f}"


def rng(lo, hi):
    return f"{lo:.3f}–{hi:.3f}"


def main():
    data, missing = {}, []
    for exp, rel, nrows, nrep, nodes, desc in SOURCES:
        if (ROOT / rel).exists():
            data[exp] = load(exp, rel, nodes)
        else:
            missing.append((exp, rel))

    md = ["# 实验数据汇总与检查\n",
          "由 `gol_C/merge_results.py` 生成。数值是 `kernel_seconds`（只计迭代，不含初始化、收集和输出）。"
          "每组取中位数，括号或“范围”列是最小–最大。speedup = 同一作业内单核中位数 / 并行中位数；"
          "efficiency = speedup / 总核数（Hybrid 为进程数 × 线程数）。\n"]

    # ---- checks
    md.append("## 1. 检查\n")
    body, problems = [], []
    for exp, rel, nrows, nrep, nodes, desc in SOURCES:
        if exp not in data:
            body.append([exp, JOBS[exp], "**缺文件**", "", "", "", desc])
            problems.append(f"{exp}: 缺 {rel}")
            continue
        rows = data[exp]
        jobs = sorted({r["job_id"] for r in rows})
        groups = defaultdict(list)
        for r in rows:
            groups[(r["mode"], r["W"], r["H"], r["steps"], r["ranks"], r["threads"],
                    r["decomp"], r["comm"])].append(r)
        reps_ok = all(len(g) == nrep for g in groups.values())
        ck = defaultdict(set)
        for r in rows:
            ck[(r["W"], r["H"], r["steps"])].add(r["checksum"])
        ck_ok = all(len(s) == 1 for s in ck.values())
        if len(rows) != nrows:
            problems.append(f"{exp}: {len(rows)} 行，应为 {nrows}")
        if not reps_ok:
            problems.append(f"{exp}: 有的配置重复次数不是 {nrep}")
        if not ck_ok:
            problems.append(f"{exp}: 同一网格和步数下 checksum 不一致")
        if jobs != [JOBS[exp]]:
            problems.append(f"{exp}: 作业号 {jobs}，应为 {JOBS[exp]}")
        body.append([exp, ", ".join(jobs), f"{len(rows)} / {nrows}",
                     "OK" if reps_ok else "**不对**", "OK" if ck_ok else "**不一致**", nodes, desc])
    md.append(table(["实验", "作业", "行数 / 应有", f"每组重复次数", "checksum", "节点", "内容"], body))

    # same grid and steps must give the same checksum across all jobs
    cross = defaultdict(set)
    for exp, rows in data.items():
        for r in rows:
            cross[(r["W"], r["H"], r["steps"])].add((r["checksum"], exp))
    body = []
    for k in sorted(cross):
        sums = {c for c, _ in cross[k]}
        exps = sorted({e for _, e in cross[k]})
        if len(exps) > 1:
            ok = len(sums) == 1
            body.append([f"{k[0]}×{k[1]} × {k[2]} 步", ", ".join(exps), ", ".join(sorted(sums)),
                         "OK" if ok else "**不一致**"])
            if not ok:
                problems.append(f"{k}: 不同作业 checksum 不一致")
    md.append("\n跨作业：同一网格和步数在不同作业里的 checksum\n\n")
    md.append(table(["网格和步数", "出现在", "checksum", "结果"], body))
    md.append("\n" + ("**所有检查通过。**\n" if not problems else
                      "**发现问题：**\n\n" + "\n".join(f"- {p}" for p in problems) + "\n"))

    def serial_base(exp):
        v = [r["kernel_seconds"] for r in data[exp] if r["mode"] == "serial"]
        return st.median(v) if v else None

    def grouped(exp, key, pred=lambda r: True):
        g = defaultdict(list)
        for r in data.get(exp, []):
            if pred(r):
                g[key(r)].append(r["kernel_seconds"])
        return g

    # ---- serial scaling
    md.append("\n## 2. 单核用时随网格大小\n")
    for exp in ("serial_A", "serial_B"):
        if exp not in data:
            continue
        body = []
        for (w, h, s), v in sorted(grouped(exp, lambda r: (r["W"], r["H"], r["steps"])).items()):
            m, lo, hi = stats(v)
            body.append([f"{w}×{h}", s, f3(m), rng(lo, hi), f"{m / (w * h * s) * 1e9:.2f}"])
        md.append(f"\n{exp}（job {JOBS[exp]}，{SOURCES[[x[0] for x in SOURCES].index(exp)][4]}）\n\n")
        md.append(table(["网格", "步数", "中位数 (s)", "范围 (s)", "ns / 格 / 步"], body))

    # ---- strong scaling
    md.append("\n## 3. E1 强扩展（16384²，200 步，blocking）\n")
    for exp in ("E1", "E1_1node"):
        if exp not in data:
            continue
        base = serial_base(exp)
        g = grouped(exp, lambda r: (r["decomp"], r["ranks"]), lambda r: r["mode"] == "mpi")
        ranks = sorted({k[1] for k in g})
        body = []
        for n in ranks:
            row = [n]
            for d in ("1d", "2d"):
                if (d, n) in g:
                    m, lo, hi = stats(g[(d, n)])
                    row += [f3(m), rng(lo, hi), f"{base / m:.1f}", f"{base / m / n:.1%}"]
                else:
                    row += ["", "", "", ""]
            body.append(row)
        sv = [r["kernel_seconds"] for r in data[exp] if r["mode"] == "serial"]
        md.append(f"\n{exp}（job {JOBS[exp]}），单核中位数 {f3(base)} s（{rng(min(sv), max(sv))}）\n\n")
        md.append(table(["进程", "1D (s)", "1D 范围", "1D speedup", "1D 效率",
                         "2D (s)", "2D 范围", "2D speedup", "2D 效率"], body))

    # ---- weak scaling
    if "E2" in data:
        md.append("\n## 4. E2 弱扩展（每进程 1024² 格，500 步，blocking）\n\n"
                  "效率 = 同一分解方式下 1 进程用时 / N 进程用时（每进程工作量不变，理想情况用时不变）。\n\n")
        g = grouped("E2", lambda r: (r["decomp"], r["ranks"], r["W"], r["H"]))
        t1 = {d: st.median(v) for (d, n, w, h), v in g.items() if n == 1}
        body = []
        for n in sorted({k[1] for k in g}):
            row = [n]
            for d in ("1d", "2d"):
                k = [k for k in g if k[0] == d and k[1] == n]
                if k:
                    m, lo, hi = stats(g[k[0]])
                    row = row if len(row) > 1 else row + [f"{k[0][2]}×{k[0][3]}"]
                    row += [f3(m), rng(lo, hi), f"{t1[d] / m:.1%}"]
            body.append(row)
        md.append(table(["进程", "全局网格", "1D (s)", "1D 范围", "1D 效率",
                         "2D (s)", "2D 范围", "2D 效率"], body))

    # ---- backends
    if "E3" in data:
        base = serial_base("E3")
        md.append(f"\n## 5. E3 四种通信方式（16384²，200 步，job {JOBS['E3']}）\n\n"
                  f"单核中位数 {f3(base)} s。“相对 blocking” = 该方式用时 / 同进程数同分解的 blocking 用时。\n\n")
        g = grouped("E3", lambda r: (r["ranks"], r["decomp"], r["comm"]), lambda r: r["mode"] == "mpi")
        body = []
        for (n, d, c), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                   ["blocking", "nonblocking", "overlap", "rma"].index(kv[0][2]))):
            m, lo, hi = stats(v)
            b = st.median(g[(n, d, "blocking")])
            body.append([n, d, c, f3(m), rng(lo, hi), f"{base / m:.1f}", f"{m / b:.3f}"])
        md.append(table(["进程", "分解", "通信方式", "中位数 (s)", "范围 (s)", "speedup", "相对 blocking"], body))

    # ---- hybrid
    md.append("\n## 6. Hybrid（16384²，200 步，2D，blocking）\n")
    for exp, cores in (("E4", 256), ("E5", 64)):
        if exp not in data:
            continue
        base = serial_base(exp)
        g = grouped(exp, lambda r: (r["ranks"], r["threads"]), lambda r: r["mode"] != "serial")
        pure = st.median(g[(cores, 1)])
        body = []
        for (n, t), v in sorted(g.items(), key=lambda kv: -kv[0][0]):
            m, lo, hi = stats(v)
            body.append([f"{n}×{t}", f3(m), rng(lo, hi), f"{base / m:.1f}", f"{base / m / (n * t):.1%}",
                         f"{m / pure:.3f}"])
        md.append(f"\n{exp}（job {JOBS[exp]}，{cores} 核），单核中位数 {f3(base)} s\n\n")
        md.append(table(["进程×线程", "中位数 (s)", "范围 (s)", "speedup", "效率", "相对纯 MPI"], body))

    # ---- GPU
    if "E6" in data:
        md.append(f"\n## 7. E6 GPU 与 CPU（300 步，5 次，job {JOBS['E6']}）\n\n"
                  "CPU = 同节点一个完整插槽的 36 线程（hybrid 程序，1 进程）。\n\n")
        g = grouped("E6", lambda r: (r["W"], r["mode"]))
        body = []
        for w in sorted({k[0] for k in g}):
            gm, glo, ghi = stats(g[(w, "gpu")])
            cm, clo, chi = stats(g[(w, "hybrid-cpu")])
            cells = w * w * 300
            body.append([f"{w}²", f3(gm), rng(glo, ghi), f3(cm), rng(clo, chi), f"{gm / cm:.2f}",
                         f"{cells / gm / 1e9:.1f}", f"{cells / cm / 1e9:.1f}"])
        md.append(table(["网格", "GPU (s)", "GPU 范围", "CPU (s)", "CPU 范围", "GPU / CPU 用时",
                         "GPU 10⁹ 格/s", "CPU 10⁹ 格/s"], body))

    # ---- cross-job variation
    md.append("\n## 8. 同一配置在不同作业里的用时（16384²，200 步，blocking）\n\n"
              "64 进程只用 1 个节点，256 进程用 4 个节点。比较只应在同一作业内部做。\n\n")
    body = []
    pick = [("E1", "mpi", 1), ("E1_1node", "mpi", 1), ("E3", "mpi", 1), ("recheck", "mpi", 1),
            ("E4", "hybrid", 1), ("E5", "hybrid", 1), ("recheck", "hybrid", 1)]
    for exp, mode, t in pick:
        if exp not in data:
            continue
        nodes = SOURCES[[x[0] for x in SOURCES].index(exp)][4]
        for n in (64, 256):
            for d in ("1d", "2d"):
                v = [r["kernel_seconds"] for r in data[exp] if r["mode"] == mode and r["ranks"] == n
                     and r["threads"] == t and r["decomp"] == d and r["comm"] == "blocking"]
                if v:
                    m, lo, hi = stats(v)
                    body.append([n, d, exp, JOBS[exp], "mpi" if mode == "mpi" else "hybrid, 1 线程",
                                 nodes, f3(m), rng(lo, hi)])
    body.sort(key=lambda r: (r[0], r[1]))
    md.append(table(["进程", "分解", "实验", "作业", "程序", "节点", "中位数 (s)", "范围 (s)"], body))

    if missing:
        md.append("\n## 缺少的文件\n\n" + "\n".join(f"- {e}: `{p}`" for e, p in missing) + "\n")

    # ---- write
    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for exp, rel, *_ in SOURCES:
            for r in data.get(exp, []):
                w.writerow(r)
    text = "".join(md)
    # blank line after every heading, so every Markdown viewer renders the tables
    text = "\n".join(l + "\n" if l.startswith("#") else l for l in text.split("\n"))
    (OUTDIR / "tables.md").write_text(text, encoding="utf-8")
    n = sum(len(v) for v in data.values())
    print(f"{n} runs from {len(data)} files -> {OUTDIR}")
    print("problems:", problems if problems else "none")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
