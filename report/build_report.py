"""Build the project report (PDF and Markdown) from the recorded results.

Usage, from the repository root:
    python3 report/build_report.py [--authors authors.json] [--out DIR]

Tables are computed from the CSV files in Game_of_Life_A/results, gol_B/results and
gol_C/results. authors.json may hold {"authors": [{"name": ..., "student_id": ...}],
"repository_url": ...}. Needs reportlab and pillow.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (CondPageBreak, Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Game_of_Life_A" / "tools"))
import stats                                   # noqa: E402
from validation_suite import report_plan      # noqa: E402

FINAL_MPI_SHA = "80ded24e045d"        # gol_B/src/mpi.c as submitted
EXPERIMENT_MPI_SHA = "256a5201ceda"   # revision recorded by jobs 1529185-1532914


def pct(value):
    return "{:.1f}%".format(100 * value)


def span(rows, digits=3):
    med, low, high = stats.summary(rows)
    return "{:.{d}f}".format(med, d=digits), "{:.{d}f}–{:.{d}f}".format(low, high, d=digits)


class Report:
    """Collects the same content for the PDF and for a Markdown copy."""

    def __init__(self):
        base = getSampleStyleSheet()
        ink = colors.HexColor("#17212b")
        self.styles = {
            "title": ParagraphStyle("title", parent=base["Normal"], fontName="Helvetica-Bold",
                                    fontSize=20, leading=25, spaceAfter=6),
            "h1": ParagraphStyle("h1", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=13.5,
                                 leading=17, spaceBefore=13, spaceAfter=6, keepWithNext=True),
            "h2": ParagraphStyle("h2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=11,
                                 leading=14, spaceBefore=9, spaceAfter=4, keepWithNext=True),
            "text": ParagraphStyle("text", parent=base["Normal"], fontName="Helvetica", fontSize=10,
                                   leading=14, spaceAfter=7, textColor=ink),
            "caption": ParagraphStyle("caption", parent=base["Normal"], fontName="Helvetica",
                                      fontSize=8.8, leading=11.5, spaceAfter=9,
                                      textColor=colors.HexColor("#475666")),
            "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Helvetica", fontSize=8.3,
                                   leading=10.3),
            "code": ParagraphStyle("code", parent=base["Normal"], fontName="Courier", fontSize=8.2,
                                   leading=10.8, backColor=colors.HexColor("#f3f6f8"), borderPadding=6,
                                   spaceBefore=3, spaceAfter=10),
        }
        self.story = []
        self.md = []

    def title(self, text):
        self.story.append(Paragraph(escape(text), self.styles["title"]))
        self.md.append("# " + text + "\n")

    def h1(self, text):
        self.story.append(Paragraph(escape(text), self.styles["h1"]))
        self.md.append("\n## " + text + "\n")

    def h2(self, text):
        self.story.append(Paragraph(escape(text), self.styles["h2"]))
        self.md.append("\n### " + text + "\n")

    def p(self, text, style="text"):
        self.story.append(Paragraph(escape(text), self.styles[style]))
        self.md.append(text + "\n")

    def caption(self, text):
        self.p(text, "caption")

    def code(self, text):
        self.story.append(Preformatted(text, self.styles["code"]))
        self.md.append("```text\n" + text + "\n```\n")

    def image(self, name, caption, width=160 * mm):
        from PIL import Image as PILImage
        path = HERE / "figures" / name
        with PILImage.open(path) as im:
            w, h = im.size
        self.story.append(Image(str(path), width=width, height=width * h / w))
        self.story.append(Spacer(1, 4))
        self.caption(caption)
        self.md.append("![" + caption + "](figures/" + name + ")\n")

    def table(self, caption, headers, rows, widths):
        # Keep the caption on the same page as the first rows. After a heading the
        # heading's own keep-with-next applies, so no break is inserted there.
        previous = self.story[-1] if self.story else None
        if not (isinstance(previous, Paragraph) and getattr(previous.style, "keepWithNext", False)):
            self.need(32 * mm)
        self.caption(caption)
        cells = [[Paragraph(escape(str(c)), self.styles["cell"]) for c in row] for row in [headers] + rows]
        total = sum(widths)
        t = Table(cells, colWidths=[170 * mm * w / total for w in widths], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff4")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#9dadba")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d7e0e8"))]))
        self.story.append(t)
        self.story.append(Spacer(1, 9))
        self.md.append("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" +
                       "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows) + "\n")

    def page(self):
        self.story.append(PageBreak())

    def need(self, height):
        """Start a new page unless this much room is left, so a heading keeps its table."""
        self.story.append(CondPageBreak(height))


def build(authors, repository_url, out_dir):
    r = Report()
    runs = stats.all_runs()
    later = stats.later_method_runs()
    halo = stats.halo_runs()
    a_dir = ROOT / "Game_of_Life_A" / "results"
    scaling = json.loads((a_dir / "serial_scaling" / "summary.json").read_text())
    validation = json.loads((a_dir / "validation_report" / "summary.json").read_text())
    regression = json.loads((a_dir / "verification" / "summary.json").read_text())
    with (a_dir / "validation_report" / "validation_report.csv").open(newline="") as handle:
        tests = list(csv.DictReader(handle))
    consistent, grids_checked = stats.checksums_consistent(runs + later + halo)
    if not consistent or validation["counts"]["FAIL"] or validation["environment"] != "vanda":
        raise RuntimeError("recorded results are inconsistent; refusing to build the report")

    # ---- numbers used in the text ------------------------------------------------
    e1 = [x for x in runs if x["experiment"] == "E1"]
    t1 = stats.serial_median(runs, "E1")
    e1g = stats.grouped([x for x in e1 if x["mode"] == "mpi"], "decomp", "ranks")
    strong = {k: stats.summary(v)[0] for k, v in e1g.items()}
    e2g = stats.grouped([x for x in runs if x["experiment"] == "E2"], "decomp", "ranks")
    weak = {k: stats.summary(v)[0] for k, v in e2g.items()}
    e3 = [x for x in runs if x["experiment"] == "E3" and x["mode"] == "mpi"]
    methods = stats.grouped(e3 + [x for x in later if x["mode"] == "mpi"], "ranks", "decomp", "comm")
    method_median = {k: stats.summary(v)[0] for k, v in methods.items()}
    e1_total = stats.grouped(e1, "mode", "decomp", "ranks")
    serial_total = stats.summary(e1_total[("serial", "na", 1)], "total_seconds")[0]
    total_256 = stats.summary(e1_total[("mpi", "1d", 256)], "total_seconds")[0]
    gpu = stats.grouped([x for x in runs if x["experiment"] == "E6"], "W", "mode")
    g = scaling["groups"]

    # ---- front matter ------------------------------------------------------------
    r.title("Conway's Game of Life with MPI, OpenMP and GPU offload")
    r.caption("DSA5208 Project 3 report")
    if authors:
        r.p("Group members: " + "; ".join("{} ({})".format(a["name"], a.get("student_id") or
                                                             "[student ID to be added]") for a in authors) + ".")
    else:
        r.p("Group members: [names and student ID numbers to be added].")
    if repository_url:
        # No full stop after the address, so copying it gives a working link; clickable in the PDF.
        url = escape(repository_url)
        r.story.append(Paragraph('Source code and README: <link href="{0}" color="#1f5fa8">{0}</link>'.format(url),
                                 r.styles["text"]))
        r.md.append("Source code and README: " + repository_url + "\n")
    else:
        r.p("Source code: [public repository link to be added].")

    r.h1("1. Summary")
    r.p("We implemented Conway's Game of Life in C11 in four forms that share one update kernel: a serial program, "
        "an MPI program with a 1D and a 2D domain decomposition and six ways of exchanging boundary data, an "
        "MPI+OpenMP hybrid, and an OpenMP target-offload version for a GPU. Every version is checked cell by cell "
        "against an independent reference program.")
    r.p("On the Vanda cluster a {0} x {0} grid runs {1:.0f} times faster on 256 cores than on one "
        "(parallel efficiency {2}). Weak scaling keeps {3} efficiency at 256 cores. Efficiency is above 99% up to 16 "
        "cores, falls to between 80% and 85% when one 64-core node is full, and falls again when the job spans several "
        "nodes. "
        "The measurements indicate that neither drop is mainly the cost of sending messages: all six communication methods "
        "agree within 2.5% on one node, and a run with one MPI process and 64 threads, which exchanges no messages "
        "between processes, is as slow as 64 MPI processes. MPI+OpenMP combinations perform like pure MPI at the "
        "same core count. One A40 GPU is twice as fast as 36 CPU cores on the largest grid and slower than them "
        "below 4096 x 4096.".format(16384, t1 / strong[("1d", 256)], pct(t1 / strong[("1d", 256)] / 256),
                                    pct(weak[("1d", 1)] / weak[("1d", 256)])))

    # ---- algorithm ---------------------------------------------------------------
    r.h1("2. The algorithm")
    r.p("Conway's Game of Life evolves a grid of cells, each either alive (1) or dead (0), in discrete generations. "
        "Our grid has H rows and W columns and periodic boundaries: the top edge joins the bottom and the left edge "
        "joins the right, so the grid is a torus and every cell has eight neighbours. These eight surrounding cells "
        "are called the Moore neighbourhood.")
    r.p("Every cell is updated at once from the previous generation. A dead cell with exactly three live neighbours "
        "becomes alive; a live cell with two or three live neighbours stays alive; every other cell is dead in the "
        "next generation. A live cell with fewer than two live neighbours is said to die of underpopulation, and one "
        "with more than three of overcrowding. This rule is written B3/S23 (born on 3, survives on 2 or 3).")
    r.code("n = sum of the eight neighbours in the current generation\n"
           "next[r,c] = (n == 3) or (current[r,c] == 1 and n == 2)")
    r.p("The algorithm is a stencil computation: each cell needs only nearby cells, but it needs them in every "
        "generation. When the grid is divided among processes, cells at the edge of a block depend on cells owned by "
        "another process, so the processes must exchange data every generation and cannot run independently. On "
        "grids narrower than three cells a neighbour position can wrap onto the same cell. We count each of the "
        "eight offsets separately, so the rule stays defined at every size.")
    r.p("Terms used below. A rank is one MPI process. A block is the rectangular part of the grid that one rank "
        "owns. A ghost cell is a local copy of a cell owned by a neighbouring "
        "rank, or of a wrapped cell on the same rank. A halo is the set of ghost cells around a block, and halo "
        "exchange is the communication that refreshes it. A process grid arranges ranks in rows and columns. Pure "
        "MPI means one thread per process; a hybrid run gives each process several OpenMP threads. Kernel "
        "time is the time spent in the generations only, including halo exchange, and excluding start-up, final "
        "collection and file output. Speedup on p cores is the serial kernel time divided by the kernel time on p "
        "cores, and parallel efficiency is speedup divided by p. Strong scaling keeps the grid fixed while p grows; "
        "weak scaling keeps the number of cells per rank fixed, so the ideal time is constant, and weak-scaling "
        "efficiency is the one-rank time divided by the p-rank time. The checksum of a grid is a 64-bit FNV-1a hash "
        "of its cells in row order, so two runs that end with the same grid report the same checksum. A job is one "
        "submission to the cluster's batch system; job numbers show which results were measured together.")
    r.p("Inputs are synthetic. Every timed run starts from a random grid in which each cell is alive with "
        "probability 50%; we call this probability the density. With rows and columns counted from zero, the cell "
        "in row r and column c has the global index i = r x W + c. Its start value is computed from the seed and i "
        "alone, with the mixing function of the SplitMix64 generator in unsigned 64-bit arithmetic:")
    r.code("z = seed + 0x9e3779b97f4a7c15 * (i + 1)\n"
           "z = (z xor (z >> 30)) * 0xbf58476d1ce4e5b9\n"
           "z = (z xor (z >> 27)) * 0x94d049bb133111eb\n"
           "z = z xor (z >> 31)\n"
           "alive = (z mod 1000000) < 500000")
    r.p("No random stream is shared between cells. Each rank can therefore generate its own block, the starting "
        "grid does not depend on the number of ranks or threads, and a grid of any size is produced the same way. "
        "This is how the grids of the weak-scaling experiment grow with the number of processes. All timed runs use "
        "seed 20261026. The validation tests use hand-written patterns and small random grids, described in "
        "Section 5.")

    # ---- serial ------------------------------------------------------------------
    r.h1("3. Serial implementation and runtime")
    r.p("The serial program stores one byte per cell in two arrays, each with one ghost layer on every side. Each "
        "generation fills the ghost layer from the opposite edges, computes all cells from the current array into the "
        "next array, then swaps the two. Two arrays are needed because updating in place would mix generations.")
    r.code("for each generation:\n    copy opposite edges into the ghost layer of current\n"
           "    for each cell: next[r,c] = rule(current, r, c)\n    swap(current, next)")
    r.p("Each generation does O(HW) work, so T generations take O(THW) time, and memory is 2(H+2)(W+2) bytes. The "
        "update of one row is a single function, life_update_row, which the MPI, OpenMP and GPU versions reuse, so "
        "all versions share one implementation of the rule.")
    env = (a_dir / "serial_scaling" / "environment.txt").read_text().splitlines()
    r.p("We timed the serial program for {} square grids from {} to {} cells per side, {} generations, {} repeats "
        "each, on one core of Vanda node {} ({}), built with GCC {} and -O3. Times are medians of the kernel "
        "time.".format(len(g), g[0]["size"], g[-1]["size"], scaling["steps"], g[0]["repeats"], env[2].strip(),
                       env[3].split(":", 1)[1].strip().replace("(R)", ""), env[0].split()[-1]))
    r.table("Table 1. Serial kernel time against grid size (job 1513317).",
            ["Grid", "Cells", "Median time (s)", "Range (s)", "ns per cell update"],
            [["{0} x {0}".format(x["size"]), "{:,}".format(x["cells"]), "{:.3f}".format(x["median_seconds"]),
              "{:.3f}–{:.3f}".format(x["min_seconds"], x["max_seconds"]),
              "{:.2f}".format(x["median_ns_per_cell_update"])] for x in g], [30, 32, 32, 44, 32])
    r.image("03_serial_scaling.png", "Figure 1. Serial kernel time and cost per cell update against grid size, on "
            "logarithmic axes. The dashed line is exactly linear growth from the smallest grid.")
    costs = [x["median_ns_per_cell_update"] for x in g]
    weak_one = stats.core_ns_per_cell(weak[("1d", 1)], e2g[("1d", 1)][0])
    strong_one = stats.core_ns_per_cell(t1, e1_total[("serial", "na", 1)][0])
    r.p("A least-squares fit on the logarithms gives time proportional to cells^{:.3f}, matching the expected linear "
        "growth in the number of cells. Repeats agree closely and produce identical checksums. The cost per cell "
        "update stays between {:.2f} and {:.2f} ns from the smallest grid to the largest. The rule costs the same "
        "whether a cell is alive or dead, because the kernel scans every cell, so we vary grid size and not "
        "density.".format(scaling["exponent_time_vs_cells"], min(costs), max(costs)))
    r.p("The parallel experiments ran on a different CPU model (Intel Xeon Platinum 8452Y), so the absolute times "
        "above should not be compared with them; every speedup below uses a single-core time measured in the same "
        "job. On those nodes one core needs {:.2f} ns per cell update for the 16384 x 16384 grid and {:.2f} ns for a "
        "1024 x 1024 grid, probably because a grid of that size fits in cache. A small block can therefore be about "
        "{:.0f}% cheaper per cell there. This is small compared with the efficiency changes discussed in "
        "Section 7.".format(strong_one, weak_one, 100 * (strong_one - weak_one) / strong_one))

    # ---- design ------------------------------------------------------------------
    r.h1("4. Parallel design")
    r.h2("4.1 Domain decomposition")
    r.p("We divide the grid into blocks and give each rank one block, because the rule is local: a rank can update "
        "its own cells once it has a one-cell-wide copy of its neighbours' edges. Each block is stored like a small "
        "serial grid, with its own ghost layer, so the serial row-update function works unchanged on every rank.")
    r.p("The 1D decomposition gives each rank a band of whole rows. The 2D decomposition arranges the ranks in a "
        "periodic process grid and gives each rank a rectangle. For P ranks we take the factor pair a x b = P with "
        "the smallest difference, with a process rows and b process columns and a <= b; if only the swapped pair "
        "fits the grid we use that, and if no pair fits, every rank exits with an error. Rows and columns are then "
        "divided separately:")
    r.code("length(i) = T/parts + (i < T % parts ? 1 : 0)\noffset(i) = i*(T/parts) + min(i, T % parts)")
    r.p("Every block gets at least one row and column, and block sizes differ by at most one in each direction. "
        "Ranks are numbered row by row, which is what MPI_Cart_create produces with reordering disabled.")
    r.image("fig_decomposition.png", "Figure 2. Left: two 1D row blocks, each with ghost rows above and below and "
            "ghost columns at the sides. Right: a 4 x 4 process grid, where every rank has a ghost ring.",
            width=150 * mm)
    r.p("The two decompositions exchange different amounts of data. In 1D a rank sends its first and last rows, "
        "2(W+2) cells per generation whatever P is. The rows include the wrapped left and right ghost cells, so the "
        "corners arrive with them. In 2D a rank has eight neighbours and sends about 2(H/a + W/b) + 4 cells: four "
        "edges and four single corner cells, which need their own messages. For the 16384 x 16384 grid at P = 256 "
        "this is 32,772 cells per rank in 1D against 4,100 on a 16 x 16 process grid, in two messages against eight. "
        "On small process grids one rank fills several roles (on a 2 x 2 grid, north and south are the same rank), "
        "so message tags name the ghost region a message fills, not the sender.")
    r.h2("4.2 Halo exchange: six methods")
    r.p("To see how the way of sending and receiving affects performance, the program offers six halo-exchange "
        "methods, chosen with --comm: blocking calls, three non-blocking variants, separate sends and receives, and "
        "MPI remote memory access (RMA). All of them fill exactly the same ghost "
        "cells before a generation is computed, so the result does not depend on the method.")
    r.table("Table 2. The six halo-exchange methods.",
            ["--comm", "MPI calls each generation", "How it stays correct"],
            [["blocking", "MPI_Sendrecv, two in 1D and up to eight in 2D",
              "Each call sends to one neighbour and receives from the opposite one, so every message is matched "
              "inside the call that sends it."],
             ["nonblocking", "MPI_Irecv and MPI_Isend, then MPI_Waitall",
              "All receives are posted before any send."],
             ["overlap", "As nonblocking",
              "Interior rows, which need no ghost cells, are computed while the messages are in flight; the border "
              "rows are computed after MPI_Waitall."],
             ["rma", "MPI_Put into a neighbour's window, between two MPI_Win_fence calls",
              "One-sided: a rank writes its edge directly into the neighbour's ghost cells, and the neighbour makes "
              "no receive call. A window is a region of memory that a rank opens for such access. Both grid arrays "
              "are exposed as windows, because the arrays swap every generation."],
             ["persistent", "MPI_Startall on requests created once with MPI_Send_init and MPI_Recv_init, then "
              "MPI_Waitall", "One set of requests per grid array, selected by the parity of the generation."],
             ["sendrecv2", "Separate MPI_Send and MPI_Recv calls",
              "Even-numbered ranks send first and odd-numbered ranks receive first, so the chain of waiting ranks "
              "always ends, also on rings of odd length."]],
            [24, 50, 96])
    r.p("Two mistakes found during development shaped these rules. An early blocking version paired the send and "
        "the receive of one call with the same neighbour; it worked on one rank and deadlocked on two or more. The "
        "first RMA version exposed only one of the two grid arrays, so from the second generation onwards the "
        "neighbours wrote into the array that was no longer current; again one rank passed and several ranks gave "
        "wrong grids. Both were caught by the multi-rank tests in Section 5. A run on one rank says nothing about "
        "the communication pattern.")
    r.h2("4.3 Start-up and collection")
    r.p("With a random start each rank fills its own block from the global cell index, so no data is distributed "
        "and the grid is the same for every number of ranks. With an input file, rank 0 reads it and distributes the "
        "blocks with MPI_Scatterv. At the end rank 0 collects all blocks with MPI_Gatherv, computes the checksum and "
        "the live-cell count, and writes the output. A 2D block is not contiguous in the row-by-row order of the "
        "whole grid, so blocks are packed one after another for these two calls and unpacked on the other side. "
        "Rank 0 therefore holds the whole grid at the start and at the end. This is simple and keeps file access on "
        "one rank, at the cost discussed in Section 9.")
    r.h2("4.4 MPI + OpenMP")
    r.p("The hybrid program is the same source built with OpenMP. Inside each rank the loop over the rows of the "
        "block is shared among threads with a static schedule; rows are independent because they read only the "
        "previous generation. All MPI calls are made by the main thread outside the parallel region "
        "(MPI_THREAD_FUNNELED). With the overlap method the threads compute interior rows while the messages are in "
        "flight.")
    r.h2("4.5 GPU offload")
    r.p("The GPU version is again the same source, built with OpenMP target offload: the compiler turns the marked "
        "row loop into code that runs on the GPU. Below, the host is the CPU side of the program and the device is "
        "the GPU. Both grid arrays are copied to "
        "the GPU once and stay there. In each generation the host fetches only the two edge rows that MPI must "
        "send, and pushes back the two ghost rows that MPI received, so four rows cross between host and GPU instead "
        "of the whole grid. The GPU loop applies the left and right wrap itself. The GPU build supports the 1D "
        "decomposition only; it rejects --decomp 2d, because 2D ghost columns would need a column of every row on "
        "the host. If no GPU is visible the program stops, so a CPU run cannot be reported as a GPU run by accident.")

    # ---- validation --------------------------------------------------------------
    r.h1("5. Validation")
    counts = validation["counts"]
    r.p("Validation asks one question: does every program produce exactly the right final grid? The reference is an "
        "independent Python program that stores only the coordinates of live cells and adds up neighbour "
        "contributions. It shares no code or data structure with the C kernel, so one mistake is unlikely to appear "
        "in both. Known answers for three kinds of pattern were written by hand and checked against the reference "
        "as well. A still life never changes; ours is the 2 x 2 square of live cells known as the block (not to be "
        "confused with the blocks of the decomposition). An oscillator returns to its start after a fixed number of "
        "generations; the blinker, three live cells in a line, does so every two. A glider is a five-cell pattern "
        "that reappears one row and one column further on every four generations.")
    r.p("We report {} tests, chosen so that each covers something the others do not. Table A1 in the appendix lists "
        "every test with the reason for it; in that table Case is the name of the input grid in the test suite, "
        "Layout is the decomposition and P is the number of ranks. Two families are reported as one test each: all 512 states of a 3 x 3 "
        "grid, because that covers every neighbourhood the rule can see, and 84 random grids in 14 "
        "shapes. The random grids range from 1 x 1 to 31 x 37 cells. They are drawn in Python from a generator with "
        "a fixed seed, at densities of 5%, 20%, 50%, 80% and 95%, and given to the programs as input files. The "
        "larger grid of group L is generated by the programs themselves as described in Section 2.".format(len(tests)))
    groups = [("S", "Serial rule", "Known patterns, all 3 x 3 states, random shapes"),
              ("D", "Decomposition", "Process grids, single-row and single-column blocks, uneven blocks, repeated "
               "neighbours, corners"),
              ("C", "Communication method", "The five methods other than blocking, in 1D and 2D"),
              ("H", "Threads", "MPI ranks with OpenMP threads, including more threads than rows"),
              ("L", "Larger grid", "1024 x 768 at 36 ranks, compared byte for byte with the serial output")]
    r.table("Table 3. Reported validation tests by group (job {}).".format(validation["job_id"].split(".")[0]),
            ["Group", "Purpose", "Covers", "Tests", "Result"],
            [[k, name, cover, sum(t["id"][0] == k for t in tests),
              "{} of {} pass".format(sum(t["id"][0] == k and t["status"] == "PASS" for t in tests),
                                     sum(t["id"][0] == k for t in tests))] for k, name, cover in groups],
            [13, 36, 76, 14, 31])
    r.p("Groups S to H compare every cell of the final grid with the reference, and every test also checks the "
        "reported dimensions, rank and thread counts, live-cell count and checksum. Result: {} pass, {} fail. The "
        "persistent and sendrecv2 methods were added last and have four tests each. They run on layouts the other "
        "tests already cover, so their tests target what is new: reuse of the same requests on two alternating "
        "arrays, and the order of sends and receives on rings of odd length, where a wrong order would "
        "deadlock.".format(counts["PASS"], counts["FAIL"]))
    r.p("Further checks support the reported tests. They repeat the same kinds of case on more combinations and all "
        "of them pass. First, {} runs compare every cell with the reference: the 610 grids behind group S run one by "
        "one (14 named cases, the 512 states of the 3 x 3 grid and the 84 random grids), and the 14 named cases on "
        "1, 2, 4 and 8 MPI ranks and, in the hybrid program, on 1, 2 and 4 ranks with 1, 2 and 4 threads, wherever the grid has at "
        "least as many rows as ranks (163 runs). Second, 186 runs compare the checksum and live-cell count with the "
        "serial program for all six methods in both decompositions: three random grids (64 x 64, 53 x 97 and "
        "13 x 5) on 1, 2, 3, 4 and 8 ranks where the grid is large enough for the process grid, and one grid read "
        "from a file on 4 ranks. These two sets ran in the same job as the reported tests. Third, the GPU build was "
        "compared with the serial checksum on an A40 (job 1524542): five grids from 1 x 1 to 97 x 53 on 1 and 4 "
        "ranks with the four earlier methods, 32 runs. Finally, "
        "every timed run in Sections 7 and 8, including the GPU runs, reports the same checksum as every other run "
        "of the same grid and number of generations ({} combinations of grid and generations), and the serial "
        "program is among them wherever it was run.".format(regression["counts"]["PASS"], grids_checked))
    r.p("These tests ran on Vanda on the submitted MPI source file (SHA-256 begins {}). All performance jobs except "
        "the two for the later methods ran before persistent and sendrecv2 were added; the strong-scaling, recheck, "
        "hybrid and GPU jobs recorded the source revision ({}). The submitted file adds the two methods and leaves "
        "the code of the other four unchanged. The tests show that the programs compute the "
        "right grid on the tested configurations. They do not prove the absence of races or deadlocks in general, "
        "since a run can succeed by chance of timing.".format(FINAL_MPI_SHA, EXPERIMENT_MPI_SHA))
    r.image("01_known_patterns.png", "Figure 3. Program output for two known patterns: a block is unchanged and a "
            "blinker returns after two generations. Teal cells are alive.", width=108 * mm)
    r.image("02_periodic_glider.png", "Figure 4. A glider crossing the bottom and right edges. It moves one row and "
            "one column every four generations, with coordinates wrapped around the grid.", width=108 * mm)

    # ---- setup -------------------------------------------------------------------
    r.h1("6. Experimental setup")
    r.p("All timed parallel runs except the halo-timer runs of Section 7.6 used the NUS Vanda cluster. Its CPU "
        "nodes have two Intel Xeon Platinum 8452Y processors with 36 cores each; jobs requested 64 of the 72 cores of each node, so runs of up to 64 "
        "processes use one node, 128 use two and 256 use four. The GPU nodes have the same processors and an NVIDIA "
        "A40. CPU programs were built inside each job with GCC 13.3.0 and Open MPI 5.0.3 (module foss/2024a) using "
        "-O3; the GPU job used the NVIDIA HPC compiler 26.1 for both the GPU program and its CPU comparison. "
        "In the CPU jobs MPI processes were bound to cores, so a process stays on the cores it was given at the "
        "start.")
    r.p("Each configuration was run three times (five for the GPU job) and we report the median, with the minimum "
        "and maximum as the range. Kernel time is taken after all ranks have started together and is the largest "
        "value over the ranks. The serial baseline for a speedup is the serial program timed in the same job. "
        "Speedups and efficiencies are computed from the unrounded medians, so their last digit can differ from a "
        "value recomputed from the rounded times in the tables. Table 4 lists the jobs.")
    r.table("Table 4. Jobs behind the performance results.",
            ["Experiment", "Job", "Nodes", "Content"],
            [["Serial scaling", "1513317", "1 (Xeon Gold 6438N)", "7 grid sizes, 500 generations"],
             ["Strong scaling", "1529185", "4 (GPU nodes, CPUs only)", "16384², 200 generations, 1 to 256 processes, "
              "1D and 2D, blocking"],
             ["Strong scaling, repeat", "1530869", "1 (GPU node, CPUs only)", "Same grid, serial and 2 to 64 "
              "processes, 1D and 2D"],
             ["Weak scaling", "1527955", "4", "1024² cells per process, 500 generations, 1 to 256 processes"],
             ["Communication methods", "1527956", "4 CPU nodes", "Four methods, 16384², 200 generations, 16, "
              "64 and 256 processes"],
             ["Later methods", "1535552, 1535553", "1 and 4", "persistent and sendrecv2, same grid and process "
              "counts"],
             ["1D and 2D recheck", "1532914", "4 CPU nodes", "64 and 256 processes, blocking"],
             ["Hybrid, 64 cores", "1530361", "1 CPU node", "Four splits of 64 cores into processes and threads"],
             ["Hybrid, 256 cores", "1530362", "4 CPU nodes", "Six splits of 256 cores"],
             ["GPU", "1530468", "1 GPU node", "One A40 against 36 CPU threads, four grid sizes, 300 generations"],
             ["Halo timer", "1038437 (Atlas)", "1", "Six methods at 16, 32 and 64 processes, one run each"]],
            [34, 30, 34, 72])

    # ---- MPI performance ---------------------------------------------------------
    r.h1("7. MPI performance")
    r.h2("7.1 Strong scaling")
    r.image("fig_strong_scaling.png", "Figure 5. Strong scaling: kernel time on logarithmic axes with the ideal "
            "line, and parallel efficiency. The shaded area marks runs on more than one node.")
    rows = []
    for p in (2, 4, 8, 16, 32, 64, 128, 256):
        a, b = strong[("1d", p)], strong[("2d", p)]
        rows.append([p, span(e1g[("1d", p)])[0], span(e1g[("1d", p)])[1], "{:.1f}".format(t1 / a), pct(t1 / a / p),
                     span(e1g[("2d", p)])[0], span(e1g[("2d", p)])[1], "{:.1f}".format(t1 / b), pct(t1 / b / p)])
    r.table("Table 5. Strong scaling, 16384 x 16384, 200 generations, blocking exchange. The serial program took "
            "{:.3f} s in the same job. P is the number of processes.".format(t1),
            ["P", "1D time (s)", "1D range", "1D speedup", "1D efficiency",
             "2D time (s)", "2D range", "2D speedup", "2D efficiency"], rows, [10, 18, 25, 16, 18, 18, 25, 16, 18])
    r.p("Speedup is ideal up to 16 processes (efficiency above 99%), about {:.0f} at 64 processes on one full node "
        "({} in 1D) and {:.0f} at 256 processes on four nodes ({}). 1D and 2D differ by less than 4% at every "
        "process count in this job.".format(t1 / strong[("1d", 64)], pct(t1 / strong[("1d", 64)] / 64),
                                             t1 / strong[("1d", 256)], pct(t1 / strong[("1d", 256)] / 256)))
    repeat = stats.grouped([x for x in runs if x["experiment"] == "E1_1node" and x["mode"] == "mpi"],
                           "decomp", "ranks")
    repeat_dev = max(abs(stats.summary(v)[0] / strong[k] - 1) for k, v in repeat.items())
    r.p("The single-node part of this curve is reproducible. A second job on one node of a different allocation "
        "(1530869) repeated the serial run and the runs on 2 to 64 processes: its serial time was {:.3f} s, and "
        "every median was within {:.1f}% of the value in Table 5."
        .format(stats.serial_median(runs, "E1_1node"), 100 * repeat_dev))
    r.need(60 * mm)
    r.h2("7.2 Weak scaling")
    rows = []
    for p in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        sample = e2g[("1d", p)][0]
        rows.append([p, "{} x {}".format(sample["H"], sample["W"]), span(e2g[("1d", p)])[0], span(e2g[("1d", p)])[1],
                     pct(weak[("1d", 1)] / weak[("1d", p)]), span(e2g[("2d", p)])[0], span(e2g[("2d", p)])[1],
                     pct(weak[("2d", 1)] / weak[("2d", p)])])
    r.table("Table 6. Weak scaling, 1024 x 1024 cells per process, 500 generations, blocking exchange.",
            ["P", "Grid (rows x columns)", "1D time (s)", "1D range", "1D efficiency",
             "2D time (s)", "2D range", "2D efficiency"], rows, [10, 32, 18, 25, 19, 18, 25, 19])
    r.p("The grid grows with the number of processes (second column of Table 6), and each grid is generated at its "
        "own size as described in Section 2. "
        "With the work per process fixed, the time stays within 8% of the one-process time up to 16 processes, is "
        "22% to 31% above it at 32 and 64 processes on one node, and about 80% above it on two and four nodes. The "
        "steps occur at the same process counts as in strong scaling.")
    r.image("fig_weak_scaling.png", "Figure 6. Weak scaling: kernel time and efficiency. The ideal time is constant.")
    r.h2("7.3 Where the efficiency is lost")
    r.p("To compare the two experiments we convert each time into the core time spent per cell update: kernel time "
        "x processes / (cells x generations). With perfect scaling this number is the same at every process count.")
    rows = []
    for p in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        cells = []
        for table, grouped in ((strong, e1g), (weak, e2g)):
            for d in ("1d", "2d"):
                if (d, p) in table:
                    cells.append("{:.2f}".format(stats.core_ns_per_cell(table[(d, p)], grouped[(d, p)][0])))
                else:
                    cells.append("{:.2f} (serial)".format(strong_one))
        rows.append([p] + cells)
    r.table("Table 7. Core time per cell update in nanoseconds, from the runs of Tables 5 and 6.",
            ["P", "Strong scaling, 1D", "Strong scaling, 2D", "Weak scaling, 1D", "Weak scaling, 2D"], rows, [14, 32, 32, 32, 32])
    r.image("fig_core_time.png", "Figure 7. Core time per cell update against the number of processes. A flat line "
            "would be perfect scaling.")
    s64, s128, s256 = (stats.core_ns_per_cell(strong[("1d", p)], e1g[("1d", p)][0]) for p in (64, 128, 256))
    r.p("Three regimes are visible. Up to 16 processes the cost per cell does not change. Between 16 and 64 "
        "processes, all on one node, it rises by 15% to 25%. When the job spans two or four nodes it rises again, "
        "to about 4 ns in these two jobs.")
    hy = stats.grouped([x for x in runs if x["experiment"] in ("E4", "E5")], "experiment", "ranks", "threads")
    one_rank = stats.summary(hy[("E5", 1, 64)])[0]
    all_ranks = stats.summary(hy[("E5", 64, 1)])[0]
    few_ranks = stats.summary(hy[("E4", 8, 32)])[0]
    many_ranks = stats.summary(hy[("E4", 256, 1)])[0]
    r.p("The rise inside one node is not the cost of exchanging messages. A run with one process and 64 threads, "
        "which has no neighbouring process to exchange with, takes {:.3f} s, against {:.3f} s for 64 processes with "
        "one thread each (Section 8.1). The six communication methods also agree within 2.5% there (Section 7.4). "
        "A likely cause, which we did not measure directly, is memory bandwidth, the rate at which data can move "
        "between a node's memory and its cores: every generation reads and writes the whole grid, and the cores of "
        "a node share this rate.".format(one_rank, all_ranks))
    blocking = [x for x in runs if x["W"] == 16384 and x["steps"] == 200 and x["comm"] == "blocking"
                and x["mode"] in ("mpi", "hybrid") and x["threads"] == 1]
    by_job = stats.grouped(blocking, "ranks", "decomp", "experiment", "mode")
    t256 = [stats.summary(v)[0] for k, v in by_job.items() if k[0] == 256 and k[3] == "mpi"]
    spread64 = {}
    for d in ("1d", "2d"):
        t64 = [stats.summary(v)[0] for k, v in by_job.items() if k[0] == 64 and k[1] == d]
        spread64[d] = 100 * (max(t64) / min(t64) - 1)
    r.p("The rise across nodes is not a fixed cost per message or per generation either. In strong scaling, going "
        "from 128 to 256 processes halves the block of each process and exactly halves the time ({:.3f} s to "
        "{:.3f} s), so the cost per cell is unchanged ({:.2f} and {:.2f} ns); a fixed delay per generation would "
        "not halve. Reducing the number of processes does not remove it either: on 256 cores, 8 processes with 32 threads "
        "each take {:.3f} s against {:.3f} s for 256 processes, although they exchange far fewer messages. The loss "
        "behaves like computation that is about 40% to 45% slower per cell than on one full node. Its size depends on "
        "the job: 256-process runs with blocking exchange took between {:.2f} s and {:.2f} s in different jobs "
        "(Section 7.5), while 64-process runs on one node agreed within {:.1f}% in 1D and {:.1f}% in 2D across jobs. "
        "Because every rank must wait for its "
        "neighbours' edges in each generation, the slowest rank sets the pace for all, so a single slower node or "
        "core is enough to slow the whole run. We could not isolate the cause with the data we have; running the "
        "same number of processes spread over one, two and four nodes would separate the effect of the network "
        "from the load on each node.".format(strong[("1d", 128)], strong[("1d", 256)], s128, s256, few_ranks,
                                             many_ranks, min(t256), max(t256), spread64["1d"], spread64["2d"]))
    r.h2("7.4 Communication methods")
    rows = []
    for m in stats.METHODS:
        rows.append([m + (" †" if m in stats.LATER_METHODS else "")] +
                    ["{:.3f}".format(method_median[(p, d, m)]) for p in (16, 64, 256) for d in ("1d", "2d")])
    r.table("Table 8. Median kernel time in seconds for the six methods, 16384 x 16384, 200 generations. "
            "† measured in a later job (1535552 and 1535553); the other four in job 1527956.",
            ["Method", "16, 1D", "16, 2D", "64, 1D", "64, 2D", "256, 1D", "256, 2D"], rows, [32, 23, 23, 23, 23, 23, 23])
    r.image("fig_comm_methods.png", "Figure 8. Kernel time of the six methods: median and range of three runs, "
            "logarithmic time axis. Each panel has its own scale.")

    def spread(p, d):
        values = [method_median[(p, d, m)] for m in stats.METHODS]
        return 100 * (max(values) - min(values)) / min(values)

    e3m = {m: (method_median[(256, "1d", m)], method_median[(256, "2d", m)]) for m in stats.METHODS}
    r.p("On one node the method does not matter: the six medians differ by at most {:.1f}% at 16 processes and "
        "{:.1f}% at 64, which is within the run-to-run range. In the later job one of the three runs of each "
        "16-process configuration was about 14% slower, which gives the long ranges for those two methods."
        .format(max(spread(16, "1d"), spread(16, "2d")), max(spread(64, "1d"), spread(64, "2d"))))
    r.p("At 256 processes on four nodes the methods separate, and only comparisons inside one job are meaningful. "
        "In job 1527956, RMA is the slowest method, {:.0f}% slower than blocking in 1D and {:.0f}% in 2D: its two "
        "MPI_Win_fence calls per generation synchronise all ranks. Its 1D runs also vary most ({}). Non-blocking "
        "and overlap are {:.0f}% and {:.0f}% faster than blocking in 2D and within 1.5% of it in 1D. Computing "
        "interior rows while messages are in flight therefore gains little, which fits Section 7.3: if waiting for "
        "messages were a large share of the time, overlap would recover it. The two later methods were measured in "
        "another job, so at 256 processes they cannot be ranked against the first four; at 16 and 64 processes they "
        "match them."
        .format(100 * (e3m["rma"][0] / e3m["blocking"][0] - 1), 100 * (e3m["rma"][1] / e3m["blocking"][1] - 1),
                span(methods[(256, "1d", "rma")])[1] + " s", 100 * (1 - e3m["nonblocking"][1] / e3m["blocking"][1]),
                100 * (1 - e3m["overlap"][1] / e3m["blocking"][1])))
    r.h2("7.5 1D against 2D")
    recheck = stats.grouped([x for x in runs if x["experiment"] == "recheck" and x["mode"] == "mpi"], "decomp", "ranks")
    compare = [("Strong scaling", "1529185", "blocking", strong[("1d", 256)], strong[("2d", 256)])]
    compare += [("Communication methods", "1527956", m, e3m[m][0], e3m[m][1]) for m in stats.METHODS[:4]]
    compare += [("Recheck", "1532914", "blocking", stats.summary(recheck[("1d", 256)])[0],
                 stats.summary(recheck[("2d", 256)])[0])]
    compare += [("Later methods", "1535553", m, e3m[m][0], e3m[m][1]) for m in stats.LATER_METHODS]
    r.table("Table 9. 1D against 2D at 256 processes, 16384 x 16384, 200 generations. Each row compares two "
            "medians from the same job.",
            ["Experiment", "Job", "Method", "1D (s)", "2D (s)", "2D relative to 1D"],
            [[name, job, m, "{:.3f}".format(a), "{:.3f}".format(b),
              "{:.0f}% {}".format(abs(100 * (1 - b / a)), "faster" if b < a else "slower")]
             for name, job, m, a, b in compare], [40, 22, 26, 20, 20, 32])
    r.image("fig_crossjob.png", "Figure 9. The same configuration in different jobs: 64 processes on one node "
            "agree, 256 processes on four nodes do not. Median and range, logarithmic time axis. E1 is the strong-scaling "
            "job, E3 the communication-methods job, E4 and E5 the hybrid jobs with one thread per process; cn and gn-a40 "
            "are the CPU and GPU node types.")
    gains = [100 * (1 - b / a) for _, _, _, a, b in compare]
    r.p("2D sends an eighth of the data at this size, and it was faster at 256 processes in three of the four jobs. "
        "The size of the gain is not reproducible: between {:.0f}% and {:.0f}% in three jobs, and {:.0f}% slower in "
        "the fourth. The spread between jobs is larger than most of these gains. On one node 1D and 2D agree within "
        "0.3% at 16 processes and within 6% at 64 in every job. We conclude that 2D does not hurt and can help on "
        "several nodes, but our data cannot attach one number to the gain."
        .format(min(x for x in gains if x > 0), max(gains), -min(gains)))
    r.h2("7.6 Time inside the halo exchange")
    hg = stats.grouped(halo, "ranks", "decomp", "comm")
    rows = []
    for m in stats.METHODS:
        rows.append([m] + [pct(hg[(p, d, m)][0]["halo_seconds"] / hg[(p, d, m)][0]["kernel_seconds"])
                           for d in ("1d", "2d") for p in (16, 32, 64)])
    r.table("Table 10. Share of kernel time spent inside the halo exchange, measured with an optional timer on "
            "the Atlas cluster (job 1038437, one node, 16384 x 16384, 200 generations, one run per entry).",
            ["Method", "1D, 16", "1D, 32", "1D, 64", "2D, 16", "2D, 32", "2D, 64"], rows, [32, 23, 23, 23, 23, 23, 23])
    r.p("The program can time its halo exchange separately. On Vanda this timer hung in multi-node runs, so it was "
        "left off there and used on one node of the older Atlas cluster instead. The timer counts everything "
        "between entering and leaving the exchange, including time spent waiting for a neighbour that is still "
        "computing. For most configurations the share is below 10%. It is 35% to 46% for three methods in 1D "
        "(blocking, nonblocking and persistent), and in those runs the kernel time itself was 53% to 74% longer "
        "than with the other 1D methods. On Vanda the same methods are within 2.5% of each other on one node "
        "(Table 8), so this slowdown is specific to the Atlas runs, a different cluster and MPI build, and not a "
        "property of the methods. Each entry is a "
        "single run. We therefore use this table only to show that, where no such effect occurs, the exchange takes "
        "a few percent of the kernel time at these block sizes.")

    # ---- MPI + X -----------------------------------------------------------------
    r.h1("8. MPI with OpenMP, and a GPU")
    r.h2("8.1 MPI + OpenMP at a fixed number of cores")
    r.p("We kept the number of cores fixed and changed how they are divided into processes and threads, on the "
        "16384 x 16384 grid with the 2D decomposition and blocking exchange. Threads of one process stay on one "
        "processor (32 cores of each processor were used), except for the single process with 64 threads.")
    rows = []
    for exp, label, splits in (("E5", "64 cores, 1 node", [(64, 1), (16, 4), (4, 16), (1, 64)]),
                               ("E4", "256 cores, 4 nodes", [(256, 1), (128, 2), (64, 4), (32, 8), (16, 16), (8, 32)])):
        base = stats.serial_median(runs, exp)
        pure = stats.summary(hy[(exp, splits[0][0], 1)])[0]
        for ranks, threads in splits:
            med = stats.summary(hy[(exp, ranks, threads)])[0]
            rows.append([label, "{} x {}".format(ranks, threads), span(hy[(exp, ranks, threads)])[0],
                         span(hy[(exp, ranks, threads)])[1], "{:.1f}".format(base / med),
                         pct(base / med / (ranks * threads)), "{:.3f}".format(med / pure)])
    r.table("Table 11. Hybrid runs, 16384 x 16384, 200 generations (jobs 1530361 and 1530362). The last column is "
            "the time relative to pure MPI in the same job.",
            ["Cores", "Processes x threads", "Time (s)", "Range (s)", "Speedup", "Efficiency", "Relative to pure MPI"],
            rows, [30, 28, 18, 28, 18, 20, 28])
    r.image("fig_hybrid.png", "Figure 10. Hybrid runs at 64 and 256 cores: median and range, logarithmic time axis. "
            "The horizontal line is pure MPI.")
    rel64 = [stats.summary(hy[("E5", a, b)])[0] / all_ranks for a, b in ((16, 4), (4, 16), (1, 64))]
    rel256 = [stats.summary(hy[("E4", a, b)])[0] / many_ranks for a, b in ((128, 2), (64, 4), (32, 8), (16, 16), (8, 32))]
    r.p("Every split performs like pure MPI. On one node the hybrid times are within {:.1f}% of it. On four nodes "
        "they lie between {:.1f}% faster and {:.1f}% slower, which is inside the range of repeated runs of a single "
        "configuration. The kernel does the same memory traffic however the cores are divided, and message passing "
        "was already a small share of the time, so replacing processes by threads has nothing to recover. For this "
        "problem the hybrid version offers the same speed with fewer MPI processes, not more speed.".format(100 * (max(rel64) - 1), 100 * (1 - min(rel256)), 100 * (max(rel256) - 1)))
    r.h2("8.2 One GPU against 36 CPU cores")
    rows = []
    ratios = {}
    for size in (2048, 4096, 8192, 16384):
        gm, cm = stats.summary(gpu[(size, "gpu")])[0], stats.summary(gpu[(size, "hybrid-cpu")])[0]
        ratios[size] = gm / cm
        work = size * size * gpu[(size, "gpu")][0]["steps"] / 1e9
        rows.append(["{0} x {0}".format(size), span(gpu[(size, "gpu")])[0], span(gpu[(size, "gpu")])[1],
                     span(gpu[(size, "hybrid-cpu")])[0], span(gpu[(size, "hybrid-cpu")])[1],
                     "{:.2f}".format(gm / cm), "{:.1f}".format(work / gm), "{:.1f}".format(work / cm)])
    r.table("Table 12. One NVIDIA A40 against 36 CPU threads on one processor of the same node, 300 generations, "
            "five runs each (job 1530468). Throughput is in billions of cell updates per second.",
            ["Grid", "GPU time (s)", "GPU range", "CPU time (s)", "CPU range", "GPU / CPU time",
             "GPU throughput", "CPU throughput"], rows, [26, 19, 25, 19, 25, 20, 18, 18])
    r.image("fig_gpu_cpu.png", "Figure 11. GPU and CPU kernel time against grid size on logarithmic axes, and "
            "their ratio. Below 1 the GPU is faster.")
    r.p("The GPU is {:.2f} times slower than the 36 cores on the 2048 x 2048 grid, about equal at 4096 x 4096 and "
        "{:.1f} times faster at 16384 x 16384. Each generation has fixed costs on the GPU: starting the offloaded "
        "loop and moving the four halo rows between host and device. On a small grid these dominate; on a large "
        "grid they are spread over more cells, and the GPU throughput rises from {} to {} billion cell updates per "
        "second while the CPU stays between 9 and 11. The GPU runs produced the same checksums as the CPU runs. We tested "
        "one process with one GPU; several GPUs, and the 2D decomposition on a GPU, were not tested."
        .format(ratios[2048], 1 / ratios[16384], rows[0][6], rows[3][6]))

    # ---- discussion --------------------------------------------------------------
    r.h1("9. Discussion")
    r.p("How and why we parallelised. The rule is local, so the natural parallelisation is domain decomposition "
        "with one layer of ghost cells: each rank owns a block, and one exchange of block edges per generation gives "
        "it everything it needs. Two arrays per rank keep generations separate, exactly as in the serial program. "
        "Building every version around the same row-update function meant that the parallel work was only about "
        "moving edges correctly, and that all versions could be validated against one reference. Inside a rank the "
        "rows are independent, which is why OpenMP threads and a GPU loop can be added without changing the "
        "communication.")
    halo_1d = 2 * (16384 + 2)
    halo_2d = 2 * (1024 + 1024) + 4
    r.p("Where communication is greatest. Per generation, the communication is the halo exchange. In 1D each rank "
        "sends {:,} cells per generation on the 16384-wide grid whatever the number of ranks, so relative to the "
        "work of a rank it grows as blocks get thinner: at 256 ranks a block has 64 rows and the halo is {} of its "
        "cells. In 2D at 256 ranks a rank sends {:,} cells, {} of its block. The ratio is largest for thin 1D blocks, "
        "that is, for small grids on many ranks. The largest single transfer is not in the kernel at all: at the "
        "end, all {:.0f} MB of the grid are collected on rank 0."
        .format(halo_1d, pct(halo_1d / (64 * 16384)), halo_2d, pct(halo_2d / (1024 * 1024)), 16384 * 16384 / 1e6))
    outside = total_256 - strong[("1d", 256)]
    r.p("What limits performance. Message passing does not appear to be the main limit in the configurations we "
        "tested: where we could time it, it is a few percent of the kernel time at our block "
        "sizes, six methods and two decompositions change little, and replacing processes by threads changes "
        "nothing. The efficiency losses are more consistent with contention for the shared resources of a node, "
        "and with variation between nodes "
        "combined with the fact that the halo exchange makes every rank wait for the slowest one. We did not "
        "isolate their precise causes (Section 7.3). Outside the "
        "kernel, the serial part matters: for the strong-scaling grid, start-up and collection take {:.2f} s at 256 "
        "processes, about as long as the 200 generations ({:.2f} s). The kernel speedup at 256 processes is {:.0f}, "
        "but the speedup of the whole run is {:.0f}. Collecting the grid through one rank is the first thing to "
        "replace for longer or larger runs, for example with MPI parallel file output."
        .format(outside, strong[("1d", 256)], t1 / strong[("1d", 256)], serial_total / total_256))
    r.p("Limitations. Strong scaling was measured on one grid size; smaller grids ran too briefly at 256 processes "
        "to time reliably. Times at 256 processes vary between jobs, so we compare configurations only inside a "
        "job, and we did not isolate the cause of the multi-node loss. The halo timer could not be used on Vanda "
        "across nodes, and the Atlas timer data is one run per configuration. All performance jobs except those for "
        "the two later methods ran the four-method revision of the MPI source, and the GPU build was tested with "
        "those four methods only.")

    r.h1("10. AI usage")
    r.p("AI coding assistants, including OpenAI Codex and Anthropic Claude Code, were used for planning, for drafting C code, "
        "tests, job and analysis scripts and figures, for checking results, and for drafting and editing this "
        "report. All numbers in the report come from runs that group members submitted on Vanda and Atlas; the "
        "tables are computed from the recorded result files by scripts in the repository. The group reviewed the "
        "code and the text and is responsible for them.")

    # ---- appendix ----------------------------------------------------------------
    r.h1("Appendix A. Every reported validation test")
    plan = {t["id"]: t for t in report_plan()}
    r.table("Table A1. The {} reported tests. S = serial rule, D = decomposition, C = communication method, "
            "H = threads, L = larger grid.".format(len(tests)),
            ["ID", "Case", "Layout", "Method", "Ranks x threads", "Why it exists", "Result"],
            [[t["id"], t["case"], t["decomp"] or "serial", t["comm"] or "-",
              "{} x {}".format(t["ranks"], t["threads"]), plan[t["id"]]["purpose"], t["status"]] for t in tests],
            [10, 27, 12, 20, 17, 70, 14])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667584"))
        canvas.drawString(20 * mm, 13 * mm, "DSA5208 Project 3 | Game of Life with MPI, OpenMP and GPU offload")
        canvas.drawRightString(190 * mm, 13 * mm, str(doc.page))
        canvas.restoreState()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "DSA5208_Project3_Report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=20 * mm, leftMargin=20 * mm, topMargin=18 * mm,
                            bottomMargin=20 * mm, title="Conway's Game of Life with MPI, OpenMP and GPU offload",
                            author="DSA5208 Project 3 group")
    doc.build(r.story, onFirstPage=footer, onLaterPages=footer)
    (out_dir / "DSA5208_Project3_Report.md").write_text("\n".join(r.md), encoding="utf-8")
    print("wrote", pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", type=Path, help="JSON file with authors and repository_url")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    meta = json.loads(args.authors.read_text(encoding="utf-8")) if args.authors else {}
    authors = [a for a in meta.get("authors", []) if a.get("name")]
    build(authors, meta.get("repository_url", ""), args.out)


if __name__ == "__main__":
    main()
