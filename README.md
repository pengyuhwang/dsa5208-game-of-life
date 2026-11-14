# Conway's Game of Life with MPI, OpenMP and GPU offload

DSA5208 Project 3. Conway's Game of Life (rule B3/S23 on a periodic grid) in C11 as

- a serial program,
- an MPI program with a 1D or 2D domain decomposition and six halo-exchange methods,
- an MPI+OpenMP hybrid (the same source built with OpenMP),
- an OpenMP target-offload build for one GPU.

All versions share one update kernel (`Game_of_Life_A/src/life.c`) and are checked
cell by cell against an independent Python reference.

## Layout

```
Game_of_Life_A/   serial program, shared kernel, reference checker, validation tools
  serial.c, src/life.c, include/life.h     serial program and kernel
  src/life_decomp.c, include/life_decomp.h 2D layout rules, tested without MPI
  tools/oracle.py                          independent reference (sparse set of live cells)
  tools/validation_suite.py                the 98 tests described in the report
  tools/verify.py                          wider regression (773 runs with MPI and hybrid)
  tools/serial_scaling.py, plot_scaling.py serial runtime against grid size
  docs/INTERFACE.md, docs/VALIDATION_SUITE.md
  results/                                 recorded Vanda results (CSV, JSON)
gol_B/            MPI, hybrid and GPU driver
  src/mpi.c                                one source for build/mpi, build/hybrid, build/gpu
  scripts/test_comm.sh, test_gpu.sh        checksum matrices against the serial program
  scripts/bench_atlas.sh, bench_gpu.sh, halo_stats.sh   benchmark drivers
  INTERFACE.md                             command line, JSON output, message tags
  results/                                 recorded results (CSV)
gol_C/            experiment drivers and merged results
  bench_hybrid.sh                          hybrid runs at a fixed number of cores
  merge_results.py                         merges every result file, checks it, writes tables
  results/                                 recorded results, all_runs.csv, tables.md
report/           scripts and figures of the report
```

The directory names are fixed: `gol_B/Makefile` expects `../Game_of_Life_A`.

## Requirements

- A C11 compiler and Make; Python 3.6 or newer (standard library only for the tests).
- An MPI library with `mpicc` and `mpirun` for the parallel programs. The recorded runs
  used GCC 13.3.0 and Open MPI 5.0.3 (`module load foss/2024a` on the NUS Vanda cluster).
- For the GPU build, a compiler with OpenMP device offload. The recorded runs used the
  NVIDIA HPC compiler 26.1 (`module load nvhpc-hpcx-cuda12/26.1`).
- For the report and figures only: `pip install reportlab pillow matplotlib`.

## Build

```bash
make -C Game_of_Life_A all        # Game_of_Life_A/build/serial
make -C gol_B all                 # gol_B/build/mpi and gol_B/build/hybrid
make -C gol_B install-a           # also copy both into Game_of_Life_A/build for the tests
make -C gol_B gpu MPICC=mpicc GPUFLAGS=-mp=gpu   # gol_B/build/gpu, NVIDIA compiler
```

## Run

```bash
# serial
Game_of_Life_A/build/serial --rows 1024 --cols 1024 --steps 500 --seed 42 --output final.pbm

# MPI: 2D decomposition, non-blocking exchange
mpirun -np 16 gol_B/build/mpi --rows 4096 --cols 4096 --steps 200 --seed 42 \
    --decomp 2d --comm nonblocking

# MPI + OpenMP: 4 processes with 8 threads each
OMP_NUM_THREADS=8 mpirun -x OMP_NUM_THREADS -np 4 gol_B/build/hybrid \
    --rows 4096 --cols 4096 --steps 200 --decomp 2d

# start from a file instead of a random grid
mpirun -np 8 gol_B/build/mpi --input initial.pbm --steps 100 --output final.pbm
```

Options: `--rows`, `--cols`, `--steps`, `--seed`, `--density-ppm` (default 500000 = 50%),
`--input FILE.pbm`, `--output FILE.pbm`, and for the MPI programs `--decomp 1d|2d`
(default `1d`) and `--comm blocking|nonblocking|overlap|rma|persistent|sendrecv2`
(default `blocking`). Files are plain P1 PBM, 1 = alive. Each run prints one JSON record
with the grid size, rank and thread counts, live-cell count, an FNV-1a checksum of the
final grid, `kernel_seconds` (the generations only) and `total_seconds`.

With Open MPI, pass `-x OMP_NUM_THREADS` to `mpirun`. Without it the hybrid program can
run with one thread per process on remote nodes; the result is still correct, so the
mistake shows only in the `threads` field of the JSON record.

The GPU build supports `--decomp 1d` only and stops if no GPU is visible.

## Tests

```bash
make -C Game_of_Life_A test       # serial program against the reference, and unit tests
make -C Game_of_Life_A sanitize   # the same cases under AddressSanitizer and UBSan

# parallel programs: build, install, then run from Game_of_Life_A
make -C gol_B install-a
cd Game_of_Life_A
python3 tools/validation_suite.py --run --launcher "mpirun -x OMP_NUM_THREADS" --require-all
python3 tools/verify.py --require-all
cd ../gol_B && bash scripts/test_comm.sh      # six methods x 1D/2D x 1,2,3,4,8 ranks
bash scripts/test_gpu.sh                      # GPU build, needs a GPU
```

`validation_suite.py` runs the 98 tests listed in `Game_of_Life_A/docs/VALIDATION_SUITE.md`,
each with its purpose. A missing MPI or hybrid program is reported as BLOCKED, never as a
pass. The largest test needs 36 cores.

Recorded results on Vanda for the submitted code: 98 of 98 reported tests, 773 of 773
regression runs and 186 of 186 runs of `test_comm.sh` pass
(`Game_of_Life_A/results/validation_report`, `Game_of_Life_A/results/verification`).

## Experiments

All timings are `kernel_seconds`, three runs per configuration (five on the GPU), seed
20261026, density 50%. The drivers build nothing; build first.

| Experiment | Driver and settings | Recorded result |
| --- | --- | --- |
| Serial runtime against size | `Game_of_Life_A/tools/serial_scaling.py`, 256² to 16384², 500 generations | `Game_of_Life_A/results/serial_scaling/` |
| Strong scaling | `gol_B/scripts/bench_atlas.sh`, `GRIDS=16384 STEPS=200 RANKS="2 4 8 16 32 64 128 256" DECOMPS="1d 2d" COMMS=blocking` | `gol_C/results/e1_strong.csv` |
| Weak scaling | `bench_atlas.sh`, 1024² cells per process, 500 generations, 1 to 256 processes | `gol_B/results/vanda_e2_weak_256.csv` |
| Four communication methods | `bench_atlas.sh`, `COMMS="blocking nonblocking overlap rma"`, 16, 64 and 256 processes | `gol_B/results/vanda_e3_backends_256.csv` |
| Two later methods | `bench_atlas.sh`, `COMMS="persistent sendrecv2"` | `gol_B/results/e4_newcomm_p1.csv`, `e4_newcomm_p2.csv` |
| 1D and 2D recheck | `bench_atlas.sh`, 64 and 256 processes | `gol_C/results/e2d_recheck.csv` |
| Hybrid at 64 and 256 cores | `gol_C/bench_hybrid.sh`, `PAIRS="64x1 16x4 4x16 1x64"` and `"256x1 128x2 64x4 32x8 16x16 8x32"` | `gol_C/results/e5_hybrid.csv`, `e4_hybrid.csv` |
| GPU against 36 CPU threads | `gol_B/scripts/bench_gpu.sh`, `GRIDS="2048 4096 8192 16384" STEPS=300 REPEAT=5` | `gol_C/results/e6_gpu_1530468.csv` |
| Time inside the halo exchange | `gol_B/scripts/halo_stats.sh` (sets `GOL_HALO_STATS=1`) | `gol_B/results/halo_atlas.csv` |

The `*_meta.txt` files next to the results record the job, nodes, compiler and source
hashes. The cluster runs used 64 processes per node with `mpirun --bind-to core`, built
the programs inside the job, and for several nodes needed a site-specific `ssh` wrapper
for `mpirun`. Batch-system scripts are not part of this repository.

The halo timer (`GOL_HALO_STATS=1`) is for diagnosis only. It hung in multi-node runs on
Vanda, so leave it off for timing.

## Tables, figures and report

```bash
python3 gol_C/merge_results.py     # checks all result files, rewrites all_runs.csv and tables.md
python3 report/make_figures.py     # fig_comm_methods.png, fig_core_time.png
python3 report/build_report.py --out report/out   # report PDF and Markdown
```

`report/build_report.py` computes every table of the report from the recorded CSV files.
The other figures in `report/figures/` were drawn from the same files:
`03_serial_scaling.png` by `Game_of_Life_A/tools/plot_scaling.py`, the two pattern figures
by `Game_of_Life_A/tools/figures.py`, and the block diagram by `gol_B/scripts/plot_results.py`.

## AI usage

AI coding assistants were used for planning, drafting code, tests, scripts and figures, and
for drafting the report. All results come from runs submitted by the group on the Vanda and
Atlas clusters. The group reviewed the code and is responsible for it.
