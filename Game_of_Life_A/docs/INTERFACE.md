# Game of Life interface, version 1.1

Version 1.0 is below and unchanged. Version 1.1 adds 2D decomposition and the
communication flags in the next section. Adapt commands in config/variants.json
if needed; do not change the mathematical semantics between versions.

## Version 1.1: 2D decomposition

include/life_decomp.h is the layout contract. It has no MPI dependency and is
tested without a launcher. gol_B/src/mpi.c currently keeps its own copy of the
same formulas; the two must agree, and a test table is in docs/VALIDATION_SUITE.md.

- CLI: `--decomp 1d|2d` (default 1d),
  `--comm blocking|nonblocking|overlap|rma|persistent|sendrecv2` (default blocking).
  New communication methods add values to `--comm`.
- JSON: add `decomp` and `comm`. All version 1.0 fields keep their meaning.
- Process grid: for P ranks choose the factor pair a*b=P with the smallest
  difference, a<=b, a process rows, b process columns, when a<=H and b<=W. If only
  the swapped pair fits, use it. Otherwise every rank exits with an error.
- Ranks are row-major: rank = process_row*process_cols + process_col, which is
  MPI_Cart_create with periods {1,1} and reorder 0.
- Block sizes use the 1.0 formula separately for rows and columns. Each block is a
  LifeGrid whose rows AND cols are local, with ghosts on all four sides.
- Ghosts: north, south, west, east segments plus four single corner cells. On small
  process grids a neighbour can be this rank or appear in several directions, so
  message tags must identify the ghost segment they fill.
- Random start: use life_fill_block, whose global index is (row_offset+r)*W +
  col_offset+c. life_fill_random assumes whole rows and is for serial and 1D only.
- Scatterv/Gatherv: block-major order, as in life_pack_blocks and life_unpack_blocks.
  2D blocks are not contiguous in the row-major global grid.

As before, a P=1 run must give the same grid as the serial program, at any setting.


## Mathematical problem

A height H by width W binary grid uses toroidal (periodic) boundaries. The
eight Moore offsets are (-1,-1) through (1,1), excluding (0,0). Every generation
reads only the previous generation. A dead cell is born with exactly three live
neighbours; a live cell survives with two or three (B3/S23).

For H or W below 3, count the eight offset positions with multiplicity after
wrapping; do not deduplicate coordinates. This makes tiny tests well defined.
For example, an alive 1x1 grid has eight live offset-neighbours and dies.

## Representation and shared kernel

See include/life.h and src/life.c. LifeGrid contains rows, cols, stride and cells.
rows is the LOCAL height in MPI; cols is the GLOBAL width; stride = cols+2.
The allocated uint8_t array has (rows+2)*(cols+2) entries. Owned cells use
one-based indices r=1..rows, c=1..cols. Values are always 0 or 1.

The top/bottom ghost rows are r=0 and r=rows+1. Ghost columns are c=0, cols+1.
Full-grid files and checksums exclude all ghosts.

Serial iteration:

```c
life_wrap_periodic(&current);
life_step(&current, &next);
life_swap(&current, &next);
```

MPI/Hybrid iteration contract:

1. Fill left/right ghosts of owned rows with life_wrap_horizontal.
2. Exchange stride bytes (including corner/ghost columns): send first owned row
   to the previous rank and receive the next rank's first row into the bottom
   ghost; send last owned row to the next rank and receive the previous rank's
   last row into the top ghost. Use separate tags for these two directions.
3. Update owned rows with life_update_row. C may parallelize that row loop.
4. Finish every thread's writes, then swap the grids outside the parallel loop.

The first rank's previous neighbour is the last rank. P=1 and P=2 need testing:
in P=2 the upper and lower neighbour are the same rank. Send/receive buffers
must not overlap. MPI_Sendrecv can implement the two exchanges safely.

Use MPI_Init_thread with MPI_THREAD_FUNNELED, check the provided support level,
and keep MPI calls on the initializing thread. Put MPI calls outside OpenMP
parallel regions for the baseline. Do not add nowait before swapping arrays.
Official reference: https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Init_thread.3.html

## Decomposition

For rank p in 0..P-1:

```text
local_rows(p) = H/P + (p < H%P ? 1 : 0)
global_offset(p) = p*(H/P) + min(p, H%P)
```

All divisions above are integer divisions. Baseline requires 1 <= P <= H.
Reject P>H collectively, with a clear error; zero-row ranks are not supported by
this kernel contract. Guard MPI int counts/displacements before casting.
Broadcast root input-validation failures before all ranks enter collectives.

The in-memory slab test validates this layout and corner exchange under uneven
partitions and local_rows=1. It does not test MPI message matching or deadlocks.

## Reproducible initialization

Random mode uses a stateless SplitMix64-style mapping. Global cell index is
r_global*W+c. Pass global_offset to life_fill_random for local slabs. The same
seed and density generate the same full input at every process/thread count.
The exact formula is in life_initial_cell; all arithmetic is unsigned 64-bit.
Do not use independent libc rand streams per rank.

Input mode reads a P1 PBM: magic P1, WIDTH HEIGHT, then exactly H*W binary tokens;
# comments and ASCII whitespace are allowed. 1 means alive/black. Reject extra
or missing pixels. Rank 0 can load and distribute this fixed full grid.

## Common CLI

```text
--rows H --cols W --steps N --seed S --density-ppm D --output final.pbm
--input initial.pbm --steps N --output final.pbm
```

N>=0, unsigned 64-bit S, D in [0,1000000]. Seed/density are invalid with --input.
Explicit dimensions with --input must match its header. Rank 0 writes one
complete global P1 PBM, after gathering owned rows in their GLOBAL row order.
Use text-mode PBM for correctness/export; never write grid files each timed step.

Only rank 0 writes one JSON object to stdout; diagnostics go to stderr. Include:

```json
{"interface_version":"1.0","variant":"mpi","rows":17,"cols":19,"steps":8,
 "init":"pbm","seed":null,"density_ppm":null,"ranks":4,"threads":1,
 "live_cells":0,"checksum_fnv1a64":"0000000000000000",
 "kernel_seconds":0.01,"total_seconds":0.02}
```

The numbers/checksum above illustrate the schema, not a measurement or correct
checksum. Valid variant strings: serial, mpi, hybrid. threads is threads PER
rank; MPI-only uses 1. FNV-1a-64 hashes interior 0/1 bytes in global row-major
order. Compute it on the gathered grid; do not XOR rank-local FNV hashes.

## Timing and resource contract

- kernel_seconds: all iterations, including periodic/halo preparation,
  communication, update, synchronization and swapping; exclude initialization,
  final gathering, diagnostics and file output.
- total_seconds: allocation + initialization/input + iterations + final
  gathering and diagnostics; exclude final PBM/JSON writing. This field is a
  defined simulation interval, not process-launch or queue wall time.
- Parallel timing uses a synchronized start and max elapsed duration over
  ranks, with the same scope. Do not report rank 0 duration alone.
- Serial uses CLOCK_MONOTONIC and identical scopes. Report the chosen scope
  when computing speedup; never mix kernel and total times.
- Keep compiler optimization comparable. Record real compiler/build logs,
  CPU/node layout, ranks, threads, binding settings, job ID and source/binary hashes.
- ranks*threads must fit allocated CPU cores. Runs at 256 cores span several
  nodes; Vanda nodes have 72 cores. Record the node count and placement.

## Integration check

Place B/C sources with this project; build executables as build/mpi and
build/hybrid, or edit required_paths and command arrays in config/variants.json.
The collector records available project sources and actual binary hashes; keep
the external sources/build logs too. Run:

```bash
python3 tools/verify.py --require-all
```

Exit 0 means all requested variants passed; 1 means a failed test; 2 means an
external variant is unavailable. The default without --require-all returns 0
when serial passes even if external variants are BLOCKED; inspect summary.json.
It never labels unexecuted MPI/Hybrid work PASS.
