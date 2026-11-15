# Member B — MPI Implementation and Communication Design

Correctness counts were recorded on Atlas (2026-11-05); the six-backend
matrix was re-run there on 2026-11-10, together with the halo-cost batch.
Performance numbers come from the final Vanda `batch_cpu` batches under
`foss/2024a` (jobs 1527052 preflight, 1527951/52/55 weak,
1527953/54/56 strong+backends, 2026-11-07; 1535552/53 backfill for the two
later backends, 2026-11-10); figures live in
`gol_B/report/figures/`, raw data in `gol_B/results/*.csv`.

## 1. Scope and interface contract

This section covers the MPI driver (`src/mpi.c`) and the hybrid MPI+OpenMP
variant (the same source built with `-DGOL_USE_OPENMP`). The grid type, update
rule, initialisation, PBM I/O and checksum are taken from A's `include/life.h`
and `src/life.c`; B's code contains only the parallel decomposition,
communication, collectives and diagnostics. There is therefore one
implementation of the mathematical semantics for all three variants.

Interface 1.0, as proposed in `Game_of_Life_A/docs/INTERFACE.md`:

| Item | Convention |
|---|---|
| Grid storage | `LifeGrid`, `uint8_t`, `stride = cols + 2`, owned rows `1..rows`, columns `1..cols` |
| Boundaries | toroidal, B3/S23, eight Moore offsets counted with multiplicity |
| Initial state | SplitMix64-style `life_initial_cell` over global index `r_global*W + c` |
| Random CLI | `--rows H --cols W --steps N --seed S --density-ppm D --output final.pbm` |
| Input CLI | `--input initial.pbm --steps N --output final.pbm` |
| Output | one complete P1 PBM written by rank 0 |
| Diagnostics | one JSON object on stdout from rank 0 |
| Timing | `kernel_seconds` (iterations) and `total_seconds` (initialisation + iterations + gather) |
| Threading | `MPI_Init_thread` with `MPI_THREAD_FUNNELED`; `OMP_NUM_THREADS` sets hybrid threads |

The hybrid binary reports `threads = omp_get_max_threads()` in the JSON, which
is the value exported by `OMP_NUM_THREADS`.

## 2. Data layout and 1D row-block decomposition

The domain is split into contiguous row blocks. With `P` ranks and `H` rows:

```
base = H / P          rem = H % P
local_rows(p) = base + (p < rem ? 1 : 0)
global_offset(p) = p * base + min(p, rem)
```

The first `H` modulo `P` ranks receive one extra row. The program requires
`1 <= P <= H` and rejects the whole job otherwise, because the kernel contract
does not support zero-row ranks.

Each rank owns one `LifeGrid` with `rows = local_rows`, `cols = W` (the global
width) and `stride = W + 2`. The allocation is `(local_rows + 2) * (W + 2)`:

```
row 0            : top ghost, c = 0..W+1
row 1..local_rows: owned rows, c = 1..W
row local_rows+1 : bottom ghost, c = 0..W+1
```

Columns `0` and `W+1` are horizontal ghosts on every row, including the two
vertical ghosts. Exchanging complete `stride`-byte rows therefore moves the four
corner cells along with the halo, and no extra corner messages are needed.

Memory per rank is `O(HW/P + W)`. Rank 0 holds the full grid only while reading
the PBM input and after the final gather.

![Block and halo layout. Left: two 1D row blocks, each with one ghost row above and below and one ghost column on each side; the arrows mark the two stride rows exchanged per step. Right: the 2D process grid, where every rank owns a ghost ring and the halo travels as eight tagged edge segments plus four corner transfers.](figures/decomp.png)

## 3. Halo exchange

Each iteration first fills the left and right ghost columns of every owned row:

```c
life_wrap_horizontal(&current);   /* c=0 and c=W+1 */
```

It then exchanges the first and last owned rows. With periodic neighbours
`up = (rank - 1 + P) % P` and `down = (rank + 1) % P`:

```c
/* first owned row -> up; receive down's first row into my bottom ghost */
MPI_Sendrecv(current + stride, stride, MPI_UINT8_T, up, TAG_UP,
             current + (local_rows + 1) * stride, stride, MPI_UINT8_T, down, TAG_UP,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

/* last owned row -> down; receive up's last row into my top ghost */
MPI_Sendrecv(current + local_rows * stride, stride, MPI_UINT8_T, down, TAG_DOWN,
             current, stride, MPI_UINT8_T, up, TAG_DOWN,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

Two details are essential:

1. **Each call pairs a send with a receive from the opposite neighbour.** The
   two calls together form a ring, so every message is matched inside the call
   that issues it. An earlier version paired both the send and the receive of
   the first call with the upper neighbour. That neighbour sends its `TAG_DOWN`
   message only in its second call, so every rank waited for a message that
   could not yet exist and `P >= 2` deadlocked. The failure was found by running
   the matrix, not by compiling. "It runs with `P = 1`" is not evidence that the
   communication pattern is correct.
2. **The unit is the complete `stride`-byte row, not the `W` interior bytes.**
   The two extra bytes carry the corner ghost cells.

For `P = 1` the two receives are replaced by `memcpy` of the last owned row into
the top ghost and the first owned row into the bottom ghost, which avoids a
self-send. For `P = 2` the upper and lower neighbour are the same rank; the two
tags and the non-overlapping send/receive buffers make this safe. A local block
of one row is handled without a special case: both ghosts become copies of that
row.

## 4. Collective communication

Only rank 0 reads the PBM input, generates nothing in input mode, and writes the
final grid. Initialisation stays deterministic and file I/O stays off the other
ranks.

- **Distribution.** Rank 0 packs the interior of its full `LifeGrid` into a
  contiguous `H*W` buffer (the stride rows contain padding, so the interior is
  not contiguous). `MPI_Scatterv` then sends `local_rows*W` bytes to each rank
  with element counts and displacements computed from the same partition
  formula. Each rank unpacks into its `LifeGrid`.
- **Collection.** Each rank packs its owned rows into a contiguous buffer;
  `MPI_Gatherv` places them in global row order on rank 0, which unpacks into a
  full grid for diagnostics and PBM output.

`Scatterv`/`Gatherv` express the uneven partition directly and use one collective
per phase. The packing cost is `O(HW)` per rank at the boundaries of the run and
is outside the timed kernel.

In random mode each rank fills its own slab directly with
`life_fill_random(grid, seed, density, global_offset)`, so no full grid and no
distribution are needed. The global row offset is what makes the same seed
produce the same input for every process count.

## 5. Double buffering and the update loop

Each rank allocates two grids of identical shape. The loop is:

```
for step in 0 .. steps-1:
    life_wrap_horizontal(current)     # horizontal ghosts
    exchange_halo(current)            # two Sendrecv calls
    update owned rows: next <- f(current)
    swap(current, next)
```

`life_update_row` reads only the previous buffer, so no cell is updated in
place. The next iteration overwrites the horizontal and vertical ghosts before
they are read again, so stale halo data is never used. Grids are swapped after
the loop, outside any OpenMP region.

With `GOL_USE_OPENMP`, only the row loop is parallelised:

```c
#pragma omp parallel for schedule(static)
for (long long r = 1; r <= local_rows; ++r)
    life_update_row(previous->cells, next->cells, stride, r, cols);
```

All MPI calls are made by the initialising thread outside parallel regions,
which is exactly what `MPI_THREAD_FUNNELED` permits.

## 6. Communication backends and 2D decomposition

The driver now exposes two orthogonal options (defaults preserve the baseline
exactly): `--decomp 1d|2d` and
`--comm blocking|nonblocking|overlap|rma|persistent|sendrecv2`. The
full specification, message tags and synchronization protocol are in
`INTERFACE.md`; the essential points:

**Backends.** All six fill identical ghost bytes before any update, so the
final grid is backend-independent.

| Backend | MPI calls per step | Synchronisation | Expected overlap | Constraints |
|---|---|---|---|---|
| blocking | 2 Sendrecv (1d) / up to 8 (2d) | implicit in Sendrecv | none | baseline; pairing must be antipodal to avoid deadlock |
| nonblocking | 2 Irecv + 2 Isend (1d); 8-16 (2d) | Waitall | none | Irecvs posted before Isends, so buffering is guaranteed regardless of eager/rndv thresholds |
| overlap | same posts as nonblocking | Waitall for border rows | interior rows computed between post and Waitall | border rows recomputed after Waitall; `update_interior`/`update_borders` split |
| rma | `MPI_Win_fence` + `MPI_Put`s | one fence pair per step (complete/access epochs) | none in the correctness-first design | both grid buffers are exposed because the loop swaps current/next each step; the live window is `win_cells[k & 1]` (identical loop on all ranks makes the parity deterministic); target displacements from neighbour block shape; 2d ghost columns staged through one contiguous auxiliary window |
| persistent | `MPI_Startall` on pre-registered `MPI_Send_init`/`MPI_Recv_init` requests | Waitall | none | requests are bound to stable buffer addresses once: one request set per grid buffer (the loop swaps current/next), selected by the same `step & 1` parity as the RMA windows; per-step cost is two library calls instead of argument re-listing |
| sendrecv2 | explicit `MPI_Send` + `MPI_Recv` pairs, one ring shift at a time | synchronous | none | odd/even parity stagger: even ranks send first, odd ranks receive first, so the wait chain along any message cycle terminates at an opposite-parity rank (acyclic for odd cycle lengths too) |

Deadlock-freedom: blocking Sendrecv pairs each send with a receive from the
*opposite* neighbour (each direction is an independent ring); nonblocking posts
all receives first; sendrecv2 staggers send-first/receive-first by rank parity
along each ring cycle; RMA keeps all local writes to owned interior bytes between
the fences, so closing-fence visibility is well defined. A second lesson from
the matrix: the first RMA version exposed only the initial `current` buffer,
but the loop swaps current/next every step, so from step 1 the neighbours'
`MPI_Put`s landed in a retired buffer and every multi-rank RMA run diverged
while `P = 1` still passed. Both buffers are now exposed and the live window
is selected as `win_cells[step & 1]` — deterministic because all ranks run the
identical loop. "It runs with `P = 1`" remains not evidence.

**2D decomposition.** A periodic Cartesian process grid
(`MPI_Cart_create`, `reorder=0`, most balanced factor pair with `a <= H`,
`b <= W`, chosen rank-locally so no broadcast is needed). Each rank owns a
self-contained `LifeGrid` with ghost rings on all four sides; edge ghosts use
two Sendrecv pairs (or column wrap when `cp == 1`) and corners four diagonal
ring shifts. Tags are named after the ghost segment they fill on the receiving
side, which disambiguates the role aliasing of small grids (in 2x2, NW == SE
and N == S). Scatterv/Gatherv use a block-major layout because a 2d row block
is not contiguous in the global grid; 1d degenerates to the interface 1.0
row-major layout.

**Status of evidence.** Atlas (`foss/2025b`, OpenMPI 5), zero-warning build:
the interface 1.0 regression reports **PASS=773, FAIL=0, BLOCKED=0** and the
full `decomp x comm x ranks` matrix (`scripts/test_comm.sh`: 64x64, 97x53 and
5x13 — the last forces single-column blocks in a 2x4 process grid, covering
A's validation-suite cases D20/D21 — with ranks 1/2/3/4/8, random and
`--input` initialisation) reports **PASS=124,
FAIL=0** over the four original backends, all checksums matching A's serial
kernel. Re-run 2026-11-09 with the six-backend matrix (`persistent`,
`sendrecv2` included): **PASS=186, FAIL=0**, interface 1.0 baseline still
773/0/0. For the report narrative
the test suite is A's 90-test validation suite (`docs/VALIDATION_SUITE.md`);
773, 124 and 186 are execution counts behind it. Note on terminology: the blocking backend uses `MPI_Sendrecv`, a single call
that issues the send and the receive required by the ring pairing. Because the
assignment asks for blocking send/recv and the nonblocking variants explicitly,
the backend list additionally provides `sendrecv2` (separate `MPI_Send`/
`MPI_Recv` calls with a parity stagger) and `persistent` (`MPI_Send_init`/
`MPI_Recv_init` + `MPI_Startall`), completing the transport menu: two blocking
forms, two nonblocking forms (one with overlap), one RMA form and one
persistent form.

**Performance evidence (Vanda `batch_cpu`, job IDs below, every checksum equal
to the serial reference).** The 256-core target was first reached on 2026-11-05
(4096² pilot), then re-measured at full scale on 2026-11-07 under the
`foss/2024a` toolchain the group experiment plan mandates (GCC 13.3 +
OpenMPI 5.0.3), with the program and the serial baseline built inside the job:

| Job | Nodes x ranks | Content |
|---|---|---|
| 1527052 `gol_b_pre` | 2 x 64 | preflight: in-job build, OMPI-5 cross-node launch, 56-test matrix |
| 1527951 / 52 / 55 `gol_e2*` | 1 x 64 / 2 x 64 / 4 x 64 | weak scaling, 9 points, 1–256 ranks |
| 1527953 / 54 / 56 `gol_e3*` | 1 x 64 / 2 x 64 / 4 x 64 | strong scaling + backend matrix, 16384² x 200 steps |
| 1535552 / 53 `gol_e4p*` | 1 x 64 / 4 x 64 | backfill for `persistent` and `sendrecv2`, 16/64/256 ranks x 1d/2d x 3 repeats; in-job serial baseline 132.12 s |

Results, figures and the plotting script are archived in `gol_B/results/`,
`gol_B/report/figures/` and `gol_B/scripts/plot_results.py`.

Headline numbers (median of 3 repeats, kernel-second scope per interface 1.0):

- **Strong scaling**, 16384² x 200 steps, serial baseline 131.18 s: 16 ranks
  15.8x (1D) / 15.9x (2D), 64 ranks 53.0x / 53.8x, **256 ranks 146.3x
  (1D, 57.1%) vs 197.8x (2D, 77.3%)**.
- **Weak scaling**, 1024² cells per rank x 500 steps: 1.140 s at 1 rank,
  1.471 s at 64 (**77.5%**), 2.048 s at 256 (**55.7%**, 1D) / 2.071 s
  (55.0%, 2D). The 64→128 step costs ~0.55 s: that is the cross-node
  interconnect, not a decomposition artefact.
- **Backend comparison**, 16384² x 200 steps at 256 ranks (median s):
  1D blocking 0.897 / nonblocking 0.908 / overlap 0.906 / **RMA 0.959**;
  2D blocking 0.663 / nonblocking 0.641 / **overlap 0.637** / RMA 0.748.
  At 16 and 64 ranks all four backends sit within 2%: the kernel is
  compute-bound there, so there is nothing for a backend to change.
- **2D beats 1D at 256 ranks** because one step's halo drops from two rows of
  16384 bytes to two rows plus two columns of 1024 bytes plus four corners.
  The saving only appears once the ranks outnumber the cores of a single node.
- **The two later backends cost nothing** (job 1535552/53, same protocol,
  median s): 1D `persistent` 8.357 / 2.503 / 1.009 and `sendrecv2` 8.356 /
  2.474 / 1.044 at 16 / 64 / 256 ranks; 2D `persistent` 8.340 / 2.457 /
  0.935 and `sendrecv2` 8.347 / 2.431 / 0.992. All 39 rows carry the same
  `checksum 66c45e3153703ed2` as the serial reference. At 16 and 64 ranks
  they fall inside the 2% band of the original four, so pre-registering the
  requests (`persistent`) and splitting `Sendrecv` into explicit
  `Send`/`Recv` (`sendrecv2`) are both free in practice.

**Cross-job caution for 256 ranks.** Absolute 256-rank times vary between
jobs by much more than the backend effect: the same 1D blocking configuration
measured 0.897 s in E3, 1.149 s in C's recheck (job 1532914) and 0.886 s in
C's E1, and 2D ranges 0.663–1.128 s. Single-node 64-rank results agree within
4% everywhere. All backend and decomposition comparisons above are therefore
made **inside one job**, never across jobs; the 26% figure belongs to E3's
own 256-rank allocation and is reported as such.

The 4096² pilot point at 256 ranks had a 40–110 ms kernel, i.e. below node
noise, so it only proved "runs at 256 ranks"; every figure below uses the
16384² x 200-step measurement instead. The free-queue twin
(1519498, `cpu_parallel`) was cancelled once the billed batches finished: a
128-core request needs ~80% of that queue's 160-core budget idle.

## 6.1 GPU variant (OpenMP target offload)

Lecture 12 requires an actual GPU test when X=OpenMP. The GPU variant is the
same source compiled with `-DGOL_USE_OMP_TARGET` (`make gpu`); it reports
`"variant":"gpu"` and `"threads":1`, so a GPU row can never be mistaken for a
host row in the CSV.

Data placement: both grids are mapped once (`target enter data`) and stay
device-resident for the whole run, so one step moves four rows across PCIe --
the two rows MPI sends and the two received ghost rows -- instead of the whole
grid. The kernel folds the periodic column wrap in
(`cl = c==1 ? cols : c-1`), which is exactly `life_wrap_horizontal`'s rule, so
the host copy is allowed to stay stale between steps. The fetch/flush hooks sit
inside `halo_post`/`halo_finish`, which means every communication backend
and the overlap schedule are reused unchanged. `--decomp 2d` is rejected on
purpose: the 2D halo needs every row's first and last column on the host, which
contradicts that design, and a silent wrong answer is worse than an error.

Evidence status: the CPU-side logic path builds zero-warning on Atlas with
`-foffload=disable` (Atlas's own nvptx offload is unusable because the
`/app1/ebapps` compiler directories are not readable by regular accounts), and
the full correctness matrix passes on that path with
`GOL_ALLOW_HOST_TARGET=1`: **PASS=56, FAIL=0** over 5 grids (97x53, 64x64,
5x13, 3x3, 1x1) x ranks 1/2/4/8 x 4 communication backends, every checksum
equal to A's serial kernel. That validates the kernel, the folded wrap and the
host/MPI halo plumbing, **not** device transfer or GPU speed:
`GOL_ALLOW_HOST_TARGET=1` is a logic-only check. On Vanda the variant compiles
with `nvhpc-hpcx-cuda12/26.1` (`-mp=gpu`) and the fatbin carries sm_75 through
sm_121, so the A40 nodes (sm_86) are covered. Device results come from job
1524542 on `batch_gpu` (36 cores + 1 GPU): the checksum matrix passes
**32/32** across five grids, ranks 1/4 and the four backends that existed when
the GPU variant was written (`persistent` and `sendrecv2` were added later and
have not been re-run through `GOL_USE_OMP_TARGET`; they share the same
`halo_post`/`halo_finish` hooks, and the host-side matrix covers all six), and
the same job
then compares **1 GPU against 36 CPU threads of the same node under the same
NVHPC toolchain** (300 steps, 3 repeats, median).

| grid | GPU (A40) | 36 CPU threads | GPU relative |
|---|---|---|---|
| 2048² | 0.2505 s | 0.1245 s | 2.01x slower |
| 4096² | 0.4919 s | 0.4653 s | 1.06x slower |

The GPU does **not** beat the multi-core CPU here, and that is the honest
result rather than a defect to hide. Every step must hand four rows back and
forth across PCIe and launch one kernel, because the 1D halo is exchanged by
host MPI; at 2048² those fixed costs dominate, and they amortise as the grid
grows (2.01x -> 1.06x at 4096²), while the kernel itself is limited by uint8
memory traffic rather than floating-point throughput. Beating the CPU would
require keeping the halo on the device (CUDA-aware MPI or NVSHMEM), which is
outside this assignment's scope. Two implementation facts are worth recording
because they cost real debugging: NVHPC rejects a target region whose only
statement is a declare-target call (NVC++-S-0155), so the loop is inlined; and
the blocking backend completes inside `halo_post`, so the host-to-device ghost
push has to happen there too -- otherwise the first GPU run passed nonblocking,
overlap and rma while failing every blocking case.

A free-pool duplicate (1524530) was queued alongside the paid job; it never
started (`Insufficient amount of resource: node_pool`) and was cancelled on
2026-11-06 while Vanda was closed for urgent maintenance, since the paid job had
already delivered the full data set.

## 7. Correctness validation

The interface ships with A's verification harness (`tools/verify.py`). It:

- evolves every case with an independent Python oracle,
- runs the binary through the external command array in `config/variants.json`,
- compares complete P1 PBM grids cell by cell,
- validates every JSON field, including `live_cells` and `checksum_fnv1a64`,
- runs the external variants over ranks 1/2/4/8 and hybrid threads 1/2/4.

The B-side commands are `make test`, which builds `build/mpi` and `build/hybrid`,
installs them into A's `build/`, builds the serial reference and runs
`python3 tools/verify.py --require-all`; and `make test-comm` (or
`bash scripts/test_comm.sh`), which validates the new `--decomp`/`--comm`
paths: for each grid, A's serial checksum is the ground truth and every
`DECOMPS x COMMS x RANKS` combination must reproduce it, echo the requested
`decomp`/`comm` in its JSON, and carry all interface 1.0 fields, including one
`--input` run per backend at four ranks.

**Atlas result with `foss/2025b`:** **PASS=773, FAIL=0, BLOCKED=0** — 610
serial cases plus 163 external MPI/hybrid runs. All `live_cells` and
`checksum_fnv1a64` fields matched the independent oracle. The reported suite
is A's 90-test validation suite (`docs/VALIDATION_SUITE.md`); 773 is the
underlying execution count, dominated by the exhaustive 3x3 sweep (512 of the
610 serial runs). Cases from the
earlier private binary format are not reused because that format no longer
exists.

Cases that matter for this decomposition:

| Category | Cases exercised |
|---|---|
| Still lifes and oscillators | empty, single_dies, block_stable, blinker_odd/even, corner_block (7x9 and 8x9) |
| Periodic edges | edge_blinker (7x9), glider_four (12x13), glider_wrap (12x13) |
| Tiny domains | one_cell_alive (1x1), all_alive (5x7) |
| Partition stress | local_rows_one (4 rows, 4 ranks), nondivisible_rows (5 rows, 2 and 4 ranks) |
| Zero steps | zero_steps (5x7): distribution/gathering only |
| Rank counts | 1, 2, 4, 8; combinations with `ranks > rows` are skipped by the harness |
| Hybrid | 1/2/4 ranks x 1/2/4 threads for each named case |

The zero-step case isolates `Scatterv`/`Gatherv` from the stencil: a mismatch
there cannot be blamed on the update or on the halo exchange.

## 8. Performance and communication cost

Timing definitions from interface 1.0:

- `kernel_seconds`: every iteration, including horizontal wrap, halo exchange,
  update, synchronisation and swapping; initialisation, gathering, diagnostics
  and file output are excluded.
- `total_seconds`: allocation, input, iterations, final gather and diagnostics;
  PBM/JSON writing is excluded.
- Both are reduced with `MPI_MAX` after a synchronised start. Only
  `kernel_seconds` is used for speedup; the two scopes are never mixed.
- **Rounding convention.** Speedup and efficiency are computed from the
  unrounded three-run medians (`kernel_seconds` carries nine decimals in the
  CSVs, i.e. `MPI_Wtime` resolution). Times are then printed to three
  decimals. Recomputing a speedup from the printed times can therefore differ
  by up to ~0.1%: for the 1D blocking run at 256 ranks,
  131.182403 / 0.896566 = 146.31 with full precision, but
  131.182 / 0.897 = 146.25 from the printed values. The difference is display
  rounding, not measurement error; the timer itself is far more precise than
  the table.

Per step a rank sends and receives `2*(W+2)` bytes in two messages. Compute per
rank is `local_rows*W` cells with roughly nine neighbour reads each. Ignoring
latency, the communication-to-compute ratio is about `2/(9*local_rows)`; it
grows as ranks increase and `local_rows` shrinks. This is the expected
strong-scaling limit of a 1D decomposition, and efficiency should fall faster on
1024x1024 than on 2048x2048. For the 2D decomposition the halo volume drops to
about `2*(local_rows + local_cols) + 4` bytes per rank per step, but in up to
eight message segments, so the comparison is latency-sensitive.

To make the communication share measurable, an optional instrumentation
(`GOL_HALO_STATS=1`) accumulates the time spent inside halo exchanges into a
separate `halo_seconds` JSON field. It is debug instrumentation: it is never
enabled in official timing batches, and on the Vanda cluster it currently
wedges across nodes (see limitations), so halo shares are taken from Atlas runs
only. The dedicated Atlas batch (`scripts/halo_stats_atlas.pbs`,
16384² x 200 steps, ranks 16/32/64 on one node — Atlas caps a job at 96
ncpus and cross-node mpirun fails for this account) measures `halo_seconds`
for all six backends and writes `results/halo_atlas.csv` (job 1038437,
36 runs, every checksum identical to the serial reference, including both
new backends). Shares below are `halo_seconds / kernel_seconds` from that CSV:

| ranks | 1D blocking | 1D nonblocking | 1D persistent | 1D overlap / rma / sendrecv2 | 2D all six |
|---|---|---|---|---|---|
| 16 | 36.8% | 35.5% | 38.8% | 0.6–0.8% | 0.8–2.1% |
| 32 | 42.0% | 36.8% | 36.4% | 1.3–3.0% | 2.7–3.4% |
| 64 | 45.6% | 40.0% | 40.0% | 5.7–8.9% | 6.5–8.2% |

Three observations. First, decomposition shape dominates: the 1D ring
serialises into a P-long wait chain per step, while the 2D grid splits it into
sqrt(P)-short chains, so at 64 ranks the same message volume costs 45.6% of
kernel time (1D blocking) vs 7.0% (2D blocking). Second, ordering decides how
much of that wait lands in the halo clock: `overlap` hides it behind interior
compute and `sendrecv2`'s parity stagger almost eliminates it (0.6–7.0%),
whereas the three synchronous-in-spirit backends (blocking, nonblocking,
persistent) pay the full chain wait; `persistent` tracks `nonblocking`
within 3.3 percentage points at every rank count, confirming the
pre-registered path costs nothing over ordinary Isend/Irecv. Third, these
shares are a **single-node shared-memory phenomenon**: on Vanda (4 nodes,
`--bind-to core`, different fabric) all six backends differ by ≤2% at 16 and
64 ranks (section 6), so the official scaling
numbers above are unaffected — the halo-cost table is reported for the Atlas
environment as measured, alongside its job ID and raw CSV.

**Official numbers (Vanda `batch_cpu`, 2026-11-07, `foss/2024a`).** Raw
CSVs in `gol_B/results/` (`vanda_e2_weak_256.csv`,
`vanda_e3_backends_256.csv`, `e4_newcomm_p1.csv`, `e4_newcomm_p2.csv`,
`halo_atlas.csv`), figures in `gol_B/report/figures/` (log axes
as the lecture requires), generated by `scripts/plot_results.py`:

| Study | Configuration | Result (median of 3, min–max in figure) |
|---|---|---|
| Serial scaling (fig. 4) | single core, 512²–8192², 100 steps, 3 repeats | 0.111 / 0.442 / 1.273 / 5.088 / 20.317 s; log-log slope **0.927** vs cell count; throughput 2.4–3.3 M cells/s |
| Strong scaling (fig. 1) | 16384² x 200 steps, ranks 16/64/256, serial baseline 131.18 s | 1D: 15.8x / 53.0x / **146.3x**; 2D: 15.9x / 53.8x / **197.8x** |
| Weak scaling (fig. 2) | 1024² cells/rank x 500 steps, ranks 1–256 | 1.140 → 1.471 s @64 (**77.5%**) → 2.048 s @256 (**55.7%**, 1D) / 2.071 s (55.0%, 2D) |
| Backend comparison (fig. 3) | 16384² x 200 steps, 16/64/256 ranks | ≤2% spread at 16/64; @256 RMA +7% (1D) / +13% (2D); overlap −4% (2D @256) |
| New backends (job 1535552/53) | 16384² x 200 steps, 16/64/256 ranks | `persistent` and `sendrecv2` inside the 2% band at 16/64; 2D faster than 1D by 5–7% at 256 ranks in the same job; 39/39 checksums equal to serial |

Cold-start handling: figures plot the per-group median and the min–max span;
the median is also what `scripts/summary.py` reports. Every grid/rank/decomp
group has three repeats whose checksums are identical, and the checksum equals
the serial run of the same grid and step count — the timing gate rejects any
row that disagrees, so no run in these CSVs can be mislabelled.

Communication-share instrumentation (`GOL_HALO_STATS=1`) stays off on Vanda
(cross-node wedge, see limitations); halo shares come from Atlas runs only.

## 9. Limitations

- `P <= H` is required; thin domains cannot use many ranks.
- Rank 0 holds the full grid for input and output, so `Scatterv`/`Gatherv`
  memory and time are `O(HW)` on that rank.
- Blocking communication serialises halo exchange and computation; measured
  2026-11-07: `overlap` recovers 4% in 2D at 256 ranks and nothing measurable
  elsewhere, while `rma` is the slowest backend at 256 ranks (+7% 1D, +13% 2D;
  the per-step fence pair is a global epoch). At 16 and 64 ranks all backends
  are within 2%. The two later backends (`persistent`, `sendrecv2`) add no
  measurable cost in the same band (job 1535552/53, 16/64/256 ranks), but
  neither was re-run through the GPU offload path.
- Weak scaling measured 2026-11-07 on the full range: 1024² cells/rank x 500
  steps, 1–256 ranks, 77.5% efficiency at 64 and 55.7% at 256. The drop
  between 64 and 128 is the transition from one node to two nodes.
- The 1D decomposition keeps two messages per rank and step; the 2D
  decomposition trades them for up to eight smaller segments plus a
  pack/unpack cost for column ghosts. Measured outcome: within 1% at
  16/64 ranks, then 2D 26% faster at 256 ranks in E3's own allocation
  (0.663 vs 0.897 s). This margin is **not** reproduced in every job: the same
  comparison is only 5–7% in job 1535552/53, and C measured 1D/2D differences
  of a few percent at 256 ranks in other allocations. The 2D advantage is
  real but its size is allocation-dependent; see the cross-job caution in
  section 6.
- **Open cluster issue (investigation paused).** On Vanda (OpenMPI 4.1.8,
  multi-node), running with `GOL_HALO_STATS=1` wedges at the post-kernel halo
  aggregation: the root rank stalls between the two reductions while all
  compute results are already correct. Reproduced 8/8 with tracing across two
  aggregation designs — `MPI_Reduce` and an explicit Send/Recv collection —
  and bare collectives (Barrier/Allreduce/Reduce/Bcast) pass in isolation, so
  the trigger is the surrounding code path, not the collective itself. After
  three failed fix attempts the investigation was paused per the escalation
  rule: the code was reverted to the last-validated `MPI_Reduce` form, and
  Vanda batches run with the instrumentation off. Atlas is unaffected (all
  regressions passed with the instrumentation on), and `halo_seconds` data
  comes from Atlas runs only.

## 10. AI usage (MPI section)

- The assistant helped draft the MPI driver, the halo-exchange pairing, the
  communication backends, the test and benchmark scripts, and this section.
- Every reported run was compiled and executed on Atlas or Vanda by the
  member responsible for this section; every code change was verified before
  it was used for a measurement.
- All performance figures come from the archived CSV batches in `results/`,
  and every grid is reproducible from the commands in `README.md`.
- The text was drafted with AI assistance and reviewed manually for accuracy.
