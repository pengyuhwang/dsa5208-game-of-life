# A/B Exchange-Interface Specification and 2D Integration Notes

The frozen A-side contract stays
`../Game_of_Life_A/docs/INTERFACE.md`; this file only adds to it. Anything here
that contradicts A's contract is a bug in this file.

Status: implemented in `gol_B/src/mpi.c`. Atlas (`foss/2025b`, OpenMPI 5) with
zero-warning builds: interface 1.0 regression PASS=773/FAIL=0/BLOCKED=0, and
the full `decomp x comm x ranks` matrix in `scripts/test_comm.sh`
(64x64/97x53/5x13 — the last forces single-column blocks in a 2x4 process
grid, matching A's validation-suite cases D20/D21 — ranks 1/2/3/4/8, random
and `--input` init) reports PASS=124/FAIL=0, independently reproduced by A on
Vanda (PASS=88/FAIL=0 for the pre-5x13 matrix). Baseline behaviour (no new
flags) is byte-identical to the previously validated 1D blocking driver.

## 1. What changed for A and C

Nothing in the frozen interface 1.0 contract changes. The extensions are
strictly additive:

- **CLI**: two new optional flags, `--decomp 1d|2d` (default `1d`) and
  `--comm blocking|nonblocking|overlap|rma|persistent|sendrecv2` (default
  `blocking`). Defaults reproduce the old driver exactly; unknown values are
  rejected on every rank.
- **JSON record**: two new fields `"decomp"` and `"comm"` inserted after
  `"variant"`. Every interface 1.0 field keeps its name, type and meaning.
  `tools/verify.py` validates required fields only, so old consumers are
  unaffected.
- **CSV schema**: two trailing columns `decomp,comm` appended by
  `scripts/bench_atlas.sh` (`na,na` for serial rows). `scripts/summary.py`
  groups by them when present and falls back to `1d/blocking` for older CSVs.
  Serial reference, kernel-seconds timing scope and checksum semantics are
  untouched.

## 2. Decompositions

Both decompositions reuse the frozen contiguous-block formula
`len(i) = T/P + (i < T%P)`, `off(i) = i*(T/P) + min(i, T%P)`.

### 2.1 `--decomp 1d` (baseline)

`P` row blocks on a periodic ring, `local_cols = W`. Unchanged from interface
1.0, including tags 11/12 and the full-`stride` row halo that carries corners.

### 2.2 `--decomp 2d`

- Process grid: `MPI_Cart_create(..., dims, periods={1,1}, reorder=0)`.
  `dims` is chosen by a deterministic, rank-local rule — the most balanced
  factor pair `a*b = P` with `a <= H` and `b <= W` (rows first). No
  broadcast needed; if no pair exists the run aborts with a clear error.
- `rp*dims[0]` rows and `cp*dims[1]` columns are split with `block_split`,
  giving each rank `(local_rows, local_cols, row_off, col_off)`.
- **Local layout**: every rank owns a self-contained `LifeGrid` with
  `rows = local_rows`, `cols = local_cols`, `stride = local_cols + 2`; all
  four sides have ghost rings. This reuses A's `life_update_row` unmodified.
- Horizontal periodicity: `cp == 1` uses `life_wrap_horizontal`; otherwise
  owned columns are packed (`pack_full`-style, one contiguous run per owned
  row) and exchanged with two Sendrecv pairs (tags `TAG_COL_E/W`).
- Corners: `rp == 1 && cp == 1` uses memcpy; otherwise four diagonal
  Sendrecv ring shifts (tags `TAG_TL/TR/BL/BR`). Tags are named after the
  ghost segment they fill on the **receiving** side, which keeps matching
  unambiguous when one rank holds several neighbour roles — e.g. in a 2x2
  process grid NW == SE and N == S.

## 3. Halo exchange interface (`halo_post` / `halo_finish`)

Six backends selected by `--comm`. Every backend fills exactly
the same ghost bytes before the update reads them, so the mathematical
results are backend-independent.

| mode       | calls at post                          | completion                          | overlap |
|------------|----------------------------------------|-------------------------------------|---------|
| blocking   | Sendrecv pairs (1d: 2, 2d: 8 or wraps)  | synchronous                         | none    |
| nonblocking| Irecv before Isend, then Waitall        | `halo_finish` = Waitall             | none    |
| overlap    | same posts as nonblocking               | finish = Waitall (borders)          | interior rows computed between post and finish |
| rma        | `MPI_Win_fence(0)` then `MPI_Put`       | finish = `MPI_Win_fence(0)` + column unpack | none |
| persistent | `MPI_Startall` on pre-registered Send_init/Recv_init requests | finish = Waitall | none |
| sendrecv2  | explicit `MPI_Send`/`MPI_Recv` pairs, odd/even rank stagger | synchronous | none |

### 3.1 Deadlock-freedom

- blocking: every Sendrecv pairs its send with a receive from the *opposite*
  neighbour (1d: up/down ring; 2d: each of rows/columns/diagonals is an
  independent ring), so a call never blocks on the rank it is sending to.
- nonblocking/overlap: all Irecvs are posted before any Isend, guaranteeing
  receive-side buffering regardless of eager/rndv thresholds.
- persistent: same request structure as nonblocking, started with
  `MPI_Startall`; requests are pre-bound per grid buffer (the active set
  follows the same `step & 1` parity as the RMA windows) so each step's
  buffers are correct without re-registration.
- sendrecv2: each ring-shift cycle is staggered by rank parity — even ranks
  send first, odd ranks receive first. Along any cycle the chain of "send
  waits for the next rank's posted receive" terminates at the first
  opposite-parity rank, so the wait graph is acyclic for every cycle length
  (odd cycle lengths included, via the chain terminating at an odd rank).
- rma: one fence pair per step over the live window. Both grid buffers are
  exposed (`MPI_Win_create` on each) because the main loop swaps current/next
  every step; the window for step k is `win_cells[k & 1]`, and since every
  rank executes the identical loop the parity is deterministic on all ranks.
  Puts are issued after the opening fence and are made visible to local loads
  by the closing fence (MPI complete/access epochs). Between the fences the
  owner only writes its own interior (`next` buffer, the *other* window),
  never ghost bytes, so there is no local/remote write conflict. Target
  displacements are computed from the neighbour's block shape (`block_at`),
  and 2D ghost columns are staged through a single contiguous auxiliary
  window because column segments are strided in memory.

### 3.2 Update partitioning (overlap only)

`update_interior` recomputes rows `2..local_rows-1` (rows whose 3x3
neighbourhood is complete without the halo), then `halo_finish`, then
`update_borders` for rows `1` and `local_rows`. Columns never split across
ranks in 1d; in 2d the border rows still need the (already local) column
ghosts, which `blocking`/`nonblocking` finish before `update_all` runs.

## 4. Scatter/Gather layout (A/B exchange format)

- Rank 0 packs the full grid **block-major**: for `p = 0..P-1` in global rank
  order, the `lp*lc` interior cells of rank p's block are stored contiguously
  (`pack_full`). `MPI_Scatterv`/`MPI_Gatherv` displacements are the running
  sums of `counts[p] = lp*lc`. This is required for 2d because a row block
  is not contiguous in the global row-major grid; for 1d it degenerates to
  the plain row-major layout of interface 1.0.
- Each rank `unpack_rows`/`pack_rows` its local block (row-major over local
  interior rows).
- Random initialisation is rank-local and reproducible:
  `life_initial_cell(seed, (row_off + r) * W + col_off + c, density)` — the
  same seed yields the same global grid for every `decomp/comm/ranks`.
- `--input` uses `pack_full` + Scatterv, so PBM parsing stays a rank-0
  serial operation and A's `life_read_pbm` is used unchanged.

## 5. Integration notes and constraints

- A's `life.h`/`life.c` are consumed as-is; the drivers add no kernel code.
  Any change A makes to `LifeGrid` layout or `life_initial_cell` semantics
  automatically flows into all modes.
- Guard rails: `1 <= P <= H` for 1d; a valid factor pair within `(H, W)` for
  2d; `--seed/--density-ppm` still rejected with `--input`. Unsupported
  combinations fail fast with a message on stderr and a nonzero exit on all
  ranks — no silent fallback.
- Verification entry points:
  - `make test` — A's oracle harness (interface 1.0 paths, unchanged).
  - `bash scripts/test_comm.sh` — the `decomp x comm x ranks` matrix against
    A's serial kernel, including `--input` and JSON field/echo checks.
- Verified on Atlas: the full matrix above plus the 773-case interface 1.0
  regression. Backend timing is now available: the 2026-11-07 Vanda batch
  (jobs 1527953/54/56, 16384² x 200 steps, 16/64/256 ranks) puts all four
  backends within 2% up to 64 ranks and RMA 7-13% behind at 256 ranks
  (`report/figures/backends.png`).
