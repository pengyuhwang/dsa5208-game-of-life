#define _POSIX_C_SOURCE 200809L
#include "life.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GOL_USE_OPENMP
#include <omp.h>
#ifdef GOL_USE_OMP_TARGET
#define VARIANT_NAME "gpu"
#else
#define VARIANT_NAME "hybrid"
#endif
#else
#define VARIANT_NAME "mpi"
#endif

enum {
    OPT_ROWS    = 1u << 0,
    OPT_COLS    = 1u << 1,
    OPT_STEPS   = 1u << 2,
    OPT_SEED    = 1u << 3,
    OPT_DENSITY = 1u << 4,
    OPT_INPUT   = 1u << 5,
    OPT_OUTPUT  = 1u << 6,
    OPT_DECOMP  = 1u << 7,
    OPT_COMM    = 1u << 8
};

/* 1D ring tags (unchanged baseline). 2D messages are tagged by the ghost
 * segment they fill on the receiving side, which keeps matching unambiguous
 * even when one rank holds several neighbour roles (e.g. 2x2 process grids,
 * where NW == SE and N == S). */
#define TAG_UP       11  /* 1D: first owned row, going up */
#define TAG_DOWN     12  /* 1D: last owned row, going down */
#define TAG_ROW_UP   21  /* 2D: first owned row, going north */
#define TAG_ROW_DOWN 22  /* 2D: last owned row, going south */
#define TAG_COL_E    23  /* 2D: east owned column, going east */
#define TAG_COL_W    24  /* 2D: west owned column, going west */
#define TAG_TL       31  /* 2D: fills my top-left ghost corner, from NW */
#define TAG_TR       32  /* 2D: fills my top-right ghost corner, from NE */
#define TAG_BL       33  /* 2D: fills my bottom-left ghost corner, from SW */
#define TAG_BR       34  /* 2D: fills my bottom-right ghost corner, from SE */

typedef enum { DECOMP_1D, DECOMP_2D } Decomp;
typedef enum {
    COMM_BLOCKING, COMM_NONBLOCKING, COMM_OVERLAP, COMM_RMA,
    COMM_PERSISTENT, COMM_SENDRECV2
} CommMode;

static const char *decomp_name(Decomp d) { return d == DECOMP_1D ? "1d" : "2d"; }
static const char *comm_name(CommMode m) {
    switch (m) {
    case COMM_BLOCKING:    return "blocking";
    case COMM_NONBLOCKING: return "nonblocking";
    case COMM_OVERLAP:     return "overlap";
    case COMM_PERSISTENT:  return "persistent";
    case COMM_SENDRECV2:   return "sendrecv2";
    default:               return "rma";
    }
}

typedef struct {
    uint64_t rows, cols, steps, seed, density;
    const char *input, *output;
    Decomp decomp;
    CommMode comm;
    unsigned seen;
} Options;

static void usage(FILE *out) {
    fprintf(out,
        "Usage: mpi|hybrid [--rows H --cols W | --input FILE.pbm] [options]\n"
        "  --steps N          generations, including zero (default 100)\n"
        "  --seed N           unsigned 64-bit seed (default 1)\n"
        "  --density-ppm N    initial density in [0,1000000] (default 500000)\n"
        "  --output FILE.pbm  optional complete final grid (P1 PBM)\n"
        "  --decomp M         domain decomposition: 1d (default) or 2d\n"
        "  --comm M           halo exchange: blocking (default), nonblocking,\n"
        "                     overlap (nonblocking + interior compute first),\n"
        "                     rma, persistent (Send_init/Recv_init),\n"
        "                     sendrecv2 (explicit Send/Recv, parity staggered)\n"
        "  --help             show this message\n"
        "Defaults: 64 rows, 64 columns; periodic B3/S23.\n"
        "stdout: one JSON timing/diagnostic record from rank 0.\n");
}

static int parse_u64(const char *s, uint64_t *value) {
    if (!s[0]) return -1;
    for (const char *p = s; *p; ++p)
        if (*p < '0' || *p > '9') return -1;
    errno = 0;
    char *end;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end) return -1;
    *value = (uint64_t)n;
    return 0;
}

/* Returns 0 for normal operation, 1 for --help, -1 for an invalid command line.
 * Every rank parses the same argv; no communication happens here. */
static int parse_args(int argc, char **argv, Options *o) {
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (!strcmp(key, "--help")) return 1;
        unsigned bit = 0;
        uint64_t *dest = NULL;
        if (!strcmp(key, "--rows"))              { bit = OPT_ROWS;    dest = &o->rows; }
        else if (!strcmp(key, "--cols"))         { bit = OPT_COLS;    dest = &o->cols; }
        else if (!strcmp(key, "--steps"))        { bit = OPT_STEPS;   dest = &o->steps; }
        else if (!strcmp(key, "--seed"))         { bit = OPT_SEED;    dest = &o->seed; }
        else if (!strcmp(key, "--density-ppm"))  { bit = OPT_DENSITY; dest = &o->density; }
        else if (!strcmp(key, "--input"))        bit = OPT_INPUT;
        else if (!strcmp(key, "--output"))       bit = OPT_OUTPUT;
        else if (!strcmp(key, "--decomp"))       bit = OPT_DECOMP;
        else if (!strcmp(key, "--comm"))         bit = OPT_COMM;
        else {
            fprintf(stderr, "unknown option: %s\n", key);
            return -1;
        }
        if ((o->seen & bit) || i + 1 == argc) {
            fprintf(stderr, "duplicate option or missing value: %s\n", key);
            return -1;
        }
        o->seen |= bit;
        const char *value = argv[++i];
        if (dest) {
            if (parse_u64(value, dest)) {
                fprintf(stderr, "invalid unsigned integer for %s\n", key);
                return -1;
            }
        } else if (bit == OPT_INPUT) {
            o->input = value;
        } else if (bit == OPT_OUTPUT) {
            o->output = value;
        } else if (bit == OPT_DECOMP) {
            if (!strcmp(value, "1d")) o->decomp = DECOMP_1D;
            else if (!strcmp(value, "2d")) o->decomp = DECOMP_2D;
            else { fprintf(stderr, "--decomp must be 1d or 2d\n"); return -1; }
        } else if (bit == OPT_COMM) {
            if (!strcmp(value, "blocking")) o->comm = COMM_BLOCKING;
            else if (!strcmp(value, "nonblocking")) o->comm = COMM_NONBLOCKING;
            else if (!strcmp(value, "overlap")) o->comm = COMM_OVERLAP;
            else if (!strcmp(value, "rma")) o->comm = COMM_RMA;
            else if (!strcmp(value, "persistent")) o->comm = COMM_PERSISTENT;
            else if (!strcmp(value, "sendrecv2")) o->comm = COMM_SENDRECV2;
            else {
                fprintf(stderr, "--comm must be blocking, nonblocking, overlap, "
                                "rma, persistent or sendrecv2\n");
                return -1;
            }
        }
    }
    if (!o->rows || !o->cols || o->density > 1000000) {
        fprintf(stderr, "invalid dimensions or density\n");
        return -1;
    }
    if (o->input && (o->seen & (OPT_SEED | OPT_DENSITY))) {
        fprintf(stderr, "--seed/--density-ppm cannot be combined with --input\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Topology: 1D ring of row blocks or 2D periodic Cartesian grid.       */
/* ------------------------------------------------------------------ */

typedef struct {
    Decomp decomp;
    MPI_Comm cart;              /* MPI_COMM_WORLD (1d) or Cartesian comm */
    int rank, size;
    int rp, cp;                 /* process-grid height/width; 1d: rp=size, cp=1 */
    int pr, pc;                 /* my process-grid coordinates */
    int nb_up, nb_down;         /* 1d ring neighbours */
    int nb_n, nb_s, nb_w, nb_e; /* 2d edge neighbours */
    int nb_nw, nb_ne, nb_sw, nb_se; /* 2d diagonal neighbours */
    unsigned long long H, W;    /* global dimensions */
    size_t local_rows, local_cols, row_off, col_off;
} Topo;

/* Contiguous block partition, identical to the interface 1.0 formula. */
static void block_split(unsigned long long total, int parts, int idx,
                        size_t *len, size_t *off) {
    const unsigned long long base = total / (unsigned long long)parts;
    const unsigned long long rem = total % (unsigned long long)parts;
    const unsigned long long u = (unsigned long long)idx;
    *len = (size_t)(base + (u < rem ? 1 : 0));
    *off = (size_t)(u * base + (u < rem ? u : rem));
}

/* Most balanced factor pair a*b=size with a<=H and b<=W; all ranks compute
 * the same result, so no broadcast is needed. Returns 0 on success. */
static int pick_dims(unsigned long long h, unsigned long long w, int size,
                     int dims[2]) {
    int dmax = 1;
    while ((long long)(dmax + 1) * (dmax + 1) <= (long long)size) ++dmax;
    for (int d = dmax; d >= 1; --d) {
        if (size % d) continue;
        const int a = d, b = size / d;
        if ((unsigned long long)a <= h && (unsigned long long)b <= w) {
            dims[0] = a; dims[1] = b; return 0;
        }
        if ((unsigned long long)b <= h && (unsigned long long)a <= w) {
            dims[0] = b; dims[1] = a; return 0;
        }
    }
    return -1;
}

/* Returns 0 on success, -1 if the ranks cannot decompose the grid. */
static int topo_init(Topo *t, Decomp decomp, int rank, int size,
                     unsigned long long h, unsigned long long w) {
    memset(t, 0, sizeof *t);
    t->decomp = decomp;
    t->rank = rank;
    t->size = size;
    t->H = h;
    t->W = w;
    if (decomp == DECOMP_1D) {
        if ((unsigned long long)size > h) return -1;
        t->cart = MPI_COMM_WORLD;
        t->rp = size; t->cp = 1;
        t->pr = rank; t->pc = 0;
        t->nb_up = (rank - 1 + size) % size;
        t->nb_down = (rank + 1) % size;
        block_split(h, size, rank, &t->local_rows, &t->row_off);
        t->local_cols = (size_t)w;
        t->col_off = 0;
        return 0;
    }
    int dims[2] = {0, 0};
    if (pick_dims(h, w, size, dims)) return -1;
    const int periods[2] = {1, 1};
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &t->cart) !=
            MPI_SUCCESS ||
        t->cart == MPI_COMM_NULL)
        return -1;
    int coords[2];
    MPI_Cart_coords(t->cart, rank, 2, coords);
    t->rp = dims[0]; t->cp = dims[1];
    t->pr = coords[0]; t->pc = coords[1];
    const int n[2] = {(coords[0] - 1 + dims[0]) % dims[0], coords[1]};
    const int s[2] = {(coords[0] + 1) % dims[0], coords[1]};
    const int w2[2] = {coords[0], (coords[1] - 1 + dims[1]) % dims[1]};
    const int e[2] = {coords[0], (coords[1] + 1) % dims[1]};
    const int nw[2] = {n[0], w2[1]};
    const int ne[2] = {n[0], e[1]};
    const int sw[2] = {s[0], w2[1]};
    const int se[2] = {s[0], e[1]};
    MPI_Cart_rank(t->cart, n, &t->nb_n);
    MPI_Cart_rank(t->cart, s, &t->nb_s);
    MPI_Cart_rank(t->cart, w2, &t->nb_w);
    MPI_Cart_rank(t->cart, e, &t->nb_e);
    MPI_Cart_rank(t->cart, nw, &t->nb_nw);
    MPI_Cart_rank(t->cart, ne, &t->nb_ne);
    MPI_Cart_rank(t->cart, sw, &t->nb_sw);
    MPI_Cart_rank(t->cart, se, &t->nb_se);
    block_split(h, dims[0], coords[0], &t->local_rows, &t->row_off);
    block_split(w, dims[1], coords[1], &t->local_cols, &t->col_off);
    return 0;
}

/* Block shape of an arbitrary grid position (RMA target displacements and
 * the block-major Scatterv/Gatherv layout). */
static void rank_shape(const Topo *t, int p, size_t *rows, size_t *row_off,
                       size_t *cols, size_t *col_off) {
    if (t->decomp == DECOMP_1D) {
        block_split(t->H, t->size, p, rows, row_off);
        *cols = (size_t)t->W;
        *col_off = 0;
        return;
    }
    int coords[2];
    MPI_Cart_coords(t->cart, p, 2, coords);
    block_split(t->H, t->rp, coords[0], rows, row_off);
    block_split(t->W, t->cp, coords[1], cols, col_off);
}

/* Block dimensions at process-grid coordinates (RMA target displacements). */
static void block_at(const Topo *t, int pr, int pc, size_t *rows,
                     size_t *cols) {
    size_t off;
    block_split(t->H, t->rp, pr, rows, &off);
    block_split(t->W, t->cp, pc, cols, &off);
}

/* ------------------------------------------------------------------ */
/* Halo exchange: one interface, four backends. Every rank owns a       */
/* self-contained LifeGrid (ghost ring on all four sides); each mode    */
/* fills exactly the same ghost bytes before the update reads them.     */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *send_w, *send_e;   /* 2d packed owned columns */
    uint8_t *recv_w, *recv_e;   /* 2d staged ghost columns (recv_w holds both) */
    MPI_Request req[20];
    int nreq;
    MPI_Win win_cells[2];       /* RMA windows over both grid buffers */
    MPI_Win win_aux;            /* RMA window over [recv_w][recv_e] */
    int cur;                    /* window parity of the current step */
    /* persistent backend: requests pre-bound to stable buffer addresses.
     * Grid buffers swap every step, so one request set exists per grid and
     * the active set follows the same step&1 parity as the RMA windows. */
    MPI_Request pgrid[2][12];   /* 1d ring (4) or 2d rows+corners (<=12) */
    int npg;                    /* requests per grid set */
    MPI_Request pcol[4];        /* 2d column staging (stable addresses) */
    int npcol;
} Halo;

/* persistent backend: pre-bind all requests once. Grid buffers swap every
 * step, so rows/corners get one request set per grid (selected by the same
 * step&1 parity as the RMA windows); 2d column staging buffers are
 * step-invariant and share one set. */
static void register_persistent(const Topo *t, const LifeGrid *g0,
                                const LifeGrid *g1, Halo *h) {
    if (t->decomp == DECOMP_1D) {
        h->npg = 4;
        for (int gi = 0; gi < 2; ++gi) {
            const LifeGrid *g = gi ? g1 : g0;
            MPI_Request *rq = h->pgrid[gi];
            const size_t rows = g->rows, stride = g->stride;
            MPI_Recv_init(g->cells + (rows + 1) * stride, (int)stride,
                          MPI_UINT8_T, t->nb_down, TAG_UP, t->cart, &rq[0]);
            MPI_Recv_init(g->cells, (int)stride, MPI_UINT8_T, t->nb_up,
                          TAG_DOWN, t->cart, &rq[1]);
            MPI_Send_init(g->cells + stride, (int)stride, MPI_UINT8_T,
                          t->nb_up, TAG_UP, t->cart, &rq[2]);
            MPI_Send_init(g->cells + rows * stride, (int)stride, MPI_UINT8_T,
                          t->nb_down, TAG_DOWN, t->cart, &rq[3]);
        }
        return;
    }
    if (t->cp > 1) {
        h->npcol = 4;
        MPI_Recv_init(h->recv_w, (int)g0->rows, MPI_UINT8_T, t->nb_w,
                      TAG_COL_E, t->cart, &h->pcol[0]);
        MPI_Recv_init(h->recv_e, (int)g0->rows, MPI_UINT8_T, t->nb_e,
                      TAG_COL_W, t->cart, &h->pcol[1]);
        MPI_Send_init(h->send_e, (int)g0->rows, MPI_UINT8_T, t->nb_e,
                      TAG_COL_E, t->cart, &h->pcol[2]);
        MPI_Send_init(h->send_w, (int)g0->rows, MPI_UINT8_T, t->nb_w,
                      TAG_COL_W, t->cart, &h->pcol[3]);
    }
    h->npg = 0;
    for (int gi = 0; gi < 2; ++gi) {
        const LifeGrid *g = gi ? g1 : g0;
        MPI_Request *rq = h->pgrid[gi];
        const size_t rows = g->rows, cols = g->cols, stride = g->stride;
        int k = 0;
        if (t->rp > 1) {
            MPI_Recv_init(g->cells + (rows + 1) * stride + 1, (int)cols,
                          MPI_UINT8_T, t->nb_s, TAG_ROW_UP, t->cart, &rq[k++]);
            MPI_Recv_init(g->cells + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                          TAG_ROW_DOWN, t->cart, &rq[k++]);
            MPI_Send_init(g->cells + stride + 1, (int)cols, MPI_UINT8_T,
                          t->nb_n, TAG_ROW_UP, t->cart, &rq[k++]);
            MPI_Send_init(g->cells + rows * stride + 1, (int)cols,
                          MPI_UINT8_T, t->nb_s, TAG_ROW_DOWN, t->cart,
                          &rq[k++]);
        }
        if (t->rp != 1 || t->cp != 1) {
            MPI_Recv_init(&g->cells[0], 1, MPI_UINT8_T, t->nb_nw, TAG_TL,
                          t->cart, &rq[k++]);
            MPI_Recv_init(&g->cells[cols + 1], 1, MPI_UINT8_T, t->nb_ne,
                          TAG_TR, t->cart, &rq[k++]);
            MPI_Recv_init(&g->cells[(rows + 1) * stride], 1, MPI_UINT8_T,
                          t->nb_sw, TAG_BL, t->cart, &rq[k++]);
            MPI_Recv_init(&g->cells[(rows + 1) * stride + cols + 1], 1,
                          MPI_UINT8_T, t->nb_se, TAG_BR, t->cart, &rq[k++]);
            MPI_Send_init(&g->cells[rows * stride + cols], 1, MPI_UINT8_T,
                          t->nb_se, TAG_TL, t->cart, &rq[k++]);
            MPI_Send_init(&g->cells[rows * stride + 1], 1, MPI_UINT8_T,
                          t->nb_sw, TAG_TR, t->cart, &rq[k++]);
            MPI_Send_init(&g->cells[stride + cols], 1, MPI_UINT8_T, t->nb_ne,
                          TAG_BL, t->cart, &rq[k++]);
            MPI_Send_init(&g->cells[stride + 1], 1, MPI_UINT8_T, t->nb_nw,
                          TAG_BR, t->cart, &rq[k++]);
        }
        h->npg = k;
    }
}

static void halo_init(const Topo *t, const LifeGrid *g0, const LifeGrid *g1,
                      Halo *h, CommMode mode) {
    memset(h, 0, sizeof *h);
    if (t->decomp == DECOMP_2D) {
        const size_t rows = g0->rows;
        h->send_w = malloc(rows);
        h->send_e = malloc(rows);
        h->recv_w = malloc(2 * rows);   /* one allocation for the RMA window */
        h->recv_e = h->recv_w + rows;
        if (!h->send_w || !h->send_e || !h->recv_w) {
            fprintf(stderr, "allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    if (mode == COMM_PERSISTENT && t->size > 1) {
        register_persistent(t, g0, g1, h);
        return;
    }
    if (mode != COMM_RMA || t->size == 1) return;
    /* RMA: expose BOTH grid buffers. The main loop swaps current/next every
     * step, so the live window at step k is win_cells[k & 1]; every rank runs
     * the identical loop, which makes the parity deterministic everywhere. */
    if (MPI_Win_create(g0->cells, (MPI_Aint)((g0->rows + 2) * g0->stride), 1,
                       MPI_INFO_NULL, t->cart, &h->win_cells[0]) != MPI_SUCCESS ||
        MPI_Win_create(g1->cells, (MPI_Aint)((g1->rows + 2) * g1->stride), 1,
                       MPI_INFO_NULL, t->cart, &h->win_cells[1]) != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    if (t->decomp == DECOMP_2D &&
        MPI_Win_create(h->recv_w, (MPI_Aint)(2 * g0->rows), 1, MPI_INFO_NULL,
                       t->cart, &h->win_aux) != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static void halo_destroy(Halo *h) {
    for (int gi = 0; gi < 2; ++gi)
        for (int i = 0; i < h->npg; ++i) MPI_Request_free(&h->pgrid[gi][i]);
    for (int i = 0; i < h->npcol; ++i) MPI_Request_free(&h->pcol[i]);
    if (h->win_cells[0]) MPI_Win_free(&h->win_cells[0]);
    if (h->win_cells[1]) MPI_Win_free(&h->win_cells[1]);
    if (h->win_aux) MPI_Win_free(&h->win_aux);
    free(h->send_w);
    free(h->send_e);
    free(h->recv_w);
}

static void pack_columns(const LifeGrid *g, uint8_t *w, uint8_t *e) {
    for (size_t r = 1; r <= g->rows; ++r) {
        w[r - 1] = g->cells[r * g->stride + 1];
        e[r - 1] = g->cells[r * g->stride + g->cols];
    }
}

static void unpack_columns(LifeGrid *g, const uint8_t *w, const uint8_t *e) {
    for (size_t r = 1; r <= g->rows; ++r) {
        g->cells[r * g->stride + 0] = w[r - 1];
        g->cells[r * g->stride + g->cols + 1] = e[r - 1];
    }
}

/* ---- shared self-neighbour pieces ---- */

/* 1D ring, P=1: full stride rows, ghost columns included. */
static void self_1d(LifeGrid *g) {
    const size_t rows = g->rows, stride = g->stride;
    memcpy(g->cells, g->cells + rows * stride, stride);
    memcpy(g->cells + (rows + 1) * stride, g->cells + stride, stride);
}

/* 2D, single row band: ghost rows carry only the local_cols interior
 * bytes; their two extra cells are the corner ghosts, exchanged (or
 * wrapped) separately, so no ordering dependency is introduced. */
static void self_rows_2d(LifeGrid *g) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    memcpy(g->cells + 1, g->cells + rows * stride + 1, cols);
    memcpy(g->cells + (rows + 1) * stride + 1, g->cells + stride + 1, cols);
}

/* ---- 1D backends (baseline semantics preserved exactly) ---- */

/* Blocking: two ring-shift Sendrecv calls. Each pairs its send with a
 * receive from the OPPOSITE neighbour, which is deadlock-free for all P. */
static void blocking_1d(const Topo *t, LifeGrid *g) {
    const size_t rows = g->rows, stride = g->stride;
    if (t->size == 1) { self_1d(g); return; }
    MPI_Sendrecv(g->cells + stride, (int)stride, MPI_UINT8_T, t->nb_up, TAG_UP,
                 g->cells + (rows + 1) * stride, (int)stride, MPI_UINT8_T,
                 t->nb_down, TAG_UP, t->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(g->cells + rows * stride, (int)stride, MPI_UINT8_T,
                 t->nb_down, TAG_DOWN, g->cells, (int)stride, MPI_UINT8_T,
                 t->nb_up, TAG_DOWN, t->cart, MPI_STATUS_IGNORE);
}

/* Nonblocking/overlap post: receives before sends, distinct tags per
 * direction; send buffers stay stable until halo_finish. */
static void post_1d(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, stride = g->stride;
    if (t->size == 1) { self_1d(g); return; }
    h->nreq = 0;
    MPI_Irecv(g->cells + (rows + 1) * stride, (int)stride, MPI_UINT8_T,
              t->nb_down, TAG_UP, t->cart, &h->req[h->nreq++]);
    MPI_Irecv(g->cells, (int)stride, MPI_UINT8_T, t->nb_up, TAG_DOWN,
              t->cart, &h->req[h->nreq++]);
    MPI_Isend(g->cells + stride, (int)stride, MPI_UINT8_T, t->nb_up, TAG_UP,
              t->cart, &h->req[h->nreq++]);
    MPI_Isend(g->cells + rows * stride, (int)stride, MPI_UINT8_T,
              t->nb_down, TAG_DOWN, t->cart, &h->req[h->nreq++]);
}

/* RMA: fence epoch; Put my first/last owned rows (full stride, corners
 * included) into the vertical ghosts of the ring neighbours. */
static void rma_post_1d(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, stride = g->stride;
    if (t->size == 1) { self_1d(g); return; }
    size_t ur, dr, off;
    const MPI_Win win = h->win_cells[h->cur];
    block_split(t->H, t->size, t->nb_up, &ur, &off);
    block_split(t->H, t->size, t->nb_down, &dr, &off);
    MPI_Win_fence(0, win);
    MPI_Put(g->cells + stride, (int)stride, MPI_UINT8_T, t->nb_up,
            (MPI_Aint)((ur + 1) * stride), (int)stride, MPI_UINT8_T, win);
    MPI_Put(g->cells + rows * stride, (int)stride, MPI_UINT8_T, t->nb_down,
            0, (int)stride, MPI_UINT8_T, win);
}

/* ---- 2D backends ---- */

/* Columns first: for COMM_OVERLAP the interior rows only need the left and
 * right ghosts, so completing the column exchange early is what makes the
 * overlap legal. cp==1 means the horizontal torus closes inside my block. */
static void exchange_columns(const Topo *t, LifeGrid *g, Halo *h) {
    if (t->cp == 1) { life_wrap_horizontal(g); return; }
    pack_columns(g, h->send_w, h->send_e);
    /* my east column fills the east neighbour's west ghost (TAG_COL_E) */
    MPI_Sendrecv(h->send_e, (int)g->rows, MPI_UINT8_T, t->nb_e, TAG_COL_E,
                 h->recv_w, (int)g->rows, MPI_UINT8_T, t->nb_w, TAG_COL_E,
                 t->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h->send_w, (int)g->rows, MPI_UINT8_T, t->nb_w, TAG_COL_W,
                 h->recv_e, (int)g->rows, MPI_UINT8_T, t->nb_e, TAG_COL_W,
                 t->cart, MPI_STATUS_IGNORE);
    unpack_columns(g, h->recv_w, h->recv_e);
}

/* Corners: each Sendrecv is a ring shift along one diagonal cycle of the
 * torus, sending to the antipodal neighbour. Corner tags name the ghost
 * they fill, so coinciding diagonal neighbours can never be confused. */
static void corners_blocking(const Topo *t, LifeGrid *g) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    if (t->rp == 1 && t->cp == 1) {
        g->cells[0] = g->cells[rows * stride + cols];
        g->cells[cols + 1] = g->cells[rows * stride + 1];
        g->cells[(rows + 1) * stride] = g->cells[stride + cols];
        g->cells[(rows + 1) * stride + cols + 1] = g->cells[stride + 1];
        return;
    }
    MPI_Sendrecv(&g->cells[rows * stride + cols], 1, MPI_UINT8_T, t->nb_se,
                 TAG_TL, &g->cells[0], 1, MPI_UINT8_T, t->nb_nw, TAG_TL,
                 t->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&g->cells[rows * stride + 1], 1, MPI_UINT8_T, t->nb_sw,
                 TAG_TR, &g->cells[cols + 1], 1, MPI_UINT8_T, t->nb_ne,
                 TAG_TR, t->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&g->cells[stride + cols], 1, MPI_UINT8_T, t->nb_ne,
                 TAG_BL, &g->cells[(rows + 1) * stride], 1, MPI_UINT8_T,
                 t->nb_sw, TAG_BL, t->cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&g->cells[stride + 1], 1, MPI_UINT8_T, t->nb_nw,
                 TAG_BR, &g->cells[(rows + 1) * stride + cols + 1], 1,
                 MPI_UINT8_T, t->nb_se, TAG_BR, t->cart, MPI_STATUS_IGNORE);
}

static void blocking_2d(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    exchange_columns(t, g, h);
    if (t->rp == 1)
        self_rows_2d(g);
    else {
        MPI_Sendrecv(g->cells + stride + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                     TAG_ROW_UP, g->cells + (rows + 1) * stride + 1,
                     (int)cols, MPI_UINT8_T, t->nb_s, TAG_ROW_UP, t->cart,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(g->cells + rows * stride + 1, (int)cols, MPI_UINT8_T,
                     t->nb_s, TAG_ROW_DOWN, g->cells + 1, (int)cols,
                     MPI_UINT8_T, t->nb_n, TAG_ROW_DOWN, t->cart,
                     MPI_STATUS_IGNORE);
    }
    corners_blocking(t, g);
}

/* Nonblocking: all eight segments in flight (columns included), receives
 * before sends, one Waitall in halo_finish. */
static void post_2d(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    h->nreq = 0;
    if (t->cp == 1) {
        life_wrap_horizontal(g);
    } else {
        pack_columns(g, h->send_w, h->send_e);
        MPI_Irecv(h->recv_w, (int)rows, MPI_UINT8_T, t->nb_w, TAG_COL_E,
                  t->cart, &h->req[h->nreq++]);
        MPI_Irecv(h->recv_e, (int)rows, MPI_UINT8_T, t->nb_e, TAG_COL_W,
                  t->cart, &h->req[h->nreq++]);
        MPI_Isend(h->send_e, (int)rows, MPI_UINT8_T, t->nb_e, TAG_COL_E,
                  t->cart, &h->req[h->nreq++]);
        MPI_Isend(h->send_w, (int)rows, MPI_UINT8_T, t->nb_w, TAG_COL_W,
                  t->cart, &h->req[h->nreq++]);
    }
    if (t->rp == 1) {
        self_rows_2d(g);
    } else {
        MPI_Irecv(g->cells + (rows + 1) * stride + 1, (int)cols,
                  MPI_UINT8_T, t->nb_s, TAG_ROW_UP, t->cart,
                  &h->req[h->nreq++]);
        MPI_Irecv(g->cells + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                  TAG_ROW_DOWN, t->cart, &h->req[h->nreq++]);
        MPI_Isend(g->cells + stride + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                  TAG_ROW_UP, t->cart, &h->req[h->nreq++]);
        MPI_Isend(g->cells + rows * stride + 1, (int)cols, MPI_UINT8_T,
                  t->nb_s, TAG_ROW_DOWN, t->cart, &h->req[h->nreq++]);
    }
    if (t->rp == 1 && t->cp == 1) {
        corners_blocking(t, g);
        return;
    }
    MPI_Irecv(&g->cells[0], 1, MPI_UINT8_T, t->nb_nw, TAG_TL, t->cart,
              &h->req[h->nreq++]);
    MPI_Irecv(&g->cells[cols + 1], 1, MPI_UINT8_T, t->nb_ne, TAG_TR,
              t->cart, &h->req[h->nreq++]);
    MPI_Irecv(&g->cells[(rows + 1) * stride], 1, MPI_UINT8_T, t->nb_sw,
              TAG_BL, t->cart, &h->req[h->nreq++]);
    MPI_Irecv(&g->cells[(rows + 1) * stride + cols + 1], 1, MPI_UINT8_T,
              t->nb_se, TAG_BR, t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[rows * stride + cols], 1, MPI_UINT8_T, t->nb_se,
              TAG_TL, t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[rows * stride + 1], 1, MPI_UINT8_T, t->nb_sw,
              TAG_TR, t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[stride + cols], 1, MPI_UINT8_T, t->nb_ne, TAG_BL,
              t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[stride + 1], 1, MPI_UINT8_T, t->nb_nw, TAG_BR,
              t->cart, &h->req[h->nreq++]);
}

/* Overlap variant: columns complete synchronously, then rows and corners
 * stay in flight while the caller computes interior rows. */
static void post_2d_overlap(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    h->nreq = 0;
    exchange_columns(t, g, h);
    if (t->rp == 1) {
        self_rows_2d(g);
    } else {
        MPI_Irecv(g->cells + (rows + 1) * stride + 1, (int)cols,
                  MPI_UINT8_T, t->nb_s, TAG_ROW_UP, t->cart,
                  &h->req[h->nreq++]);
        MPI_Irecv(g->cells + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                  TAG_ROW_DOWN, t->cart, &h->req[h->nreq++]);
        MPI_Isend(g->cells + stride + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                  TAG_ROW_UP, t->cart, &h->req[h->nreq++]);
        MPI_Isend(g->cells + rows * stride + 1, (int)cols, MPI_UINT8_T,
                  t->nb_s, TAG_ROW_DOWN, t->cart, &h->req[h->nreq++]);
    }
    if (t->rp == 1 && t->cp == 1) {
        corners_blocking(t, g);
        return;
    }
    MPI_Irecv(&g->cells[0], 1, MPI_UINT8_T, t->nb_nw, TAG_TL, t->cart,
              &h->req[h->nreq++]);
    MPI_Irecv(&g->cells[cols + 1], 1, MPI_UINT8_T, t->nb_ne, TAG_TR,
              t->cart, &h->req[h->nreq++]);
    MPI_Irecv(&g->cells[(rows + 1) * stride], 1, MPI_UINT8_T, t->nb_sw,
              TAG_BL, t->cart, &h->req[h->nreq++]);
    MPI_Irecv(&g->cells[(rows + 1) * stride + cols + 1], 1, MPI_UINT8_T,
              t->nb_se, TAG_BR, t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[rows * stride + cols], 1, MPI_UINT8_T, t->nb_se,
              TAG_TL, t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[rows * stride + 1], 1, MPI_UINT8_T, t->nb_sw,
              TAG_TR, t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[stride + cols], 1, MPI_UINT8_T, t->nb_ne, TAG_BL,
              t->cart, &h->req[h->nreq++]);
    MPI_Isend(&g->cells[stride + 1], 1, MPI_UINT8_T, t->nb_nw, TAG_BR,
              t->cart, &h->req[h->nreq++]);
}

/* RMA: one fence epoch per step. Rows and corners are Put directly into
 * the neighbour's grid window; columns (non-contiguous) are Put from
 * packed buffers into the neighbour's aux window. Same row band shares
 * local_rows, so column segment lengths always match. */
static void rma_post_2d(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    if (t->size == 1) {
        life_wrap_horizontal(g);
        self_rows_2d(g);
        corners_blocking(t, g);
        return;
    }
    if (t->cp == 1) life_wrap_horizontal(g);
    else pack_columns(g, h->send_w, h->send_e);
    if (t->rp == 1) self_rows_2d(g);
    const MPI_Win win = h->win_cells[h->cur];
    MPI_Win_fence(0, win);
    if (h->win_aux) MPI_Win_fence(0, h->win_aux);
    if (t->rp > 1) {
        size_t r1, c1;
        block_at(t, (t->pr - 1 + t->rp) % t->rp, t->pc, &r1, &c1);
        const MPI_Aint ns = (MPI_Aint)(c1 + 2);
        MPI_Put(g->cells + stride + 1, (int)cols, MPI_UINT8_T, t->nb_n,
                (MPI_Aint)((r1 + 1) * ns) + 1, (int)cols, MPI_UINT8_T, win);
        block_at(t, (t->pr + 1) % t->rp, t->pc, &r1, &c1);
        MPI_Put(g->cells + rows * stride + 1, (int)cols, MPI_UINT8_T,
                t->nb_s, 1, (int)cols, MPI_UINT8_T, win);
    }
    if (t->cp > 1) {
        MPI_Put(h->send_e, (int)rows, MPI_UINT8_T, t->nb_e, 0, (int)rows,
                MPI_UINT8_T, h->win_aux);
        MPI_Put(h->send_w, (int)rows, MPI_UINT8_T, t->nb_w, (MPI_Aint)rows,
                (int)rows, MPI_UINT8_T, h->win_aux);
    }
    if (t->rp != 1 || t->cp != 1) {
        size_t r1, c1;
        /* my top-left cell fills the NW neighbour's bottom-right corner */
        block_at(t, (t->pr - 1 + t->rp) % t->rp, (t->pc - 1 + t->cp) % t->cp,
                 &r1, &c1);
        MPI_Put(&g->cells[stride + 1], 1, MPI_UINT8_T, t->nb_nw,
                (MPI_Aint)((r1 + 1) * (c1 + 2)) + c1 + 1, 1, MPI_UINT8_T,
                win);
        /* my top-right cell fills the NE neighbour's bottom-left corner */
        block_at(t, (t->pr - 1 + t->rp) % t->rp, (t->pc + 1) % t->cp,
                 &r1, &c1);
        MPI_Put(&g->cells[stride + cols], 1, MPI_UINT8_T, t->nb_ne,
                (MPI_Aint)((r1 + 1) * (c1 + 2)), 1, MPI_UINT8_T, win);
        /* my bottom-left cell fills the SW neighbour's top-right corner */
        block_at(t, (t->pr + 1) % t->rp, (t->pc - 1 + t->cp) % t->cp,
                 &r1, &c1);
        MPI_Put(&g->cells[rows * stride + 1], 1, MPI_UINT8_T, t->nb_sw,
                (MPI_Aint)(c1 + 1), 1, MPI_UINT8_T, win);
        /* my bottom-right cell fills the SE neighbour's top-left corner */
        block_at(t, (t->pr + 1) % t->rp, (t->pc + 1) % t->cp, &r1, &c1);
        MPI_Put(&g->cells[rows * stride + cols], 1, MPI_UINT8_T, t->nb_se,
                0, 1, MPI_UINT8_T, win);
    }
}

/* ---- sendrecv2: explicit MPI_Send/MPI_Recv, parity staggered ---- */

/* One send/receive pair of a ring shift, ordered by parity: even ranks send
 * first, odd ranks receive first. Along any message cycle the chain of
 * "send waits for the next rank's posted receive" terminates at the first
 * opposite-parity rank, so this is deadlock-free for every cycle length. */
static void srx_pair(int send_first, const void *sbuf, int count, int dest,
                     int tag, void *rbuf, int src, MPI_Comm cart) {
    if (send_first) {
        MPI_Send(sbuf, count, MPI_UINT8_T, dest, tag, cart);
        MPI_Recv(rbuf, count, MPI_UINT8_T, src, tag, cart, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(rbuf, count, MPI_UINT8_T, src, tag, cart, MPI_STATUS_IGNORE);
        MPI_Send(sbuf, count, MPI_UINT8_T, dest, tag, cart);
    }
}

static void sendrecv2_1d(const Topo *t, LifeGrid *g) {
    const size_t rows = g->rows, stride = g->stride;
    if (t->size == 1) { self_1d(g); return; }
    const int sf = !(t->rank & 1);
    srx_pair(sf, g->cells + stride, (int)stride, t->nb_up, TAG_UP,
             g->cells + (rows + 1) * stride, t->nb_down, t->cart);
    srx_pair(sf, g->cells + rows * stride, (int)stride, t->nb_down, TAG_DOWN,
             g->cells, t->nb_up, t->cart);
}

static void sendrecv2_2d(const Topo *t, LifeGrid *g, Halo *h) {
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    /* each phase staggers by the coordinate that alternates along its cycle */
    if (t->cp == 1) life_wrap_horizontal(g);
    else {
        const int sf = !(t->pc & 1);
        pack_columns(g, h->send_w, h->send_e);
        srx_pair(sf, h->send_e, (int)g->rows, t->nb_e, TAG_COL_E,
                 h->recv_w, t->nb_w, t->cart);
        srx_pair(sf, h->send_w, (int)g->rows, t->nb_w, TAG_COL_W,
                 h->recv_e, t->nb_e, t->cart);
        unpack_columns(g, h->recv_w, h->recv_e);
    }
    if (t->rp == 1) self_rows_2d(g);
    else {
        const int sf = !(t->pr & 1);
        srx_pair(sf, g->cells + stride + 1, (int)cols, t->nb_n, TAG_ROW_UP,
                 g->cells + (rows + 1) * stride + 1, t->nb_s, t->cart);
        srx_pair(sf, g->cells + rows * stride + 1, (int)cols, t->nb_s,
                 TAG_ROW_DOWN, g->cells + 1, t->nb_n, t->cart);
    }
    if (t->rp == 1 && t->cp == 1) { corners_blocking(t, g); return; }
    /* diagonal cycle steps (1,1): pr parity alternates when rp>1, otherwise
     * the cycle degenerates to a pc walk */
    const int sf = !((t->rp > 1 ? t->pr : t->pc) & 1);
    srx_pair(sf, &g->cells[rows * stride + cols], 1, t->nb_se, TAG_TL,
             &g->cells[0], t->nb_nw, t->cart);
    srx_pair(sf, &g->cells[rows * stride + 1], 1, t->nb_sw, TAG_TR,
             &g->cells[cols + 1], t->nb_ne, t->cart);
    srx_pair(sf, &g->cells[stride + cols], 1, t->nb_ne, TAG_BL,
             &g->cells[(rows + 1) * stride], t->nb_sw, t->cart);
    srx_pair(sf, &g->cells[stride + 1], 1, t->nb_nw, TAG_BR,
             &g->cells[(rows + 1) * stride + cols + 1], t->nb_se, t->cart);
}

/* ---- persistent: Send_init/Recv_init once, Startall/Waitall per step ---- */

static int persistent_post_1d(const Topo *t, LifeGrid *g, Halo *h,
                              uint64_t step) {
    if (t->size == 1) { self_1d(g); return 0; }
    h->cur = (int)(step & 1);
    MPI_Startall(h->npg, h->pgrid[h->cur]);
    return 1;
}

static int persistent_post_2d(const Topo *t, LifeGrid *g, Halo *h,
                              uint64_t step) {
    if (t->size == 1) {
        life_wrap_horizontal(g);
        self_rows_2d(g);
        corners_blocking(t, g);
        return 0;
    }
    h->cur = (int)(step & 1);
    if (t->cp == 1) life_wrap_horizontal(g);
    else {
        pack_columns(g, h->send_w, h->send_e);  /* staging buffers are reused */
        MPI_Startall(h->npcol, h->pcol);
    }
    if (t->rp == 1) self_rows_2d(g);
    MPI_Startall(h->npg, h->pgrid[h->cur]);
    return 1;
}

#ifdef GOL_USE_OMP_TARGET
/* The halo send sources are owned rows 1 and local_rows, written by the
 * previous kernel; pull them back before any MPI call touches them. */
static void gpu_fetch_edges(LifeGrid *g) {
#pragma omp target update from(g->cells[g->stride:g->stride])
#pragma omp target update from(g->cells[g->rows * g->stride:g->stride])
}

/* Halo arrivals land in the host ghost rows; push them to the device so the
 * next kernel sees them. */
static void gpu_flush_ghosts(LifeGrid *g) {
#pragma omp target update to(g->cells[0:g->stride])
#pragma omp target update to(g->cells[(g->rows + 1) * g->stride:g->stride])
}
#endif

/* ---- unified interface ---- */

/* Posts halo transfers. Returns 1 when halo_finish must be called before
 * ghost-dependent rows are read; for COMM_OVERLAP the caller may compute
 * interior rows (which need no halo) in between. */
static int halo_post(const Topo *t, LifeGrid *g, Halo *h, CommMode mode,
                     uint64_t step) {
#ifdef GOL_USE_OMP_TARGET
    gpu_fetch_edges(g);   /* halo send sources are device-resident up to here */
#endif
    if (mode == COMM_RMA) h->cur = (int)(step & 1);
    if (t->decomp == DECOMP_1D) {
        switch (mode) {
        case COMM_BLOCKING:
            blocking_1d(t, g);
#ifdef GOL_USE_OMP_TARGET
            /* Blocking completes in-line, so halo_finish is never called and
             * the received ghosts would otherwise never reach the device. */
            gpu_flush_ghosts(g);
#endif
            return 0;
        case COMM_NONBLOCKING:
        case COMM_OVERLAP:  post_1d(t, g, h); return 1;
        case COMM_RMA:      rma_post_1d(t, g, h); return 1;
        case COMM_PERSISTENT: return persistent_post_1d(t, g, h, step);
        case COMM_SENDRECV2: sendrecv2_1d(t, g); return 0;
        }
    } else {
        switch (mode) {
        case COMM_BLOCKING:
            blocking_2d(t, g, h);
#ifdef GOL_USE_OMP_TARGET
            gpu_flush_ghosts(g);   /* 2d is rejected before the loop, kept honest */
#endif
            return 0;
        case COMM_NONBLOCKING: post_2d(t, g, h); return 1;
        case COMM_OVERLAP:  post_2d_overlap(t, g, h); return 1;
        case COMM_RMA:      rma_post_2d(t, g, h); return 1;
        case COMM_PERSISTENT: return persistent_post_2d(t, g, h, step);
        case COMM_SENDRECV2: sendrecv2_2d(t, g, h); return 0;
        }
    }
    return 0;
}

static void halo_finish(const Topo *t, LifeGrid *g, Halo *h, CommMode mode) {
    if (mode == COMM_PERSISTENT) {
        if (h->npg) MPI_Waitall(h->npg, h->pgrid[h->cur], MPI_STATUSES_IGNORE);
        if (h->npcol) {
            MPI_Waitall(h->npcol, h->pcol, MPI_STATUSES_IGNORE);
            unpack_columns(g, h->recv_w, h->recv_e);
        }
    } else if (mode == COMM_RMA) {
        if (h->win_cells[0]) MPI_Win_fence(0, h->win_cells[h->cur]);
        if (h->win_aux) MPI_Win_fence(0, h->win_aux);
        if (t->decomp == DECOMP_2D && t->cp > 1)
            unpack_columns(g, h->recv_w, h->recv_e);
    } else {
        if (h->nreq) MPI_Waitall(h->nreq, h->req, MPI_STATUSES_IGNORE);
        /* 2D nonblocking received the ghost columns into staging buffers */
        if (mode == COMM_NONBLOCKING && t->decomp == DECOMP_2D && t->cp > 1)
            unpack_columns(g, h->recv_w, h->recv_e);
    }
#ifdef GOL_USE_OMP_TARGET
    gpu_flush_ghosts(g);  /* ghost rows now hold this step's halo arrivals */
#endif
}

/* ------------------------------------------------------------------ */
/* Update helpers                                                       */
/* ------------------------------------------------------------------ */

#ifdef GOL_USE_OMP_TARGET
/* Device kernel with the column wrap folded in. Folding the wrap into the
 * kernel matters: the host copy of the grid is stale between steps (only the
 * halo rows are synced), so a host life_wrap_horizontal would read garbage.
 * (c==1 ? cols : c-1) reproduces that wrap for every cols >= 1, including the
 * cols < 3 cases where the left and right neighbour coincide.
 * The loop is inlined here rather than calling a declare-target helper because
 * NVHPC rejects a target region whose only statement is such a call
 * (NVC++-S-0155: a target construct may hold one TEAMS and nothing else). */
static void update_range_target(const uint8_t *prev, uint8_t *next,
                                size_t stride, size_t cols, size_t r1,
                                size_t r2) {
#pragma omp target teams distribute parallel for schedule(static)
    for (long long rr = (long long)r1; rr <= (long long)r2; ++rr) {
        const size_t r = (size_t)rr;
        const uint8_t *up = prev + (r - 1) * stride;
        const uint8_t *mid = prev + r * stride;
        const uint8_t *down = prev + (r + 1) * stride;
        uint8_t *dst = next + r * stride;
        for (size_t c = 1; c <= cols; ++c) {
            const size_t cl = c == 1 ? cols : c - 1;
            const size_t cr = c == cols ? 1 : c + 1;
            const unsigned n = (unsigned)up[cl] + up[c] + up[cr] +
                               mid[cl] + mid[cr] +
                               down[cl] + down[c] + down[cr];
            dst[c] = (uint8_t)(n == 3 || (mid[c] && n == 2));
        }
    }
}

/* Both buffers stay mapped for the whole run; per step only the two vertical
 * ghost rows travel, so PCIe traffic is O(stride) instead of O(grid). */
static void gpu_enter(const LifeGrid *a, const LifeGrid *b) {
    const size_t na = (a->rows + 2) * a->stride;
    const size_t nb = (b->rows + 2) * b->stride;
#pragma omp target enter data map(to: a->cells[0:na])
#pragma omp target enter data map(to: b->cells[0:nb])
}

static void gpu_exit(LifeGrid *a, LifeGrid *b) {
    const size_t na = (a->rows + 2) * a->stride;
    const size_t nb = (b->rows + 2) * b->stride;
#pragma omp target exit data map(from: a->cells[0:na])
#pragma omp target exit data map(from: b->cells[0:nb])
}
#endif

static void update_range(const LifeGrid *prev, LifeGrid *next, size_t r1,
                         size_t r2) {
    if (r1 > r2) return;
#ifdef GOL_USE_OMP_TARGET
    update_range_target(prev->cells, next->cells, prev->stride, prev->cols,
                        r1, r2);
#else
#ifdef GOL_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long r = (long long)r1; r <= (long long)r2; ++r)
        life_update_row(prev->cells, next->cells, prev->stride, (size_t)r,
                        prev->cols);
#endif
}

static void update_all(const LifeGrid *prev, LifeGrid *next) {
    update_range(prev, next, 1, prev->rows);
}

/* Interior rows need no halo data; border rows do. Overlap computes the
 * interior between halo_post and halo_finish. With local_rows < 3 the
 * interior is empty and overlap degenerates to plain nonblocking. */
static void update_interior(const LifeGrid *prev, LifeGrid *next) {
    update_range(prev, next, 2, prev->rows - 1);
}

static void update_borders(const LifeGrid *prev, LifeGrid *next) {
    update_range(prev, next, 1, 1);
    if (prev->rows >= 2) update_range(prev, next, prev->rows, prev->rows);
}

/* ------------------------------------------------------------------ */
/* Block-major pack/unpack for Scatterv/Gatherv. A 2D block's rows are  */
/* strided inside the full grid, so each rank's block is gathered into  */
/* one contiguous run (for 1d this equals the plain row packing).       */
/* ------------------------------------------------------------------ */

static void pack_rows(const LifeGrid *g, uint8_t *dst) {
    for (size_t r = 1; r <= g->rows; ++r)
        memcpy(dst + (r - 1) * g->cols, g->cells + r * g->stride + 1, g->cols);
}

static void unpack_rows(const uint8_t *src, LifeGrid *g) {
    for (size_t r = 1; r <= g->rows; ++r)
        memcpy(g->cells + r * g->stride + 1, src + (r - 1) * g->cols, g->cols);
}

static void pack_full(const LifeGrid *full, const Topo *t, uint8_t *dst) {
    size_t off = 0;
    for (int p = 0; p < t->size; ++p) {
        size_t lp, lo, lc, co;
        rank_shape(t, p, &lp, &lo, &lc, &co);
        for (size_t r = 0; r < lp; ++r)
            memcpy(dst + off + r * lc,
                   full->cells + (lo + r + 1) * full->stride + 1 + co, lc);
        off += lp * lc;
    }
}

static void unpack_full(const uint8_t *src, const Topo *t, LifeGrid *full) {
    size_t off = 0;
    for (int p = 0; p < t->size; ++p) {
        size_t lp, lo, lc, co;
        rank_shape(t, p, &lp, &lo, &lc, &co);
        for (size_t r = 0; r < lp; ++r)
            memcpy(full->cells + (lo + r + 1) * full->stride + 1 + co,
                   src + off + r * lc, lc);
        off += lp * lc;
    }
}

/* Reproducible init for a 2D block: same seed => same global grid. */
static void fill_block_random(LifeGrid *g, const Topo *t, uint64_t seed,
                              uint32_t density) {
    for (size_t r = 0; r < g->rows; ++r)
        for (size_t c = 0; c < g->cols; ++c)
            g->cells[(r + 1) * g->stride + c + 1] = life_initial_cell(
                seed, (uint64_t)(t->row_off + r) * t->W + t->col_off + c,
                density);
}

int main(int argc, char **argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
        MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    Options o = {64, 64, 100, 1, 500000, NULL, NULL, DECOMP_1D,
                 COMM_BLOCKING, 0};
    const int parsed = parse_args(argc, argv, &o);
    if (parsed != 0) {
        if (rank == 0) usage(parsed > 0 ? stdout : stderr);
        MPI_Finalize();
        return parsed > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

#ifdef GOL_USE_OMP_TARGET
    /* The GPU build covers 1D only: the 2D halo needs every row's first and
     * last column on the host, which would defeat the "only ghost rows move"
     * design. Fail loudly rather than silently produce wrong results. */
    if (o.decomp == DECOMP_2D) {
        if (rank == 0)
            fprintf(stderr, "error: the GPU build supports --decomp 1d only\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (omp_get_num_devices() < 1 && !getenv("GOL_ALLOW_HOST_TARGET")) {
        if (rank == 0)
            fprintf(stderr, "error: no OpenMP target device visible; the GPU "
                            "build refuses to fall back to the host "
                            "(set GOL_ALLOW_HOST_TARGET=1 to force a logic-only "
                            "run; it is NOT a GPU measurement)\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
#endif

#if defined(GOL_USE_OPENMP) && !defined(GOL_USE_OMP_TARGET)
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;   /* GPU variant: no host thread parallelism */
#endif

    double total_start = MPI_Wtime();
    int rc = EXIT_SUCCESS;
    unsigned long long rows = o.rows, cols = o.cols;
    LifeGrid full = {0}, current = {0}, next = {0}, out = {0};
    int *counts = NULL, *displs = NULL;
    uint8_t *sendbuf = NULL, *recvbuf = NULL, *packed = NULL;
    Topo topo;
    Halo halo;
    double kernel_seconds = 0.0, total_seconds = 0.0;
    /* Optional halo accounting (GOL_HALO_STATS): wall time spent inside
     * halo_post/halo_finish only, reduced with MAX like kernel_seconds and
     * reported as an additive JSON field. Off by default. */
    double halo_seconds = 0.0;
    const int halo_stats = getenv("GOL_HALO_STATS") != NULL;
    /* GOL_TRACE: per-rank stage markers on stderr (unbuffered) to locate
     * wedged collectives in remote batches; off by default. */
    const int gol_trace = getenv("GOL_TRACE") != NULL;
#define TRACE(m) do { \
    if (gol_trace) { fprintf(stderr, "[rank %d] %s\n", rank, m); fflush(stderr); } \
} while (0)
    memset(&halo, 0, sizeof halo);

    if (o.input) {
        int root_error = 0;
        if (rank == 0) {
            char error[160];
            if (life_read_pbm(o.input, &full, error, sizeof error)) {
                fprintf(stderr, "input error: %s\n", error);
                root_error = 1;
            } else {
                rows = full.rows;
                cols = full.cols;
                if (((o.seen & OPT_ROWS) && o.rows != rows) ||
                    ((o.seen & OPT_COLS) && o.cols != cols)) {
                    fprintf(stderr, "explicit dimensions disagree with input\n");
                    root_error = 2;
                }
            }
        }
        MPI_Bcast(&root_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        if (root_error) {
            if (rank == 0) life_grid_destroy(&full);
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (topo_init(&topo, o.decomp, rank, size, rows, cols)) {
        if (rank == 0)
            fprintf(stderr,
                    "error: %d ranks do not fit a %s decomposition of "
                    "%llux%llu\n", size, decomp_name(o.decomp), rows, cols);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (rows > (unsigned long long)INT_MAX - 2 ||
        cols > (unsigned long long)INT_MAX - 2 ||
        rows > (unsigned long long)INT_MAX / cols) {
        if (rank == 0)
            fprintf(stderr, "error: grid dimensions overflow MPI int counts\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (life_grid_create(&current, topo.local_rows, topo.local_cols) ||
        life_grid_create(&next, topo.local_rows, topo.local_cols)) {
        if (rank == 0) fprintf(stderr, "allocation failed\n");
        rc = EXIT_FAILURE;
        goto cleanup;
    }

    counts = malloc((size_t)size * sizeof *counts);
    displs = malloc((size_t)size * sizeof *displs);
    recvbuf = malloc(topo.local_rows * topo.local_cols);
    if (!counts || !displs || !recvbuf) {
        if (rank == 0) fprintf(stderr, "allocation failed\n");
        rc = EXIT_FAILURE;
        goto cleanup;
    }
    {
        int off = 0;
        for (int p = 0; p < size; ++p) {
            size_t lp, lo, lc, co;
            rank_shape(&topo, p, &lp, &lo, &lc, &co);
            counts[p] = (int)(lp * lc);
            displs[p] = off;            /* block-major layout */
            off += counts[p];
        }
    }

    if (o.input) {
        if (rank == 0) {
            sendbuf = malloc((size_t)(rows * cols));
            if (!sendbuf) {
                fprintf(stderr, "allocation failed\n");
                rc = EXIT_FAILURE;
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rc != EXIT_SUCCESS) goto cleanup;
        if (rank == 0) pack_full(&full, &topo, sendbuf);
        MPI_Scatterv(sendbuf, counts, displs, MPI_UINT8_T, recvbuf,
                     (int)(topo.local_rows * topo.local_cols), MPI_UINT8_T, 0,
                     MPI_COMM_WORLD);
        unpack_rows(recvbuf, &current);
    } else if (o.decomp == DECOMP_1D) {
        life_fill_random(&current, o.seed, (uint32_t)o.density, topo.row_off);
    } else {
        fill_block_random(&current, &topo, o.seed, (uint32_t)o.density);
    }

    halo_init(&topo, &current, &next, &halo, o.comm);
#ifdef GOL_USE_OMP_TARGET
    gpu_enter(&current, &next);
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    TRACE("pre_barrier");
    const double kernel_start = MPI_Wtime();
    double halo_local = 0.0, h0 = 0.0;
    for (uint64_t step = 0; step < o.steps; ++step) {
#ifndef GOL_USE_OMP_TARGET
        /* The GPU kernel wraps columns itself, so the host copy stays stale. */
        if (o.decomp == DECOMP_1D) life_wrap_horizontal(&current);
#endif
        if (halo_stats) h0 = MPI_Wtime();
        const int deferred = halo_post(&topo, &current, &halo, o.comm, step);
        if (halo_stats) halo_local += MPI_Wtime() - h0;
        if (deferred && o.comm == COMM_OVERLAP)
            update_interior(&current, &next);
        if (deferred) {
            if (halo_stats) h0 = MPI_Wtime();
            halo_finish(&topo, &current, &halo, o.comm);
            if (halo_stats) halo_local += MPI_Wtime() - h0;
        }
        if (deferred && o.comm == COMM_OVERLAP)
            update_borders(&current, &next);
        else
            update_all(&current, &next);
        life_swap(&current, &next);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    TRACE("post_barrier");
    const double kernel_local = MPI_Wtime() - kernel_start;
    MPI_Reduce(&kernel_local, &kernel_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    TRACE("kernel_reduce");
#ifdef GOL_USE_OMP_TARGET
    gpu_exit(&current, &next);   /* final grid back on the host for packing */
#endif
    if (halo_stats) {
        /* Reverted from a point-to-point experiment (job 1518064): the
         * cross-node wedge reproduces with both Reduce and P2P, so this stays
         * at the last-validated Reduce form per plan section 8. The wedge is
         * Vanda-specific; Atlas is unaffected and Vanda batches run with
         * GOL_HALO_STATS off. */
        MPI_Reduce(&halo_local, &halo_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
    }
    TRACE("halo_reduce");

    pack_rows(&current, recvbuf);
    TRACE("pack");
    if (rank == 0) {
        packed = malloc((size_t)(rows * cols));
        if (!packed) {
            fprintf(stderr, "allocation failed\n");
            rc = EXIT_FAILURE;
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rc != EXIT_SUCCESS) goto cleanup;
    MPI_Gatherv(recvbuf, (int)(topo.local_rows * topo.local_cols),
                MPI_UINT8_T, packed, counts, displs, MPI_UINT8_T, 0,
                MPI_COMM_WORLD);
    TRACE("gatherv");

    uint64_t live = 0, hash = 0;
    if (rank == 0 && rc == EXIT_SUCCESS) {
        if (life_grid_create(&out, (size_t)rows, (size_t)cols)) {
            fprintf(stderr, "allocation failed\n");
            rc = EXIT_FAILURE;
        } else {
            if (o.decomp == DECOMP_1D)
                unpack_rows(packed, &out);
            else
                unpack_full(packed, &topo, &out);
            live = life_live_count(&out);
            hash = life_checksum(&out);
        }
    }

    const double total_local = MPI_Wtime() - total_start;
    MPI_Reduce(&total_local, &total_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    TRACE("total_reduce");
    if (rank == 0 && total_seconds < kernel_seconds) total_seconds = kernel_seconds;
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    TRACE("bcast2");

    if (rc == EXIT_SUCCESS && rank == 0) {
        if (o.output && life_write_pbm(o.output, &out)) {
            fprintf(stderr, "cannot write complete output grid\n");
            rc = EXIT_FAILURE;
        } else {
            printf("{\"interface_version\":\"%s\",\"variant\":\"%s\","
                   "\"decomp\":\"%s\",\"comm\":\"%s\","
                   "\"rows\":%zu,\"cols\":%zu,\"steps\":%" PRIu64 ","
                   "\"init\":\"%s\",\"seed\":",
                   LIFE_INTERFACE_VERSION, VARIANT_NAME,
                   decomp_name(o.decomp), comm_name(o.comm),
                   out.rows, out.cols, o.steps, o.input ? "pbm" : "random");
            if (o.input) printf("null"); else printf("%" PRIu64, o.seed);
            printf(",\"density_ppm\":");
            if (o.input) printf("null"); else printf("%" PRIu64, o.density);
            printf(",\"ranks\":%d,\"threads\":%d,\"live_cells\":%" PRIu64 ","
                   "\"checksum_fnv1a64\":\"%016" PRIx64 "\","
                   "\"kernel_seconds\":%.9f,\"total_seconds\":%.9f",
                   size, threads, live, hash, kernel_seconds, total_seconds);
            if (halo_stats) printf(",\"halo_seconds\":%.9f", halo_seconds);
            printf("}\n");
        }
    }

cleanup:
    TRACE("cleanup");
    halo_destroy(&halo);
    TRACE("win_freed");
    if (topo.decomp == DECOMP_2D && topo.cart && topo.cart != MPI_COMM_WORLD)
        MPI_Comm_free(&topo.cart);
    free(counts);
    free(displs);
    free(sendbuf);
    free(recvbuf);
    free(packed);
    life_grid_destroy(&current);
    life_grid_destroy(&next);
    life_grid_destroy(&full);
    life_grid_destroy(&out);
    TRACE("finalize");
    MPI_Finalize();
    return rc;
}
