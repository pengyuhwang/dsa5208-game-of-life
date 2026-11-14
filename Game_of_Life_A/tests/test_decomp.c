#include "life_decomp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* In-memory 2D exchange checks the layout contract, not MPI message matching. */
static void copy_ghosts(LifeGrid *grids, const LifeDecomp *d, int rank) {
    int nb[LIFE_NEIGHBOURS];
    life_decomp_neighbours(d, rank, nb);
    LifeGrid *g = &grids[rank];
    const size_t rows = g->rows, cols = g->cols, stride = g->stride;
    const LifeGrid *n = &grids[nb[LIFE_N]], *s = &grids[nb[LIFE_S]];
    const LifeGrid *w = &grids[nb[LIFE_W]], *e = &grids[nb[LIFE_E]];
    assert(n->cols == cols && s->cols == cols && w->rows == rows && e->rows == rows);
    memcpy(g->cells + 1, n->cells + n->rows * n->stride + 1, cols);
    memcpy(g->cells + (rows + 1) * stride + 1, s->cells + s->stride + 1, cols);
    for (size_t r = 1; r <= rows; ++r) {
        g->cells[r * stride] = w->cells[r * w->stride + w->cols];
        g->cells[r * stride + cols + 1] = e->cells[r * e->stride + 1];
    }
    const LifeGrid *nw = &grids[nb[LIFE_NW]], *ne = &grids[nb[LIFE_NE]];
    const LifeGrid *sw = &grids[nb[LIFE_SW]], *se = &grids[nb[LIFE_SE]];
    g->cells[0] = nw->cells[nw->rows * nw->stride + nw->cols];
    g->cells[cols + 1] = ne->cells[ne->rows * ne->stride + 1];
    g->cells[(rows + 1) * stride] = sw->cells[sw->stride + sw->cols];
    g->cells[(rows + 1) * stride + cols + 1] = se->cells[se->stride + 1];
}

static void check_layout(size_t height, size_t width, int ranks, int two_d) {
    LifeDecomp d;
    assert(!(two_d ? life_decomp_2d : life_decomp_1d)(&d, height, width, ranks));
    assert(d.proc_rows * d.proc_cols == ranks);
    LifeGrid whole = {0}, whole_next = {0}, rebuilt = {0};
    assert(!life_grid_create(&whole, height, width));
    assert(!life_grid_create(&whole_next, height, width));
    assert(!life_grid_create(&rebuilt, height, width));
    life_fill_random(&whole, UINT64_C(20261026), 500000, 0);
    LifeGrid *now = calloc((size_t)ranks, sizeof(*now));
    LifeGrid *next = calloc((size_t)ranks, sizeof(*next));
    uint8_t *packed = malloc(height * width), *packed2 = malloc(height * width);
    assert(now && next && packed && packed2);
    size_t cells = 0;
    for (int p = 0; p < ranks; ++p) {
        const LifeBlock b = life_decomp_block(&d, p);
        assert(b.rows && b.cols);
        cells += b.rows * b.cols;
        assert(!life_grid_create(&now[p], b.rows, b.cols));
        assert(!life_grid_create(&next[p], b.rows, b.cols));
        life_fill_block(&now[p], &b, width, UINT64_C(20261026), 500000);
        for (size_t r = 1; r <= b.rows; ++r)
            assert(!memcmp(now[p].cells + r * now[p].stride + 1,
                           whole.cells + (b.row_offset + r) * whole.stride + b.col_offset + 1,
                           b.cols));
    }
    assert(cells == height * width);
    life_pack_blocks(&d, &whole, packed);
    life_unpack_blocks(&d, packed, &rebuilt);
    assert(life_checksum(&rebuilt) == life_checksum(&whole));
    for (int t = 0; t < 12; ++t) {
        life_wrap_periodic(&whole);
        life_step(&whole, &whole_next);
        life_swap(&whole, &whole_next);
        for (int p = 0; p < ranks; ++p) copy_ghosts(now, &d, p);
        for (int p = 0; p < ranks; ++p) {
            life_step(&now[p], &next[p]);
            life_swap(&now[p], &next[p]);
        }
        /* Compare every owned cell in block-major order after each generation. */
        life_pack_blocks(&d, &whole, packed);
        uint8_t *cursor = packed2;
        for (int p = 0; p < ranks; ++p)
            for (size_t r = 1; r <= now[p].rows; ++r, cursor += now[p].cols)
                memcpy(cursor, now[p].cells + r * now[p].stride + 1, now[p].cols);
        assert(!memcmp(packed, packed2, height * width));
    }
    for (int p = 0; p < ranks; ++p) {
        life_grid_destroy(&now[p]);
        life_grid_destroy(&next[p]);
    }
    free(now); free(next); free(packed); free(packed2);
    life_grid_destroy(&whole); life_grid_destroy(&whole_next); life_grid_destroy(&rebuilt);
}

/* Prints the computed layout so Python can compare it with its own model. */
static int probe(size_t height, size_t width, int ranks, int two_d) {
    LifeDecomp d;
    if ((two_d ? life_decomp_2d : life_decomp_1d)(&d, height, width, ranks)) {
        puts("invalid");
        return 0;
    }
    printf("dims %d %d\n", d.proc_rows, d.proc_cols);
    for (int p = 0; p < ranks; ++p) {
        const LifeBlock b = life_decomp_block(&d, p);
        int nb[LIFE_NEIGHBOURS];
        life_decomp_neighbours(&d, p, nb);
        printf("%zu %zu %zu %zu", b.rows, b.cols, b.row_offset, b.col_offset);
        for (int i = 0; i < LIFE_NEIGHBOURS; ++i) printf(" %d", nb[i]);
        putchar('\n');
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 6 && !strcmp(argv[1], "--probe"))
        return probe((size_t)atoi(argv[2]), (size_t)atoi(argv[3]), atoi(argv[4]),
                     !strcmp(argv[5], "2d"));
    static const struct { size_t h, w; int p; } layouts[] = {
        {1, 1, 1}, {1, 9, 3}, {9, 1, 3}, {2, 2, 4}, {3, 3, 9}, {4, 7, 4}, {4, 7, 16},
        {7, 4, 16}, {5, 7, 2}, {5, 7, 6}, {7, 9, 4}, {7, 9, 9}, {12, 13, 8}, {12, 13, 12},
        {97, 53, 8}, {64, 64, 16}, {16, 16, 64},
    };
    const size_t count = sizeof layouts / sizeof layouts[0];
    for (size_t i = 0; i < count; ++i) check_layout(layouts[i].h, layouts[i].w, layouts[i].p, 1);
    for (size_t i = 0; i < count; ++i)
        if ((size_t)layouts[i].p <= layouts[i].h)
            check_layout(layouts[i].h, layouts[i].w, layouts[i].p, 0);
    LifeDecomp d;
    assert(life_decomp_2d(&d, 2, 2, 5) && life_decomp_2d(&d, 1, 1, 2));
    assert(life_decomp_2d(&d, 2, 2, 0) && life_decomp_1d(&d, 2, 9, 3));
    puts("PASS: 2D layouts, block-major packing and in-memory exchange; no MPI claim");
    return 0;
}
