#include "life_decomp.h"

#include <string.h>

void life_split(size_t total, int parts, int index, size_t *length, size_t *offset) {
    const size_t p = (size_t)parts, i = (size_t)index;
    const size_t base = total / p, extra = total % p;
    *length = base + (i < extra);
    *offset = i * base + (i < extra ? i : extra);
}

int life_decomp_1d(LifeDecomp *d, size_t height, size_t width, int ranks) {
    if (ranks < 1 || !width || (size_t)ranks > height) return -1;
    *d = (LifeDecomp){height, width, ranks, 1};
    return 0;
}

int life_decomp_2d(LifeDecomp *d, size_t height, size_t width, int ranks) {
    if (ranks < 1 || !height || !width) return -1;
    int small = 1;
    while ((long long)(small + 1) * (small + 1) <= ranks) ++small;
    /* Most balanced factor pair first; fewer process rows when both orders fit. */
    for (; small >= 1; --small) {
        if (ranks % small) continue;
        const int large = ranks / small;
        if ((size_t)small <= height && (size_t)large <= width) {
            *d = (LifeDecomp){height, width, small, large};
            return 0;
        }
        if ((size_t)large <= height && (size_t)small <= width) {
            *d = (LifeDecomp){height, width, large, small};
            return 0;
        }
    }
    return -1;
}

LifeBlock life_decomp_block(const LifeDecomp *d, int rank) {
    LifeBlock b = {rank / d->proc_cols, rank % d->proc_cols, 0, 0, 0, 0};
    life_split(d->height, d->proc_rows, b.proc_row, &b.rows, &b.row_offset);
    life_split(d->width, d->proc_cols, b.proc_col, &b.cols, &b.col_offset);
    return b;
}

void life_decomp_neighbours(const LifeDecomp *d, int rank, int neighbour[LIFE_NEIGHBOURS]) {
    const int R = d->proc_rows, C = d->proc_cols;
    const int r = rank / C, c = rank % C;
    const int up = (r + R - 1) % R, down = (r + 1) % R;
    const int left = (c + C - 1) % C, right = (c + 1) % C;
    neighbour[LIFE_N] = up * C + c;
    neighbour[LIFE_S] = down * C + c;
    neighbour[LIFE_W] = r * C + left;
    neighbour[LIFE_E] = r * C + right;
    neighbour[LIFE_NW] = up * C + left;
    neighbour[LIFE_NE] = up * C + right;
    neighbour[LIFE_SW] = down * C + left;
    neighbour[LIFE_SE] = down * C + right;
}

void life_fill_block(LifeGrid *g, const LifeBlock *b, size_t width,
                     uint64_t seed, uint32_t density) {
    for (size_t r = 0; r < g->rows; ++r)
        for (size_t c = 0; c < g->cols; ++c)
            g->cells[(r + 1) * g->stride + c + 1] = life_initial_cell(
                seed, (uint64_t)(b->row_offset + r) * width + b->col_offset + c,
                density);
}

void life_pack_blocks(const LifeDecomp *d, const LifeGrid *full, uint8_t *packed) {
    for (int p = 0; p < d->proc_rows * d->proc_cols; ++p) {
        const LifeBlock b = life_decomp_block(d, p);
        for (size_t r = 0; r < b.rows; ++r, packed += b.cols)
            memcpy(packed, full->cells + (b.row_offset + r + 1) * full->stride
                               + b.col_offset + 1, b.cols);
    }
}

void life_unpack_blocks(const LifeDecomp *d, const uint8_t *packed, LifeGrid *full) {
    for (int p = 0; p < d->proc_rows * d->proc_cols; ++p) {
        const LifeBlock b = life_decomp_block(d, p);
        for (size_t r = 0; r < b.rows; ++r, packed += b.cols)
            memcpy(full->cells + (b.row_offset + r + 1) * full->stride
                       + b.col_offset + 1, packed, b.cols);
    }
}
