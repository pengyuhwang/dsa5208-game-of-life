#include "life.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int life_grid_create(LifeGrid *g, size_t rows, size_t cols) {
    if (!g) return -1;
    *g = (LifeGrid){0};
    if (!rows || !cols || rows > SIZE_MAX - 2 || cols > SIZE_MAX - 2)
        return -1;
    const size_t stride = cols + 2;
    if (rows + 2 > SIZE_MAX / stride) return -1;
    uint8_t *cells = calloc((rows + 2) * stride, sizeof(*cells));
    if (!cells) return -1;
    *g = (LifeGrid){rows, cols, stride, cells};
    return 0;
}

void life_grid_destroy(LifeGrid *g) {
    if (g) { free(g->cells); *g = (LifeGrid){0}; }
}

uint8_t life_initial_cell(uint64_t seed, uint64_t index, uint32_t density) {
    /* SplitMix64 finalizer, applied independently to each global cell.
     * Unsigned 64-bit arithmetic wraps by definition. No libc rand() dependency.
     * density/1e6 is nominal; modulo introduces negligible (<1/2^44) bias.
     */
    uint64_t z = seed + UINT64_C(0x9e3779b97f4a7c15) * (index + 1);
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return (uint8_t)(z % UINT64_C(1000000) < density);
}

void life_fill_random(LifeGrid *g, uint64_t seed, uint32_t density, size_t offset) {
    for (size_t r = 0; r < g->rows; ++r)
        for (size_t c = 0; c < g->cols; ++c)
            g->cells[(r + 1) * g->stride + c + 1] =
                life_initial_cell(seed, (uint64_t)(offset + r) * g->cols + c,
                                  density);
}

void life_wrap_horizontal(LifeGrid *g) {
    for (size_t r = 1; r <= g->rows; ++r) {
        uint8_t *row = g->cells + r * g->stride;
        row[0] = row[g->cols];
        row[g->cols + 1] = row[1];
    }
}

void life_wrap_periodic(LifeGrid *g) {
    life_wrap_horizontal(g);
    memcpy(g->cells, g->cells + g->rows * g->stride, g->stride);
    memcpy(g->cells + (g->rows + 1) * g->stride,
           g->cells + g->stride, g->stride);
}

void life_update_row(const uint8_t *restrict previous, uint8_t *restrict next,
                     size_t stride, size_t row, size_t cols) {
    const uint8_t *up = previous + (row - 1) * stride;
    const uint8_t *mid = previous + row * stride;
    const uint8_t *down = previous + (row + 1) * stride;
    uint8_t *out = next + row * stride;
    for (size_t c = 1; c <= cols; ++c) {
        const unsigned n = (unsigned)up[c-1] + up[c] + up[c+1]
                         + mid[c-1] + mid[c+1]
                         + down[c-1] + down[c] + down[c+1];
        out[c] = (uint8_t)(n == 3 || (mid[c] && n == 2));
    }
}

void life_step(const LifeGrid *previous, LifeGrid *next) {
    for (size_t r = 1; r <= previous->rows; ++r)
        life_update_row(previous->cells, next->cells, previous->stride,
                        r, previous->cols);
}

void life_swap(LifeGrid *a, LifeGrid *b) {
    LifeGrid tmp = *a; *a = *b; *b = tmp;
}

uint64_t life_live_count(const LifeGrid *g) {
    uint64_t count = 0;
    for (size_t r = 1; r <= g->rows; ++r)
        for (size_t c = 1; c <= g->cols; ++c)
            count += g->cells[r * g->stride + c];
    return count;
}

uint64_t life_checksum(const LifeGrid *g) {
    /* FNV-1a-64 over interior bytes in global row-major order; halos excluded.
     * A checksum is a diagnostic, never a replacement for grid comparison.
     */
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t r = 1; r <= g->rows; ++r)
        for (size_t c = 1; c <= g->cols; ++c) {
            hash ^= g->cells[r * g->stride + c];
            hash *= UINT64_C(1099511628211);
        }
    return hash;
}

/* P1 tokenizer: bounded tokens, ASCII whitespace, # comments. */
static int token(FILE *f, char *buf, size_t size) {
    if (ferror(f)) return -1;
    if (feof(f)) return 0;
    int ch;
    do {
        ch = fgetc(f);
        if (ch == '#') {
            while (ch != '\n' && ch != EOF) ch = fgetc(f);
        }
    } while (ch != EOF && isspace((unsigned char)ch));
    if (ch == EOF) return ferror(f) ? -1 : 0;
    size_t n = 0;
    while (ch != EOF && !isspace((unsigned char)ch) && ch != '#') {
        if (n + 1 >= size) return -1;
        buf[n++] = (char)ch;
        ch = fgetc(f);
    }
    if (ch == '#') while (ch != '\n' && ch != EOF) ch = fgetc(f);
    buf[n] = '\0';
    return ferror(f) ? -1 : 1;
}

static int positive_size(const char *s, size_t *out) {
    if (!s[0]) return -1;
    for (const char *p = s; *p; ++p) if (*p < '0' || *p > '9') return -1;
    errno = 0;
    char *end;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end || !n || n > SIZE_MAX) return -1;
    *out = (size_t)n;
    return 0;
}

int life_read_pbm(const char *path, LifeGrid *g, char *error, size_t error_size) {
    FILE *f = fopen(path, "r");
    LifeGrid tmp = {0};
    char buf[64];
    size_t rows = 0, cols = 0;
    const char *why = "invalid P1 PBM header";
    if (!f) { snprintf(error, error_size, "cannot open PBM input"); return -1; }
    if (token(f, buf, sizeof buf) != 1 || strcmp(buf, "P1")) goto fail;
    if (token(f, buf, sizeof buf) != 1 || positive_size(buf, &cols)) goto fail;
    if (token(f, buf, sizeof buf) != 1 || positive_size(buf, &rows)) goto fail;
    why = "invalid dimensions or allocation failed";
    if (life_grid_create(&tmp, rows, cols)) goto fail;
    why = "expected exactly rows*cols binary PBM pixels";
    for (size_t r = 1; r <= rows; ++r)
        for (size_t c = 1; c <= cols; ++c) {
            if (token(f, buf, sizeof buf) != 1 ||
                (strcmp(buf, "0") && strcmp(buf, "1"))) goto fail;
            tmp.cells[r * tmp.stride + c] = (uint8_t)(buf[0] == '1');
        }
    if (token(f, buf, sizeof buf) != 0) goto fail;
    if (fclose(f)) {
        life_grid_destroy(&tmp);
        snprintf(error, error_size, "failed closing PBM input"); return -1;
    }
    *g = tmp;
    return 0;
fail:
    fclose(f);
    life_grid_destroy(&tmp);
    snprintf(error, error_size, "%s", why);
    return -1;
}

int life_write_pbm(const char *path, const LifeGrid *g) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    int failed = fprintf(f, "P1\n# Game of Life; 1=alive; periodic B3/S23\n%zu %zu\n",
                         g->cols, g->rows) < 0;
    for (size_t r = 1; r <= g->rows && !failed; ++r) {
        for (size_t c = 1; c <= g->cols; ++c)
            if (fprintf(f, "%u%c", (unsigned)g->cells[r*g->stride+c],
                        c == g->cols ? '\n' : ' ') < 0) { failed = 1; break; }
    }
    if (fclose(f)) failed = 1;
    return failed ? -1 : 0;
}
