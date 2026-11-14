#ifndef LIFE_DECOMP_H
#define LIFE_DECOMP_H

#include "life.h"

/* Process grid over a global H x W torus. Ranks are row-major, matching
 * MPI_Cart_create with reorder=0. Nothing here calls MPI, so every layout can
 * be tested without a launcher; the transport belongs to the MPI driver.
 */
typedef struct {
    size_t height, width;
    int proc_rows, proc_cols;
} LifeDecomp;

/* Owned block of one rank. Unlike the 1D contract, cols is LOCAL here. */
typedef struct {
    int proc_row, proc_col;
    size_t rows, cols, row_offset, col_offset;
} LifeBlock;

enum { LIFE_N, LIFE_S, LIFE_W, LIFE_E, LIFE_NW, LIFE_NE, LIFE_SW, LIFE_SE,
       LIFE_NEIGHBOURS };

void life_split(size_t total, int parts, int index, size_t *length, size_t *offset);
/* Both return nonzero when the ranks cannot give every block at least one cell. */
int life_decomp_1d(LifeDecomp *d, size_t height, size_t width, int ranks);
int life_decomp_2d(LifeDecomp *d, size_t height, size_t width, int ranks);
LifeBlock life_decomp_block(const LifeDecomp *d, int rank);
/* A neighbour may be the caller or repeat another entry on small process grids. */
void life_decomp_neighbours(const LifeDecomp *d, int rank, int neighbour[LIFE_NEIGHBOURS]);
void life_fill_block(LifeGrid *grid, const LifeBlock *block, size_t width,
                     uint64_t seed, uint32_t density_ppm);
/* Block-major order: rank 0's owned cells, then rank 1's, for Scatterv/Gatherv. */
void life_pack_blocks(const LifeDecomp *d, const LifeGrid *full, uint8_t *packed);
void life_unpack_blocks(const LifeDecomp *d, const uint8_t *packed, LifeGrid *full);

#endif
