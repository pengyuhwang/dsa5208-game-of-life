#ifndef LIFE_H
#define LIFE_H

#include <stddef.h>
#include <stdint.h>

#define LIFE_INTERFACE_VERSION "1.0"

/* Owned rows [1, rows], columns [1, cols]; one ghost cell on each edge.
 * In 1D MPI, rows is local_rows, cols is the GLOBAL width. In 2D, cols is local
 * too; see life_decomp.h. Empty blocks are outside this interface.
 */
typedef struct {
    size_t rows, cols, stride;
    uint8_t *cells;
} LifeGrid;

/* Pass an empty/zero-initialized grid; destroy before allocating it again. */
int life_grid_create(LifeGrid *grid, size_t rows, size_t cols);
void life_grid_destroy(LifeGrid *grid);
uint8_t life_initial_cell(uint64_t seed, uint64_t global_index,
                          uint32_t density_ppm);
/* Whole rows only (serial and 1D); 2D blocks use life_fill_block. */
void life_fill_random(LifeGrid *grid, uint64_t seed, uint32_t density_ppm,
                      size_t global_row_offset);
/* MPI: call horizontal first, then exchange COMPLETE first/last owned rows
 * (stride bytes each). Serial: call life_wrap_periodic for both dimensions.
 */
void life_wrap_horizontal(LifeGrid *grid);
void life_wrap_periodic(LifeGrid *grid);
/* Thread-safe for distinct row values; previous and next must not overlap.
 * Both grids must have identical dimensions. Caller prepares previous halos
 * and synchronizes all updates before swapping the two grids.
 */
void life_update_row(const uint8_t *restrict previous, uint8_t *restrict next,
                     size_t stride, size_t row, size_t cols);
void life_step(const LifeGrid *previous, LifeGrid *next);
void life_swap(LifeGrid *a, LifeGrid *b);
uint64_t life_live_count(const LifeGrid *grid);
uint64_t life_checksum(const LifeGrid *grid);
int life_read_pbm(const char *path, LifeGrid *grid, char *error, size_t error_size);
int life_write_pbm(const char *path, const LifeGrid *grid);

#endif
