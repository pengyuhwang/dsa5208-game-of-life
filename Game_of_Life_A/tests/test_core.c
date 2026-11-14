#include "life.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* In-memory slab exchange checks the shared halo contract, not MPI itself. */
static void slabs(size_t height, size_t width, size_t ranks) {
    LifeGrid whole={0},whole_next={0};
    LifeGrid *parts=calloc(ranks,sizeof(*parts));
    LifeGrid *next=calloc(ranks,sizeof(*next));
    assert(parts && next);
    assert(!life_grid_create(&whole,height,width));
    assert(!life_grid_create(&whole_next,height,width));
    life_fill_random(&whole,UINT64_MAX,500000,0);
    size_t offset=0;
    for (size_t p=0;p<ranks;++p) {
        const size_t count=height/ranks+(p<height%ranks);
        assert(!life_grid_create(&parts[p],count,width));
        assert(!life_grid_create(&next[p],count,width));
        life_fill_random(&parts[p],UINT64_MAX,500000,offset);
        for (size_t r=1;r<=count;++r)
            assert(!memcmp(parts[p].cells+r*parts[p].stride+1,
                           whole.cells+(offset+r)*whole.stride+1,width));
        offset+=count;
    }
    for (size_t t=0;t<12;++t) {
        life_wrap_periodic(&whole);life_step(&whole,&whole_next);
        for (size_t p=0;p<ranks;++p) life_wrap_horizontal(&parts[p]);
        for (size_t p=0;p<ranks;++p) {
            LifeGrid *g=&parts[p],*up=&parts[(p+ranks-1)%ranks],*down=&parts[(p+1)%ranks];
            memcpy(g->cells,up->cells+up->rows*up->stride,g->stride);
            memcpy(g->cells+(g->rows+1)*g->stride,down->cells+down->stride,g->stride);
            life_step(g,&next[p]);
        }
        life_swap(&whole,&whole_next);offset=0;
        for (size_t p=0;p<ranks;++p) {
            life_swap(&parts[p],&next[p]);
            for (size_t r=1;r<=parts[p].rows;++r)
                assert(!memcmp(parts[p].cells+r*parts[p].stride+1,
                       whole.cells+(offset+r)*whole.stride+1,width));
            offset+=parts[p].rows;
        }
    }
    for (size_t p=0;p<ranks;++p) {
        life_grid_destroy(&parts[p]);life_grid_destroy(&next[p]);
    }
    free(parts);free(next);life_grid_destroy(&whole);life_grid_destroy(&whole_next);
}

int main(void) {
    LifeGrid invalid={0};
    assert(life_grid_create(&invalid,0,4));
    assert(life_grid_create(&invalid,SIZE_MAX,4));
    assert(life_grid_create(&invalid,SIZE_MAX/2,SIZE_MAX/2));
    assert(life_grid_create(NULL,2,2));
    life_grid_destroy(&invalid);life_grid_destroy(&invalid);
    for (size_t p=1;p<=4;p*=2) { slabs(7,11,p);slabs(p,7,p);slabs(9,1,p); }
    puts("PASS: allocation guards and 9 in-memory slab layouts over 12 steps; no MPI claim");
    return 0;
}
