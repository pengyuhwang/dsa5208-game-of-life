#define _POSIX_C_SOURCE 200809L
#include "life.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(FILE *out) {
    fprintf(out,
      "Usage: serial [--rows H --cols W | --input FILE.pbm] [options]\n"
      "  --steps N          generations, including zero (default 100)\n"
      "  --seed N           unsigned 64-bit seed (default 1)\n"
      "  --density-ppm N    initial density in [0,1000000] (default 500000)\n"
      "  --output FILE.pbm  optional complete final grid (P1 PBM)\n"
      "  --help             show this message\n"
      "Defaults: 64 rows, 64 columns; periodic B3/S23.\n"
      "stdout: one JSON timing/diagnostic record; errors: stderr/nonzero.\n");
}

static int unsigned_value(const char *s, uint64_t *value) {
    if (!s[0]) return -1;
    for (const char *p = s; *p; ++p) if (*p < '0' || *p > '9') return -1;
    errno = 0;
    char *end;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end || n > UINT64_MAX) return -1;
    *value = (uint64_t)n;
    return 0;
}

static double now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) {
        perror("clock_gettime"); exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    uint64_t rows = 64, cols = 64, steps = 100, seed = 1, density = 500000;
    unsigned seen = 0;
    const char *input = NULL, *output = NULL;
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (!strcmp(key, "--help")) { usage(stdout); return EXIT_SUCCESS; }
        unsigned bit = 0;
        uint64_t *dest = NULL;
        if (!strcmp(key,"--rows")) { bit=1; dest=&rows; }
        else if (!strcmp(key,"--cols")) { bit=2; dest=&cols; }
        else if (!strcmp(key,"--steps")) { bit=4; dest=&steps; }
        else if (!strcmp(key,"--seed")) { bit=8; dest=&seed; }
        else if (!strcmp(key,"--density-ppm")) { bit=16; dest=&density; }
        else if (!strcmp(key,"--input")) bit=32;
        else if (!strcmp(key,"--output")) bit=64;
        else { fprintf(stderr,"unknown option: %s\n",key); return EXIT_FAILURE; }
        if ((seen & bit) || i+1 == argc) {
            fprintf(stderr,"duplicate option or missing value: %s\n",key);
            return EXIT_FAILURE;
        }
        seen |= bit;
        const char *value = argv[++i];
        if (dest) {
            if (unsigned_value(value,dest)) {
                fprintf(stderr,"invalid unsigned integer for %s\n",key);
                return EXIT_FAILURE;
            }
        } else if (bit == 32) input = value;
        else output = value;
    }
    if (!rows || !cols || rows > SIZE_MAX || cols > SIZE_MAX || density > 1000000) {
        fprintf(stderr,"invalid dimensions or density\n"); return EXIT_FAILURE;
    }
    if (input && (seen & (8u|16u))) {
        fprintf(stderr,"--seed/--density-ppm cannot be combined with --input\n");
        return EXIT_FAILURE;
    }
    LifeGrid current = {0}, next = {0};
    const double total_start = now();
    if (input) {
        char error[160];
        if (life_read_pbm(input,&current,error,sizeof error)) {
            fprintf(stderr,"input error: %s\n",error); return EXIT_FAILURE;
        }
        if (((seen&1) && rows != current.rows) ||
            ((seen&2) && cols != current.cols)) {
            fprintf(stderr,"explicit dimensions disagree with input\n");
            life_grid_destroy(&current); return EXIT_FAILURE;
        }
    } else {
        if (life_grid_create(&current,(size_t)rows,(size_t)cols)) {
            fprintf(stderr,"invalid dimensions or allocation failed\n"); return EXIT_FAILURE;
        }
        life_fill_random(&current,seed,(uint32_t)density,0);
    }
    if (life_grid_create(&next,current.rows,current.cols)) {
        fprintf(stderr,"allocation failed\n"); life_grid_destroy(&current);
        return EXIT_FAILURE;
    }
    const double kernel_start = now();
    for (uint64_t step = 0; step < steps; ++step) {
        life_wrap_periodic(&current);
        life_step(&current,&next);
        life_swap(&current,&next);
    }
    const double kernel_seconds = now() - kernel_start;
    const uint64_t live = life_live_count(&current);
    const uint64_t hash = life_checksum(&current);
    const double total_seconds = now() - total_start;
    if (output && life_write_pbm(output,&current)) {
        fprintf(stderr,"cannot write complete output grid\n");
        life_grid_destroy(&current); life_grid_destroy(&next); return EXIT_FAILURE;
    }
    printf("{\"interface_version\":\"%s\",\"variant\":\"serial\","
           "\"rows\":%zu,\"cols\":%zu,\"steps\":%" PRIu64 ","
           "\"init\":\"%s\",\"seed\":",LIFE_INTERFACE_VERSION,
           current.rows,current.cols,steps,input?"pbm":"random");
    if (input) printf("null"); else printf("%" PRIu64,seed);
    printf(",\"density_ppm\":");
    if (input) printf("null"); else printf("%" PRIu64,density);
    printf(",\"ranks\":1,\"threads\":1,\"live_cells\":%" PRIu64 ","
           "\"checksum_fnv1a64\":\"%016" PRIx64 "\","
           "\"kernel_seconds\":%.9f,\"total_seconds\":%.9f}\n",
           live,hash,kernel_seconds,total_seconds);
    life_grid_destroy(&current); life_grid_destroy(&next);
    return EXIT_SUCCESS;
}
