# MPI/Hybrid drivers for the shared Game of Life interface (version 1.0).
# The kernel and grid representation come from member A's project, so there is
# exactly one implementation of the update rule and of the PBM/JSON contract.
A_DIR   ?= ../Game_of_Life_A
MPICC   ?= mpicc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
HOSTFLAGS ?= -fopenmp   # NVHPC wants -mp=multicore here, it has no -fopenmp
CPPFLAGS += -I$(A_DIR)/include
LDFLAGS ?=
BUILD   := build
LIFE    := $(A_DIR)/src/life.c
LIFE_H  := $(A_DIR)/include/life.h

.PHONY: all test test-comm test-gpu bench gpu install-a clean

all: $(BUILD)/mpi $(BUILD)/hybrid

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/mpi: src/mpi.c $(LIFE) $(LIFE_H) | $(BUILD)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) src/mpi.c $(LIFE) -o $@ $(LDFLAGS)

$(BUILD)/hybrid: src/mpi.c $(LIFE) $(LIFE_H) | $(BUILD)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $(HOSTFLAGS) -DGOL_USE_OPENMP src/mpi.c $(LIFE) -o $@ $(LDFLAGS)

# OpenMP target offload (GPU). Needs a compiler whose OpenMP supports device
# offload. Vanda: `make gpu MPICC=mpicc GPUFLAGS=-mp=gpu` under
# `module load nvhpc-hpcx-cuda12/26.1`; GCC would need -foffload=nvptx-none.
# Deliberately outside "all": a toolchain without an offload target must still
# build the CPU variants.
GPUFLAGS ?= -fopenmp
gpu: $(BUILD)/gpu
$(BUILD)/gpu: src/mpi.c $(LIFE) $(LIFE_H) | $(BUILD)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $(GPUFLAGS) \
	  -DGOL_USE_OPENMP -DGOL_USE_OMP_TARGET src/mpi.c $(LIFE) -o $@ $(LDFLAGS)

# Place the two binaries where tools/verify.py expects them (project root/build).
install-a: all
	mkdir -p $(A_DIR)/build
	cp $(BUILD)/mpi $(BUILD)/hybrid $(A_DIR)/build/

test: install-a
	$(MAKE) -C $(A_DIR) build/serial
	cd $(A_DIR) && python3 tools/verify.py --require-all

# decomp x comm x ranks correctness matrix against A's serial kernel
test-comm: all
	$(MAKE) -C $(A_DIR) build/serial
	bash scripts/test_comm.sh

# GPU variant against A's serial kernel. Real device required unless
# GOL_ALLOW_HOST_TARGET=1 is set (logic-only run, not a GPU measurement).
test-gpu: $(BUILD)/gpu
	$(MAKE) -C $(A_DIR) build/serial
	bash scripts/test_gpu.sh

bench: all
	bash scripts/bench_atlas.sh

clean:
	rm -rf $(BUILD)
