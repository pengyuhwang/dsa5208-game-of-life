CC = cc
PYTHON ?= python3
CPPFLAGS ?= -Iinclude
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
LDFLAGS ?=
SOURCES = serial.c src/life.c

.PHONY: all test sanitize figures benchmark clean
all: build/serial

build:
	mkdir -p build

build/serial: $(SOURCES) include/life.h | build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SOURCES) $(LDFLAGS) -o $@

build/serial_sanitize: $(SOURCES) include/life.h | build
	$(CC) $(CPPFLAGS) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address,undefined -fno-omit-frame-pointer $(SOURCES) -o $@

test: build/serial
	$(PYTHON) tools/verify.py --binary ./build/serial --output build/verification
	$(PYTHON) -m unittest discover -s tests -v

sanitize: build/serial_sanitize
	$(PYTHON) tools/verify.py --binary ./build/serial_sanitize --output results/sanitizer --skip-external

figures: build/serial
	$(PYTHON) tools/figures.py

benchmark: build/serial
	$(PYTHON) tools/benchmark.py --plan config/experiment_plan.csv --owner A --environment local --output results/local_baseline

clean:
	rm -f build/serial build/serial_sanitize build/test_core build/test_decomp
