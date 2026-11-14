"""Independent sparse-set Life oracle and strict, dependency-free PBM tools.

The C code gathers eight values using ghost rows. This reference instead
scatters neighbour contributions from live coordinates using a Counter.
"""
from collections import Counter
import hashlib

MASK = (1 << 64) - 1


def evolve(rows, cols, live, steps):
    live = set(live)
    for _ in range(steps):
        counts = Counter()
        for r, c in live:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        counts[((r + dr) % rows, (c + dc) % cols)] += 1
        live = {p for p, n in counts.items() if n == 3 or (n == 2 and p in live)}
    return live


def initial_cell(seed, index, density):
    z = (seed + 0x9E3779B97F4A7C15 * (index + 1)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z ^= z >> 31
    return int(z % 1000000 < density)


def random_grid(rows, cols, seed, density):
    return {(r, c) for r in range(rows) for c in range(cols)
            if initial_cell(seed, r * cols + c, density)}


def read_pbm(path):
    text = path.read_text(encoding="ascii")
    tokens = " ".join(line.split("#", 1)[0] for line in text.splitlines()).split()
    if len(tokens) < 3 or tokens[0] != "P1":
        raise ValueError("expected a P1 PBM")
    if not all(s.isdigit() for s in tokens[1:3]):
        raise ValueError("non-integer dimensions")
    cols, rows = map(int, tokens[1:3])
    if rows < 1 or cols < 1 or len(tokens) - 3 != rows * cols:
        raise ValueError("invalid dimensions/pixel count")
    pixels = tokens[3:]
    if any(s not in ("0", "1") for s in pixels):
        raise ValueError("non-binary pixel")
    return rows, cols, {(i // cols, i % cols) for i, s in enumerate(pixels) if s == "1"}


def write_pbm(path, rows, cols, live):
    live = set(live)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("P1\n# 1=alive; periodic B3/S23\n{} {}\n".format(cols, rows))
        for r in range(rows):
            f.write(" ".join("1" if (r, c) in live else "0" for c in range(cols)) + "\n")


def grid_bytes(rows, cols, live):
    return bytes(int((r, c) in live) for r in range(rows) for c in range(cols))


def grid_sha256(rows, cols, live):
    return hashlib.sha256(grid_bytes(rows, cols, live)).hexdigest()


def fnv1a64(rows, cols, live):
    h = 14695981039346656037
    for b in grid_bytes(rows, cols, live):
        h = ((h ^ b) * 1099511628211) & MASK
    return "{:016x}".format(h)
