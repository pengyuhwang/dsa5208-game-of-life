"""Deterministic test specification. Known answers are not oracle-generated."""
import random


def case(name, rows, cols, live, steps, known=None):
    return dict(name=name, rows=rows, cols=cols, live=set(live), steps=steps,
                known=None if known is None else set(known))


def named_cases():
    block = {(2, 3), (2, 4), (3, 3), (3, 4)}
    vertical = {(2, 4), (3, 4), (4, 4)}
    horizontal = {(3, 3), (3, 4), (3, 5)}
    corners = {(0, 0), (0, 8), (6, 0), (6, 8)}
    glider = {(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)}
    wrap_glider = {((r + 8) % 12, (c + 9) % 13) for r, c in glider}
    return [
        case("empty", 7, 9, set(), 17, set()),
        case("single_dies", 7, 9, {(3, 4)}, 1, set()),
        case("block_stable", 8, 9, block, 20, block),
        case("blinker_odd", 7, 9, vertical, 1, horizontal),
        case("blinker_even", 7, 9, vertical, 20, vertical),
        case("corner_block", 7, 9, corners, 20, corners),
        case("edge_blinker", 7, 9, {(0, 8), (0, 0), (0, 1)}, 1,
             {(6, 0), (0, 0), (1, 0)}),
        case("glider_four", 12, 13, glider, 4, {(r+1, c+1) for r, c in glider}),
        case("glider_wrap", 12, 13, wrap_glider, 12,
             {((r+3) % 12, (c+3) % 13) for r, c in wrap_glider}),
        case("one_cell_alive", 1, 1, {(0, 0)}, 1, set()),
        case("all_alive", 5, 7, {(r,c) for r in range(5) for c in range(7)}, 1, set()),
        case("zero_steps", 5, 7, {(0, 0), (1, 3), (4, 6)}, 0,
             {(0, 0), (1, 3), (4, 6)}),
        case("local_rows_one", 4, 7, {(0, 0), (1, 0), (2, 0), (3, 6)}, 9),
        case("nondivisible_rows", 5, 7, {(0, 0), (1, 2), (3, 5), (4, 6)}, 9),
    ]


def all_cases():
    yield from named_cases()
    for state in range(512):
        yield case("exhaustive_3x3_{:03d}".format(state), 3, 3,
                   {(i//3, i%3) for i in range(9) if state & (1 << i)}, 1)
    rng = random.Random(20261026)
    shapes = [(1,1),(1,2),(2,1),(1,9),(9,1),(2,2),(2,7),(7,2),
              (3,5),(4,7),(5,7),(7,11),(17,19),(31,37)]
    for rows, cols in shapes:
        for repetition in range(6):
            p = [0.05, 0.2, 0.5, 0.8, 0.95, 0.5][repetition]
            live = {(r,c) for r in range(rows) for c in range(cols) if rng.random() < p}
            yield case("random_{}x{}_{}".format(rows,cols,repetition), rows,cols,
                       live, [0,1,2,3,10,25][repetition])
