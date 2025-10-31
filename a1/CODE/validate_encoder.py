#!/usr/bin/env python3
"""
Validator for Non-Consecutive Sudoku CNF encoder.

Sanity checks for a puzzle:
- num_vars == N^3
- clause count matches formula: 4*N^2*(1 + C(N,2)) + 4*N*(N-1)^2 + K
  where K is number of non-zero clues
- unit clause count == K and units correspond exactly to clue literals var(r,c,v)
- all literal ids are within [1..num_vars] in absolute value

Usage:
  python a1/CODE/validate_encoder.py --in "a1/EXAMPLE puzzles (input)/example_n9.txt"

Exits with code 0 on PASS, non-zero on FAIL.
"""

from __future__ import annotations
import argparse
import math
from typing import List, Tuple
from encoder import to_cnf


def read_puzzle(path: str) -> List[List[int]]:
    grid: List[List[int]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            grid.append([int(x) for x in line.split()])
    if not grid:
        raise ValueError("Empty puzzle file")
    n = len(grid)
    for row in grid:
        if len(row) != n:
            raise ValueError(f"Puzzle must be N x N; found row length {len(row)} for N={n}")
    b = int(math.isqrt(n))
    if b * b != n:
        raise ValueError(f"N must be a perfect square; got N={n}")
    for r in range(n):
        for c in range(n):
            v = grid[r][c]
            if v < 0 or v > n:
                raise ValueError(f"Cell ({r},{c}) has value {v} outside 0..{n}")
    return grid


def var_index(r: int, c: int, v: int, N: int) -> int:
    return r * N * N + c * N + v


def expected_counts(N: int, K: int) -> Tuple[int, int]:
    # num_vars = N^3
    num_vars = N ** 3
    # clauses total:
    # 4*N^2 Exact-One groups, each contributes (1 + C(N,2)) clauses
    exact_one_groups = 4 * (N ** 2)
    exact_one_clauses = exact_one_groups * (1 + (N * (N - 1)) // 2)
    # Non-consecutive: 2*N*(N-1) adjacency pairs, each contributes 2*(N-1) binary clauses
    nonconsec = 4 * N * (N - 1) * (N - 1)
    total_clauses = exact_one_clauses + nonconsec + K
    return num_vars, total_clauses


def validate(puzzle_path: str) -> int:
    # Load puzzle and encode
    grid = read_puzzle(puzzle_path)
    N = len(grid)
    K = sum(1 for r in range(N) for c in range(N) if grid[r][c] != 0)

    clauses, num_vars = to_cnf(puzzle_path)
    clauses = list(clauses)

    # 1) num_vars
    exp_vars, exp_clauses = expected_counts(N, K)
    ok_vars = (num_vars == exp_vars)

    # 2) clause count
    ok_clause_count = (len(clauses) == exp_clauses)

    # 3) units match clues exactly
    units = [cl for cl in clauses if len(cl) == 1]
    ok_unit_count = (len(units) == K)
    # create set of required unit literals
    required_units = set()
    for r in range(N):
        for c in range(N):
            v = grid[r][c]
            if v != 0:
                required_units.add(var_index(r, c, v, N))
    present_units = set(lit for [lit] in units)
    ok_unit_membership = (present_units == required_units)

    # 4) literal ranges valid
    ok_ranges = all(1 <= abs(lit) <= num_vars for cl in clauses for lit in cl)

    # Summarize
    all_ok = ok_vars and ok_clause_count and ok_unit_count and ok_unit_membership and ok_ranges

    print("Validator report:")
    print(f"  N={N}, clues K={K}")
    print(f"  num_vars: got {num_vars}, expected {exp_vars} -> {'OK' if ok_vars else 'FAIL'}")
    print(f"  clause count: got {len(clauses)}, expected {exp_clauses} -> {'OK' if ok_clause_count else 'FAIL'}")
    print(f"  unit clauses: got {len(units)}, expected {K} -> {'OK' if ok_unit_count else 'FAIL'}")
    print(f"  unit membership (matches clues): {'OK' if ok_unit_membership else 'FAIL'}")
    print(f"  literal ranges within [1..num_vars]: {'OK' if ok_ranges else 'FAIL'}")

    if all_ok:
        print("Result: PASS")
        return 0
    else:
        print("Result: FAIL")
        return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to puzzle .txt")
    args = ap.parse_args()
    exit(validate(args.inp))


if __name__ == "__main__":
    main()
