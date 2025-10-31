#!/usr/bin/env python3
"""
SAT Assignment Part 1 - Non-consecutive Sudoku Encoder (Puzzle -> CNF)

Do NOT modify this file - instead, implement your function in encoder.py

Usage:
  python main.py --in <puzzle.txt> --out <instance.cnf>
"""

import argparse
import sys
from encoder import to_cnf  #implement


def write_dimacs(target, num_vars: int, clauses) -> None:
    """Write DIMACS CNF to a file path or file-like (stdout)."""
    close = False
    if isinstance(target, str):
        f = open(target, "w")
        close = True
    else:
        f = target
    try:
        clauses = list(clauses)
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(str(l) for l in cl) + " 0\n")
    finally:
        if close:
            f.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Path to puzzle .txt")
    p.add_argument("--out", dest="out", default=None, help="Path to write DIMACS CNF (stdout if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    clauses, num_vars = to_cnf(args.inp)
    if args.out:
        write_dimacs(args.out, num_vars, clauses)
    else:
        write_dimacs(sys.stdout, num_vars, clauses)


if __name__ == "__main__":
    main()
