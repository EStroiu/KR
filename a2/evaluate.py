#!/usr/bin/env python3
"""
Evaluate the SAT solver on randomly generated Non-consecutive Sudoku puzzles.

This script generates puzzles compatible with the encoder in `encoder.py`,
encodes them to CNF, runs `solver.solve_cnf`, and collects metrics and plots.

Compared with the original script, this version:
 - can generate solved grids locally using randomized backtracking (fast for N=4,9)
 - can create guaranteed-SAT puzzles by masking a solved grid
 - can create guaranteed-UNSAT puzzles by injecting a conflict and forcing those conflicting
     cells to remain visible after masking (guarantees UNSAT labeling)
 - exposes --unsat-proportion to control how many instances are provably unsat
 - records per-instance ground-truth labels (expected SAT/UNSAT) when known and
     prints labeled accuracy, a confusion matrix, and performance summaries split by
     predicted and expected status

Outputs:
  - CSV with one row per instance (outdir/metrics.csv)
  - Plots (PNG): time_by_size.png, status_counts.png, time_vs_dp_calls.png
  - Temporary puzzles stored in outdir/tmp

Usage examples:
  python3 a2/evaluate.py --sizes 4 9 --instances-per-size 20 --clue-density 0.5 \
      --suite-mode sat --unsat-proportion 0.2 --gen-timeout 10 --timeout 5 --outdir a2/outputs

Note:
 - This script assumes `encoder.py` and your `solver` module (imported as solver_mod)
   are present and compatible with the rest of the code.
 - Generation does not use your solver (avoids gen-timeout failures).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set

# Local modules (must exist in the same package)
import encoder
import solver as solver_mod


@dataclass
class InstanceResult:
    size: int
    clue_density: float
    n_clues: int
    num_vars: int
    n_clauses: int
    status: str
    wall_time_s: float
    dp_calls: int
    unit_clause_checks: int
    pure_literal_checks: int
    remove_literal_calls: int
    select_literal_calls: int
    model_len: int = 0
    model_cnf_valid: Optional[bool] = None
    model_semantic_valid: Optional[bool] = None
    error: str = ""
    expected_status: Optional[str] = None  # 'SAT', 'UNSAT', or None/"UNKNOWN"
    is_correct: Optional[bool] = None     # True/False if expected known, else None


# ---------------------------
# Local generator utilities
# ---------------------------
def generate_solved_grid(
    n: int,
    rng: Optional[random.Random] = None,
    non_consecutive: bool = False,
    max_tries: int = 200000,
    restarts: int = 50,
) -> Optional[List[List[int]]]:
    """MRV + backtracking generator with randomized restarts.

    - n must be a perfect square.
    - non_consecutive enforces orthogonal neighbors not differ by 1 (checked
      only against already placed neighbors).
    - max_tries limits total backtrack steps per attempt.
    - restarts controls how many independent attempts we make (with different RNG state).
    """
    if rng is None:
        rng = random.Random()
    b = int(math.isqrt(n))
    if b * b != n:
        raise ValueError("N must be a perfect square")

    digits = list(range(1, n + 1))

    def one_attempt(seed_offset: int = 0) -> Optional[List[List[int]]]:
        # local RNG so restarts differ
        local_rng = random.Random(rng.randint(0, 2**31 - 1) ^ seed_offset)
        grid = [[0] * n for _ in range(n)]
        rows = [set() for _ in range(n)]
        cols = [set() for _ in range(n)]
        boxes = [[set() for _ in range(b)] for _ in range(b)]
        tries = 0

        def candidates_for(r: int, c: int) -> List[int]:
            can = []
            for v in digits:
                if v in rows[r] or v in cols[c] or v in boxes[r // b][c // b]:
                    continue
                if non_consecutive:
                    # check only already-placed orthogonal neighbours (so we don't overconstrain)
                    if r > 0 and grid[r - 1][c] != 0 and abs(grid[r - 1][c] - v) == 1:
                        continue
                    if r + 1 < n and grid[r + 1][c] != 0 and abs(grid[r + 1][c] - v) == 1:
                        continue
                    if c > 0 and grid[r][c - 1] != 0 and abs(grid[r][c - 1] - v) == 1:
                        continue
                    if c + 1 < n and grid[r][c + 1] != 0 and abs(grid[r][c + 1] - v) == 1:
                        continue
                can.append(v)
            local_rng.shuffle(can)
            return can

        def find_mrv_cell() -> Optional[Tuple[int, int, List[int]]]:
            # return (r, c, candidates) for empty cell with smallest candidate set
            best = None
            best_len = None
            for r in range(n):
                for c in range(n):
                    if grid[r][c] != 0:
                        continue
                    cand = candidates_for(r, c)
                    if not cand:
                        return (r, c, [])  # immediate dead end
                    if best is None or len(cand) < best_len:
                        best = (r, c, cand)
                        best_len = len(cand)
                        # MRV early exit if singleton
                        if best_len == 1:
                            return best
            return best

        def backtrack() -> bool:
            nonlocal tries
            # check termination
            # find an empty cell with MRV
            cell = find_mrv_cell()
            if cell is None:
                return True  # filled everything
            r, c, cand = cell
            if len(cand) == 0:
                return False
            tries += 1
            if tries > max_tries:
                return False
            for v in cand:
                # place
                grid[r][c] = v
                rows[r].add(v); cols[c].add(v); boxes[r // b][c // b].add(v)
                if backtrack():
                    return True
                # undo
                grid[r][c] = 0
                rows[r].remove(v); cols[c].remove(v); boxes[r // b][c // b].remove(v)
            return False

        ok = backtrack()
        return grid if ok else None

    # Try several independent attempts (randomized)
    for attempt in range(restarts):
        res = one_attempt(attempt)
        if res is not None:
            return res
    return None



def mask_grid(grid: List[List[int]], density: float, rng: random.Random = None) -> List[List[int]]:
    """Mask a solved grid according to density -> returns a puzzle with zeros where masked."""
    if rng is None:
        rng = random.Random()
    n = len(grid)
    out = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if rng.random() < density:
                out[r][c] = grid[r][c]
            else:
                out[r][c] = 0
    return out


def make_unsat_from_solution(grid: List[List[int]], rng: random.Random = None) -> List[List[int]]:
    """Turn a solved grid into an UNSAT puzzle by introducing an obvious conflict.
    Example: pick a cell and replace its value with a value already present in the same row.
    """
    if rng is None:
        rng = random.Random()
    n = len(grid)
    new_grid = [row[:] for row in grid]
    # pick a cell at random
    r = rng.randrange(n)
    c = rng.randrange(n)
    # pick an existing digit from the same row (ensures a row conflict)
    row_choices = [new_grid[r][j] for j in range(n) if j != c]
    if not row_choices:
        # fallback: use column choices
        col_choices = [new_grid[i][c] for i in range(n) if i != r]
        if not col_choices:
            # give up and just flip to a different digit (may or may not create conflict)
            new_grid[r][c] = ((new_grid[r][c] % n) + 1)
            return new_grid
        conflict_digit = rng.choice(col_choices)
    else:
        conflict_digit = rng.choice(row_choices)
    new_grid[r][c] = conflict_digit
    return new_grid


def mask_grid_with_forced(
    grid: List[List[int]], density: float, forced_cells: Iterable[Tuple[int, int]], rng: Optional[random.Random] = None
) -> List[List[int]]:
    """Mask a solved grid but always reveal the given forced_cells as clues.

    Ensures contradictions we intend to expose remain in the puzzle after masking.
    """
    if rng is None:
        rng = random.Random()
    forced: Set[Tuple[int, int]] = set(forced_cells)
    n = len(grid)
    out = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if (r, c) in forced:
                out[r][c] = grid[r][c]
            else:
                out[r][c] = grid[r][c] if rng.random() < density else 0
    return out


def make_unsat_with_forced_conflict(
    grid: List[List[int]], rng: Optional[random.Random] = None
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Create an UNSAT-inducing modification and return conflict cells to force as clues.

    We pick a row r and two distinct columns c1, c2 and set new_grid[r][c1] = new_grid[r][c2],
    creating a duplicate in row r. Returning both (r,c1) and (r,c2) allows masking to keep
    the contradiction visible, guaranteeing UNSAT.
    """
    if rng is None:
        rng = random.Random()
    n = len(grid)
    new_grid = [row[:] for row in grid]
    r = rng.randrange(n)
    c1, c2 = rng.sample(range(n), 2)
    new_grid[r][c1] = new_grid[r][c2]
    forced = [(r, c1), (r, c2)]
    return new_grid, forced


# ---------------------------
# Existing helper functions
# ---------------------------
def _var_index(r: int, c: int, v: int, N: int) -> int:
    return r * N * N + c * N + v


def verify_cnf_satisfied(clauses: Iterable[Iterable[int]], model: List[int]) -> bool:
    """Check that model satisfies all clauses. Model is a list of assigned literals."""
    assign = set(model)
    for cl in clauses:
        if not any(l in assign for l in cl):
            return False
    return True


def decode_model_to_grid(model: List[int], N: int) -> Optional[List[List[int]]]:
    """Decode positive literals in the model to an N x N grid. Return None if ambiguous."""
    grid = [[0 for _ in range(N)] for _ in range(N)]
    chosen: List[List[List[int]]] = [[[] for _ in range(N)] for _ in range(N)]
    for lit in model:
        if lit <= 0:
            continue
        v_idx = lit - 1
        r = v_idx // (N * N)
        rem = v_idx % (N * N)
        c = rem // N
        v = rem % N + 1
        if 0 <= r < N and 0 <= c < N and 1 <= v <= N:
            chosen[r][c].append(v)

    for r in range(N):
        for c in range(N):
            if len(chosen[r][c]) == 1:
                grid[r][c] = chosen[r][c][0]
            elif len(chosen[r][c]) == 0:
                return None
            else:
                return None
    return grid


def verify_grid_semantics(grid: List[List[int]], clues: List[List[int]]) -> bool:
    """Verify Sudoku+non-consecutive constraints and clues on a filled grid."""
    N = len(grid)
    B = int(math.isqrt(N))
    if B * B != N:
        return False

    # All cells in 1..N
    for r in range(N):
        for c in range(N):
            v = grid[r][c]
            if not (1 <= v <= N):
                return False

    # Clues respected
    for r in range(N):
        for c in range(N):
            if clues[r][c] != 0 and grid[r][c] != clues[r][c]:
                return False

    # Row uniqueness
    for r in range(N):
        if len(set(grid[r])) != N:
            return False

    # Column uniqueness
    for c in range(N):
        col = [grid[r][c] for r in range(N)]
        if len(set(col)) != N:
            return False

    # Box uniqueness
    for br in range(0, N, B):
        for bc in range(0, N, B):
            vals = []
            for dr in range(B):
                for dc in range(B):
                    vals.append(grid[br + dr][bc + dc])
            if len(set(vals)) != N:
                return False

    # Non-consecutive (orthogonal neighbors not differing by 1)
    for r in range(N):
        for c in range(N):
            v = grid[r][c]
            if r + 1 < N and abs(v - grid[r + 1][c]) == 1:
                return False
            if c + 1 < N and abs(v - grid[r][c + 1]) == 1:
                return False

    return True


def clues_have_semantic_conflict(clues: List[List[int]]) -> bool:
    """Check if the given clues alone already violate Sudoku+non-consecutive rules.

    If True, the puzzle is guaranteed UNSAT; otherwise, it may still be SAT or UNSAT.
    """
    N = len(clues)
    B = int(math.isqrt(N))
    if B * B != N:
        return True

    # Values in domain or zero
    for r in range(N):
        for c in range(N):
            v = clues[r][c]
            if v != 0 and not (1 <= v <= N):
                return True

    # Row duplicates among non-zero clues
    for r in range(N):
        seen = set()
        for c in range(N):
            v = clues[r][c]
            if v == 0:
                continue
            if v in seen:
                return True
            seen.add(v)

    # Column duplicates among non-zero clues
    for c in range(N):
        seen = set()
        for r in range(N):
            v = clues[r][c]
            if v == 0:
                continue
            if v in seen:
                return True
            seen.add(v)

    # Box duplicates among non-zero clues
    for br in range(0, N, B):
        for bc in range(0, N, B):
            seen = set()
            for dr in range(B):
                for dc in range(B):
                    v = clues[br + dr][bc + dc]
                    if v == 0:
                        continue
                    if v in seen:
                        return True
                    seen.add(v)

    # Non-consecutive violations where both neighbours are clues
    for r in range(N):
        for c in range(N):
            v = clues[r][c]
            if v == 0:
                continue
            if r + 1 < N and clues[r + 1][c] != 0 and abs(v - clues[r + 1][c]) == 1:
                return True
            if c + 1 < N and clues[r][c + 1] != 0 and abs(v - clues[r][c + 1]) == 1:
                return True
    return False


# ---------------------------
# Instrumentation wrappers for the solver
# ---------------------------
def _make_instrumentation() -> Tuple[Dict[str, int], Dict[str, Callable], Dict[str, Callable]]:
    counters = {
        "dp_calls": 0,
        "unit_clause_checks": 0,
        "pure_literal_checks": 0,
        "remove_literal_calls": 0,
        "select_literal_calls": 0,
    }

    originals: Dict[str, Callable] = {}
    wrappers: Dict[str, Callable] = {}

    # Wrap dp
    if hasattr(solver_mod, "dp"):
        originals["dp"] = solver_mod.dp

        def dp_wrap(clauses: Iterable[Iterable[int]]):  # type: ignore
            counters["dp_calls"] += 1
            return originals["dp"](clauses)

        wrappers["dp"] = dp_wrap

    # Wrap has_unit_clause
    if hasattr(solver_mod, "has_unit_clause"):
        originals["has_unit_clause"] = solver_mod.has_unit_clause

        def has_unit_clause_wrap(clauses: Iterable[Iterable[int]]):  # type: ignore
            counters["unit_clause_checks"] += 1
            return originals["has_unit_clause"](clauses)

        wrappers["has_unit_clause"] = has_unit_clause_wrap

    # Wrap has_literal (pure literal)
    if hasattr(solver_mod, "has_literal"):
        originals["has_literal"] = solver_mod.has_literal

        def has_literal_wrap(clauses: Iterable[Iterable[int]]):  # type: ignore
            counters["pure_literal_checks"] += 1
            return originals["has_literal"](clauses)

        wrappers["has_literal"] = has_literal_wrap

    # Wrap remove_literal
    if hasattr(solver_mod, "remove_literal"):
        originals["remove_literal"] = solver_mod.remove_literal

        def remove_literal_wrap(clauses: Iterable[Iterable[int]], literal: int):  # type: ignore
            counters["remove_literal_calls"] += 1
            return originals["remove_literal"](clauses, literal)

        wrappers["remove_literal"] = remove_literal_wrap

    # Wrap select_literal
    if hasattr(solver_mod, "select_literal"):
        originals["select_literal"] = solver_mod.select_literal

        def select_literal_wrap(clauses: Iterable[Iterable[int]]):  # type: ignore
            counters["select_literal_calls"] += 1
            return originals["select_literal"](clauses)

        wrappers["select_literal"] = select_literal_wrap

    return counters, originals, wrappers


def _apply_wrappers(wrappers: Dict[str, Callable]) -> None:
    for name, func in wrappers.items():
        setattr(solver_mod, name, func)


def _restore_originals(originals: Dict[str, Callable]) -> None:
    for name, func in originals.items():
        setattr(solver_mod, name, func)


# ---------------------------
# Timeout helpers
# ---------------------------
class Timeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise Timeout()


# ---------------------------
# Run / evaluation logic
# ---------------------------
def generate_random_puzzle(n: int, density: float, rng: random.Random) -> List[List[int]]:
    """Generate an N x N puzzle with values in 0..N (0 means empty).
    Each cell is a clue with probability `density`; clues are uniform in 1..N.
    """
    b = int(math.isqrt(n))
    if b * b != n:
        raise ValueError(f"N must be a perfect square; got N={n}")
    grid: List[List[int]] = []
    for _r in range(n):
        row: List[int] = []
        for _c in range(n):
            if rng.random() < density:
                row.append(rng.randint(1, n))
            else:
                row.append(0)
        grid.append(row)
    return grid


def write_puzzle_tmp(grid: List[List[int]], tmp_dir: str) -> str:
    """Write puzzle to a temporary file and return its path."""
    os.makedirs(tmp_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="nc_sudoku_", suffix=".txt", dir=tmp_dir, text=True)
    with os.fdopen(fd, "w") as f:
        for row in grid:
            f.write(" ".join(str(x) for x in row) + "\n")
    return path


def run_one_instance(
    grid: List[List[int]],
    timeout_s: Optional[float],
    tmp_dir: str,
    expected_status: Optional[str] = None,
) -> InstanceResult:
    # Instrument
    counters, originals, wrappers = _make_instrumentation()
    _apply_wrappers(wrappers)

    # Prepare CNF
    puzzle_path = write_puzzle_tmp(grid, tmp_dir)
    try:
        clauses, num_vars = encoder.to_cnf(puzzle_path)
    except Exception as e:  # encoding error
        _restore_originals(originals)
        n = len(grid)
        n_clues = sum(1 for r in grid for v in r if v != 0)
        return InstanceResult(
            size=n,
            clue_density=(n_clues / (n * n)),
            n_clues=n_clues,
            num_vars=0,
            n_clauses=0,
            status="ERROR",
            wall_time_s=0.0,
            dp_calls=counters["dp_calls"],
            unit_clause_checks=counters["unit_clause_checks"],
            pure_literal_checks=counters["pure_literal_checks"],
            remove_literal_calls=counters["remove_literal_calls"],
            select_literal_calls=counters["select_literal_calls"],
            error=f"encode: {type(e).__name__}: {e}",
            expected_status=expected_status,
            is_correct=None if expected_status is None else False,
        )

    n = len(grid)
    n_clues = sum(1 for r in grid for v in r if v != 0)
    # Materialize clauses once if needed to avoid consuming generators multiple times
    if not isinstance(clauses, list):
        clauses = list(clauses)
    n_clauses = len(clauses)

    # Solve with timeout
    started = time.perf_counter()
    old_handler = None
    timed_out = False
    try:
        if timeout_s and timeout_s > 0 and hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max(1, int(timeout_s)))
        status, model = solver_mod.solve_cnf(clauses, num_vars)
        wall_time = time.perf_counter() - started
        result_status = status
    except Timeout:
        wall_time = time.perf_counter() - started
        result_status = "TIMEOUT"
        timed_out = True
    except Exception as e:
        wall_time = time.perf_counter() - started
        result_status = f"ERROR"
        err = f"solve: {type(e).__name__}: {e}"
    finally:
        if hasattr(signal, "SIGALRM") and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        _restore_originals(originals)

    # Build result
    error_msg = ""
    if result_status == "ERROR":
        error_msg = err  # type: ignore[name-defined]

    model_len = 0
    model_cnf_valid: Optional[bool] = None
    model_semantic_valid: Optional[bool] = None
    if result_status == "SAT":
        try:
            # CNF validity
            model = model or []
            model_len = len(model)
            cnf_ok = verify_cnf_satisfied(clauses, model)
            model_cnf_valid = cnf_ok
            # Semantic validity (decode and check constraints)
            grid_dec = decode_model_to_grid(model, n)
            if grid_dec is not None:
                sem_ok = verify_grid_semantics(grid_dec, grid)
                model_semantic_valid = sem_ok
            else:
                model_semantic_valid = False
        except Exception:
            model_cnf_valid = False
            model_semantic_valid = False

    # Compute correctness if we have an expected status label
    is_correct: Optional[bool] = None
    if expected_status in ("SAT", "UNSAT"):
        is_correct = (result_status == expected_status)

    return InstanceResult(
        size=n,
        clue_density=(n_clues / (n * n)),
        n_clues=n_clues,
        num_vars=num_vars,
        n_clauses=n_clauses,
        status=result_status,
        wall_time_s=wall_time,
        dp_calls=counters["dp_calls"],
        unit_clause_checks=counters["unit_clause_checks"],
        pure_literal_checks=counters["pure_literal_checks"],
        remove_literal_calls=counters["remove_literal_calls"],
        select_literal_calls=counters["select_literal_calls"],
        model_len=model_len,
        model_cnf_valid=model_cnf_valid,
        model_semantic_valid=model_semantic_valid,
        error=error_msg,
        expected_status=expected_status,
        is_correct=is_correct,
    )


def save_csv(rows: List[InstanceResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(rows[0]).keys()) if rows else [
                "size", "clue_density", "n_clues", "num_vars", "n_clauses",
                "status", "wall_time_s", "dp_calls", "unit_clause_checks",
                "pure_literal_checks", "remove_literal_calls", "select_literal_calls",
                "model_len", "model_cnf_valid", "model_semantic_valid", "error",
                "expected_status", "is_correct"
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def try_make_plots(rows: List[InstanceResult], outdir: str) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    os.makedirs(outdir, exist_ok=True)
    paths: List[str] = []

    # Time distributions by N (boxplot)
    by_n: Dict[int, List[float]] = {}
    for r in rows:
        if r.status in ("SAT", "UNSAT"):
            by_n.setdefault(r.size, []).append(r.wall_time_s)
    if by_n:
        labels = sorted(by_n.keys())
        data = [by_n[n] for n in labels]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=[str(l) for l in labels], showfliers=False)
        plt.title("Solve time by size (successful runs)")
        plt.xlabel("N")
        plt.ylabel("Seconds")
        p = os.path.join(outdir, "time_by_size.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)

    # Status counts by N (bar chart)
    statuses = ["SAT", "UNSAT", "TIMEOUT", "ERROR"]
    counts: Dict[int, Dict[str, int]] = {}
    for r in rows:
        d = counts.setdefault(r.size, {s: 0 for s in statuses})
        d[r.status] = d.get(r.status, 0) + 1
    if counts:
        ns = sorted(counts.keys())
        width = 0.2
        x = range(len(ns))
        plt.figure(figsize=(7, 4))
        for i, s in enumerate(statuses):
            plt.bar([xi + i * width for xi in x], [counts[n].get(s, 0) for n in ns], width=width, label=s)
        plt.xticks([xi + 1.5 * width for xi in x], [str(n) for n in ns])
        plt.xlabel("N")
        plt.ylabel("Count")
        plt.title("Status counts by size")
        plt.legend()
        p = os.path.join(outdir, "status_counts.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)

    # Scatter: time vs dp_calls (colored by N)
    xs = [r.dp_calls for r in rows if r.status in ("SAT", "UNSAT")]
    ys = [r.wall_time_s for r in rows if r.status in ("SAT", "UNSAT")]
    cs = [r.size for r in rows if r.status in ("SAT", "UNSAT")]
    if xs:
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(xs, ys, c=cs, cmap="viridis", alpha=0.7)
        plt.xlabel("dp_calls (instrumented)")
        plt.ylabel("Seconds")
        plt.title("Time vs dp_calls")
        cbar = plt.colorbar(sc)
        cbar.set_label("N")
        p = os.path.join(outdir, "time_vs_dp_calls.png")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths.append(p)

    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate the Non-consecutive Sudoku SAT solver on random instances.")
    ap.add_argument("--sizes", nargs="*", type=int, default=[4, 9], help="List of N (perfect squares) to test")
    ap.add_argument("--instances-per-size", type=int, default=10, help="How many instances per N")
    ap.add_argument("--clue-density", type=float, default=0.2, help="Probability a cell is a clue (0..1)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--timeout", type=float, default=5.0, help="Per-instance timeout in seconds (Linux only)")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"), help="Output dir for CSV and plots")
    ap.add_argument("--no-plots", action="store_true", help="Skip plotting even if matplotlib is available")
    ap.add_argument("--suite-mode", choices=["random", "sat"], default="random", help="random: independently random clues; sat: clues masked from a solved grid to ensure SAT")
    ap.add_argument("--gen-timeout", type=float, default=20.0, help="(Legacy) Timeout for generating a solved grid when suite-mode=sat; generator used instead of solver")
    ap.add_argument("--unsat-proportion", type=float, default=0.0, help="Proportion (0..1) of instances that should be guaranteed-UNSAT (only relevant for suite-mode=sat)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    results: List[InstanceResult] = []
    tmp_dir = os.path.join(args.outdir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for n in args.sizes:
        # quick validation
        b = int(math.isqrt(n))
        if b * b != n:
            print(f"[skip] N={n} is not a perfect square; skipping.")
            continue

        base_solution: Optional[List[List[int]]] = None
        if args.suite_mode == "sat":
            # Use local generator to obtain a solved grid (fast & robust)
            try:
                start_g = time.perf_counter()
                base_solution = generate_solved_grid(n, rng, non_consecutive=True, max_tries=int(max(10000, args.gen_timeout * 1000)))
                if base_solution is None:
                    print(f"N={n}: failed to produce solved grid with generator; falling back to random suite.")
                else:
                    print(f"N={n}: generated base solved grid locally in {time.perf_counter() - start_g:.3f}s for SAT suite.")
            except Exception as e:
                print(f"N={n}: error generating base solution: {e}; falling back to random suite.")

        for i in range(args.instances_per_size):
            if base_solution is not None:
                # Decide whether this instance should be guaranteed UNSAT
                make_unsat = rng.random() < args.unsat_proportion
                if make_unsat:
                    # Create a visible contradiction and force those clues to remain
                    conflict_solution, forced = make_unsat_with_forced_conflict(base_solution, rng=rng)
                    grid = mask_grid_with_forced(conflict_solution, args.clue_density, forced, rng=rng)
                    expected = "UNSAT"
                else:
                    grid = mask_grid(base_solution, args.clue_density, rng=rng)
                    # Masked from a valid solved grid -> definitely SAT
                    expected = "SAT"
            else:
                grid = generate_random_puzzle(n, args.clue_density, rng)
                # For random puzzles we can still detect hard conflicts in clues (optional)
                expected = "UNSAT" if clues_have_semantic_conflict(grid) else None

            res = run_one_instance(grid, args.timeout, tmp_dir, expected_status=expected)
            results.append(res)
            print(f"N={n} [{i+1}/{args.instances_per_size}] -> {res.status} in {res.wall_time_s:.3f}s, clauses={res.n_clauses}")

    # Save CSV
    csv_path = os.path.join(args.outdir, "metrics.csv")
    save_csv(results, csv_path)
    print(f"Saved metrics CSV -> {csv_path}")

    # Plots
    if not args.no_plots:
        plot_paths = try_make_plots(results, args.outdir)
        if plot_paths:
            for p in plot_paths:
                print(f"Saved plot -> {p}")
        else:
            print("matplotlib not available; skipped plots.")

    # Accuracy and performance summaries
    labeled = [r for r in results if r.expected_status in ("SAT", "UNSAT")]
    if labeled:
        correct = sum(1 for r in labeled if r.status == r.expected_status)
        acc = correct / len(labeled)
        print(f"Labeled accuracy: {correct}/{len(labeled)} = {acc:.3f}")

        # Confusion matrix (expected vs predicted)
        conf: Dict[str, Dict[str, int]] = {}
        for r in labeled:
            exp = r.expected_status or "?"
            conf.setdefault(exp, {})
            conf[exp][r.status] = conf[exp].get(r.status, 0) + 1
        for exp in ("SAT", "UNSAT"):
            row = conf.get(exp, {})
            print(f"Expected {exp}: predicted SAT={row.get('SAT',0)}, UNSAT={row.get('UNSAT',0)}, TIMEOUT={row.get('TIMEOUT',0)}, ERROR={row.get('ERROR',0)}")

        # Performance by predicted status
        for s in ("SAT", "UNSAT"):
            times = [r.wall_time_s for r in results if r.status == s]
            if times:
                print(f"Mean time (predicted {s}): {sum(times)/len(times):.3f}s over {len(times)} instances")

        # Performance by expected status (labeled only)
        for s in ("SAT", "UNSAT"):
            times = [r.wall_time_s for r in labeled if r.expected_status == s and r.status in ("SAT","UNSAT")]
            if times:
                print(f"Mean time (expected {s}): {sum(times)/len(times):.3f}s over {len(times)} instances")


if __name__ == "__main__":
    main()
