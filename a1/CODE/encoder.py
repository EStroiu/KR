"""
SAT Assignment Part 1 - Non-consecutive Sudoku Encoder (Puzzle -> CNF)

THIS is the file to edit.

Implement: to_cnf(input_path) -> (clauses, num_vars)

You're required to use a variable mapping as follows:
    var(r,c,v) = r*N*N + c*N + v
where r,c are in range (0...N-1) and v in (1...N).

You must encode:
  (1) Exactly one value per cell
  (2) For each value v and each row r: exactly one column c has v
  (3) For each value v and each column c: exactly one row r has v
  (4) For each value v and each sqrt(N)×sqrt(N) box: exactly one cell has v
  (5) Non-consecutive: orthogonal neighbors cannot differ by 1
  (6) Clues: unit clauses for the given puzzle
"""


from typing import Tuple, Iterable, List
import math


def _read_puzzle(path: str) -> List[List[int]]:
  grid: List[List[int]] = []
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      row = [int(x) for x in line.split()]
      grid.append(row)
  if not grid:
    raise ValueError("Empty puzzle file")
  n = len(grid)
  for row in grid:
    if len(row) != n:
      raise ValueError(f"Puzzle must be N x N, got row of length {len(row)} for N={n}")
  # Validate perfect square
  b = int(math.isqrt(n))
  if b * b != n:
    raise ValueError(f"N must be a perfect square; got N={n}")
  # Validate entries
  for r, row in enumerate(grid):
    for c, v in enumerate(row):
      if v < 0 or v > n:
        raise ValueError(f"Cell ({r},{c}) has value {v} outside allowed range 0..{n}")
  return grid


def var_index(r: int, c: int, v: int, N: int) -> int:
  """Mapping: var(r,c,v) = r*N*N + c*N + v, with r,c in 0..N-1, v in 1..N."""
  return r * N * N + c * N + v


def exactly_one(vars_list: List[int]) -> List[List[int]]:
  """Return CNF clauses encoding exactly one of the literals is true.

  Uses: at-least-one (single clause of all vars) + at-most-one (pairwise negatives).
  """
  clauses: List[List[int]] = []
  # At least one
  clauses.append(list(vars_list))
  # At most one: pairwise
  for i in range(len(vars_list)):
    for j in range(i + 1, len(vars_list)):
      clauses.append([-vars_list[i], -vars_list[j]])
  return clauses


def to_cnf(input_path: str) -> Tuple[Iterable[Iterable[int]], int]:
  """
  Read puzzle from input_path and return (clauses, num_vars).

  - clauses: iterable of iterables of ints (each clause), no trailing 0s
  - num_vars: must be N^3 with N = grid size
  """
  grid = _read_puzzle(input_path)
  N = len(grid)
  B = int(math.isqrt(N))

  clauses: List[List[int]] = []

  # 1) Exactly one value per cell
  for r in range(N):
    for c in range(N):
      vars_cell = [var_index(r, c, v, N) for v in range(1, N + 1)]
      clauses.extend(exactly_one(vars_cell))

  # 2) Row constraint: for each v and row r, exactly one column c has v
  for r in range(N):
    for v in range(1, N + 1):
      vars_row_v = [var_index(r, c, v, N) for c in range(0, N)]
      clauses.extend(exactly_one(vars_row_v))

  # 3) Column constraint: for each v and column c, exactly one row r has v
  for c in range(N):
    for v in range(1, N + 1):
      vars_col_v = [var_index(r, c, v, N) for r in range(0, N)]
      clauses.extend(exactly_one(vars_col_v))

  # 4) Box constraint: for each v and each BxB box, exactly one cell in that box has v
  for br in range(0, N, B):
    for bc in range(0, N, B):
      cells = [(br + dr, bc + dc) for dr in range(B) for dc in range(B)]
      for v in range(1, N + 1):
        vars_box_v = [var_index(r, c, v, N) for (r, c) in cells]
        clauses.extend(exactly_one(vars_box_v))

  # 5) Non-consecutive rule: orthogonal neighbors cannot differ by 1
  # We'll add constraints only to the right and down neighbors to avoid duplicates
  for r in range(N):
    for c in range(N):
      # Right neighbor
      if c + 1 < N:
        rp, cp = r, c + 1
        for v in range(1, N + 1):
          if v + 1 <= N:
            clauses.append([-var_index(r, c, v, N), -var_index(rp, cp, v + 1, N)])
          if v - 1 >= 1:
            clauses.append([-var_index(r, c, v, N), -var_index(rp, cp, v - 1, N)])
      # Down neighbor
      if r + 1 < N:
        rp, cp = r + 1, c
        for v in range(1, N + 1):
          if v + 1 <= N:
            clauses.append([-var_index(r, c, v, N), -var_index(rp, cp, v + 1, N)])
          if v - 1 >= 1:
            clauses.append([-var_index(r, c, v, N), -var_index(rp, cp, v - 1, N)])

  # 6) Clues: unit clauses
  for r in range(N):
    for c in range(N):
      v = grid[r][c]
      if v != 0:
        # sanity: v in 1..N
        if not (1 <= v <= N):
          raise ValueError(f"Clue at ({r},{c}) has value {v} outside 1..{N}")
        clauses.append([var_index(r, c, v, N)])

  num_vars = N ** 3
  return clauses, num_vars