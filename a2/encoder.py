"""
From SAT Assignment Part 1 - Non-consecutive Sudoku Encoder (Puzzle -> CNF)

Replace this code with your solution for assignment 1

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
  
  # Check that the input is perfect square
  b = int(math.isqrt(n))
  if b * b != n:
    raise ValueError(f"N must be a perfect square; got N={n}")
  
  # Check that the entries are following the proper format
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
  GridSize = len(grid) # This is N from the assignment
  BoxSize = int(math.isqrt(GridSize)) # This is B from the assignment

  clauses: List[List[int]] = []

  # 1) Basic check => exactly one value per cell
  for row_index in range(GridSize):
    for column_index in range(GridSize):
      vars_cell = [var_index(row_index, column_index, value, GridSize) for value in range(1, GridSize + 1)]
      clauses.extend(exactly_one(vars_cell))

  # 2) Main row constraint
  for row_index in range(GridSize):
    for value in range(1, GridSize + 1):
      vars_row_v = [var_index(row_index, column_index, value, GridSize) for column_index in range(0, GridSize)]
      clauses.extend(exactly_one(vars_row_v))

  # 3) Main column constraint
  for column_index in range(GridSize):
    for value in range(1, GridSize + 1):
      vars_col_v = [var_index(row_index, column_index, value, GridSize) for row_index in range(0, GridSize)]
      clauses.extend(exactly_one(vars_col_v))

  # 4) Main box constraint
  for box_row_index in range(0, GridSize, BoxSize):
    for box_column_index in range(0, GridSize, BoxSize):
      cells = [(box_row_index + dr, box_column_index + dc) for dr in range(BoxSize) for dc in range(BoxSize)]
      for value in range(1, GridSize + 1):
        vars_box_v = [var_index(row_index, column_index, value, GridSize) for (row_index, column_index) in cells]
        clauses.extend(exactly_one(vars_box_v))

  # 5) Additional non-consecutive rule => orthogonal neighbors cannot differ by 1
  # We'll add constraints only to the right and down neighbors to avoid duplicates
  for row_index in range(GridSize):
    for column_index in range(GridSize):
      
      # The right neighbor
      if column_index + 1 < GridSize:
        row_neighbors_position, current_position = row_index, column_index + 1
        for value in range(1, GridSize + 1):
          if value + 1 <= GridSize:
            clauses.append([-var_index(row_index, column_index, value, GridSize), -var_index(row_neighbors_position, current_position, value + 1, GridSize)])
          if value - 1 >= 1:
            clauses.append([-var_index(row_index, column_index, value, GridSize), -var_index(row_neighbors_position, current_position, value - 1, GridSize)])
      
      # The down neighbor
      if row_index + 1 < GridSize:
        row_neighbors_position, current_position = row_index + 1, column_index
        for value in range(1, GridSize + 1):
          if value + 1 <= GridSize:
            clauses.append([-var_index(row_index, column_index, value, GridSize), -var_index(row_neighbors_position, current_position, value + 1, GridSize)])
          if value - 1 >= 1:
            clauses.append([-var_index(row_index, column_index, value, GridSize), -var_index(row_neighbors_position, current_position, value - 1, GridSize)])

  # 6) Clues: unit clauses
  for row_index in range(GridSize):
    for column_index in range(GridSize):
      value = grid[row_index][column_index]
      if value != 0:
        # sanity: value in 1..GridSize
        if not (1 <= value <= GridSize):
          raise ValueError(f"Clue at ({row_index},{column_index}) has value {value} outside 1..{GridSize}")
        clauses.append([var_index(row_index, column_index, value, GridSize)])

  num_vars = GridSize ** 3
  return clauses, num_vars