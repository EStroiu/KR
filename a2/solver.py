"""
SAT Assignment Part 2 - Non-consecutive Sudoku Solver (Puzzle -> SAT/UNSAT)

THIS is the file to edit.

Implement: solve_cnf(clauses) -> (status, model_or_None)"""


from typing import Iterable, List, Tuple
from random import randint


def remove_tautologies(clauses: Iterable[Iterable[int]]) -> Tuple[Iterable[Iterable[int]], Iterable[int]]:
  new_clauses = []
  removed = []
  for clause in clauses:
    found_tautology = False
    for literal in clause:
      if (-literal) in clause:
        found_tautology = True
        removed.append(abs(literal))
        break
    if not found_tautology:
      new_clauses.append(clause)
  return (new_clauses, removed)


def is_empty_set(clauses: Iterable[Iterable[int]]) -> bool:
  return len(clauses) == 0


def has_empty_clause(clauses: Iterable[Iterable[int]]) -> bool:
  return any(map(lambda x: len(x) == 0, clauses))


def has_unit_clause(clauses: Iterable[Iterable[int]]) -> Tuple[bool, int]:
  for clause in enumerate(clauses):
    if len(clause) == 1:
      return (True, clause[0])
  return (False, 0)


def has_literal(clauses: Iterable[Iterable[int]]) -> Tuple[bool, int]:
  literals = set([literal for clause in clauses for literal in clause])
  for literal in literals:
    if -literal not in literals:
      return (True, literal)
  return (False, 0)


def remove_literal(clauses: Iterable[Iterable[int]], literal: int) -> Iterable[Iterable[int]]:
  new_clauses = []
  for clause in clauses:
    c = list(clause)
    if literal in c:
      continue
    if -literal in c:
      c = [l for l in c if l != -literal]
    new_clauses.append(c)
  return new_clauses


def select_literal(clauses: Iterable[Iterable[int]]) -> int:
  literals = list(set([literal for clause in clauses for literal in clause]))
  return literals[randint(0, len(literals) - 1)]


def dp(clauses: Iterable[Iterable[int]]) -> Tuple[str, List[int] | None]:
  if is_empty_set(clauses):
    return ("SAT", [])
  if has_empty_clause(clauses):
    return ("UNSAT", None)
  unit_clause = has_unit_clause(clauses)
  if unit_clause[0]:
    new_clauses = remove_literal(unit_clause[1])
    result = dp(new_clauses)
    if result[0] == "SAT":
      result[1] = result[1].append(unit_clause[1])
      return result
    else:
      return result

  pure_literal = has_literal(clauses)
  if pure_literal[0]:
    new_clauses = remove_literal(clauses, pure_literal[1])
    result = dp(new_clauses)
    if result[0] == "SAT":
      result[1] = result[1].append(pure_literal[1])
      return result
    else:
      return result

  literal = select_literal(clauses)
  new_clauses = remove_literal(clauses, literal)
  result = dp(new_clauses)
  if result[0] == "SAT":
    result[1] = result[1].append(unit_clause[1])
    return result

  new_clauses = remove_literal(clauses, -literal)
  result = dp(new_clauses)
  if result[0] == "SAT":
    result[1] = result[1].append(unit_clause[1])
    return result
  else:
    return result

def solve_cnf(clauses: Iterable[Iterable[int]], num_vars: int) -> Tuple[str, List[int] | None]:
  """
  Implement your SAT solver here.
  Must return:
    ("SAT", model)  where model is a list of ints (DIMACS-style), or
    ("UNSAT", None)
  """
  (clauses, tautologies) = remove_tautologies(clauses)
  if is_empty_set(clauses):
    return ("SAT", tautologies)
  (SAT, result) = dp(clauses)
  if (SAT == "SAT"):
    result.extend(tautologies)
    return (SAT, result)
  return SAT, result
  
