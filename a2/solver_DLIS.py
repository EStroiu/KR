from typing import Iterable, List, Tuple, Optional
from random import choice
from collections import Counter


# For majority of optimizations bellow we took inspiration from original DPLL algorithm,
# SOURCE: https://en.wikipedia.org/wiki/DPLL_algorithm?utm_source=chatgpt.com
# and also applied some heuristics.
def remove_tautologies(clauses: Iterable[Iterable[int]]) -> Tuple[List[List[int]], List[int]]:
  """ 
  Returns the filtered clauses and a list of variables that appeared in tautologies (for info).
  """
  new_clauses: List[List[int]] = []
  removed: List[int] = []
  for clause in clauses:
    c = list(clause)
    s = set(c)
    found_tautology = any((-lit) in s for lit in s)
    if found_tautology:
      removed.extend([abs(l) for l in s])
    else:
      new_clauses.append(c)
  return new_clauses, removed


def is_empty_set(clauses: Iterable[Iterable[int]]) -> bool:
  return len(list(clauses) if not isinstance(clauses, list) else clauses) == 0


def has_empty_clause(clauses: Iterable[Iterable[int]]) -> bool:
  return any(len(c) == 0 for c in clauses)


def has_unit_clause(clauses: Iterable[Iterable[int]]) -> Tuple[bool, int]:
  """Return (True, lit) if there exists a unit clause [lit], else (False, 0)."""
  for clause in clauses:
    if len(clause) == 1:
      return True, clause[0]
  return False, 0


def has_literal(clauses: Iterable[Iterable[int]]) -> Tuple[bool, int]:
  """Return (True, lit) if a pure literal exists, else (False, 0)."""
  literals = set(l for clause in clauses for l in clause)
  for literal in literals:
    if -literal not in literals:
      return True, literal
  return False, 0


def remove_literal(clauses: Iterable[Iterable[int]], literal: int) -> List[List[int]]:
  """Assign the given literal to True and simplify clauses accordingly."""
  new_clauses: List[List[int]] = []
  for clause in clauses:
    # Clause satisfied -> drop
    if literal in clause:
      continue
    # Remove negation -> shrink if present; otherwise reuse clause to avoid copy
    if -literal in clause:
      new_clauses.append([l for l in clause if l != -literal])
    else:
      # clauses are normalized to list[list[int]] in dp(), so safe to keep by reference
      new_clauses.append(clause if isinstance(clause, list) else list(clause))
  return new_clauses


# SOURCE: www.cs.cmu.edu/~emc/15-820A/reading/sat_cmu.pdf
# Use of Early branching Heuristics
def select_literal(clauses: Iterable[Iterable[int]]) -> int:
  """Pick a branching literal using DLIS (choose polarity with max occurrences).

  We optionally bias to the smallest non-unit clauses: if any exist, we count occurrences
  only among clauses with minimal length > 1; otherwise, count across all clauses.
  """
  # Gather clauses and track smallest non-unit length
  clists: List[List[int]] = []
  min_len: Optional[int] = None
  any_lit = 0
  for c in clauses:
    cl = c if isinstance(c, list) else list(c)
    clists.append(cl)
    if cl:
      any_lit = any_lit or cl[0]
    m = len(cl)
    if m > 1 and (min_len is None or m < min_len):
      min_len = m

  if not clists:
    return any_lit if any_lit != 0 else 1

  if min_len is not None:
    pool = [cl for cl in clists if len(cl) == min_len]
  else:
    pool = clists

  lits = [l for cl in pool for l in cl]
  if not lits:
    return any_lit if any_lit != 0 else 1

  vars_set = {abs(l) for l in lits}
  cnt = Counter(lits)
  best = None  # (count, literal)
  for v in vars_set:
    c_pos = cnt.get(v, 0)
    c_neg = cnt.get(-v, 0)
    # choose stronger polarity; tie -> positive
    if c_pos >= c_neg:
      cand = (c_pos, v)
    else:
      cand = (c_neg, -v)
    if best is None or cand[0] > best[0]:
      best = cand

  if best is None or best[0] <= 0:
    return any_lit if any_lit != 0 else 1
  return best[1]


def dp(clauses: Iterable[Iterable[int]], model: Optional[List[int]] = None) -> Tuple[str, List[int] | None]:
  """Iterative (non-recursive since it can time out otherwise because python moment) DPLL 
  solver (was explained in the assignment). Returns ("SAT", model) or ("UNSAT", None).
  """
  # Normalize clauses to list of lists for consistent operations -> also helps with performance
  current_clauses: List[List[int]] = [list(c) for c in clauses]
  current_model: List[int] = [] if model is None else list(model)

  # We checks on the initial input
  if is_empty_set(current_clauses):
    return ("SAT", list(current_model))
  if has_empty_clause(current_clauses):
    return ("UNSAT", None)

  def simplify(clauses_in: List[List[int]], model_in: List[int]) -> Tuple[str, Optional[List[List[int]]], Optional[List[int]]]:
    clauses_loc = clauses_in
    model_loc = model_in

    # Our main unit propagation loop
    changed = True
    while changed:
      changed = False
      found, lit = has_unit_clause(clauses_loc)
      if found:
        clauses_loc = remove_literal(clauses_loc, lit)
        model_loc = model_loc + [lit]
        changed = True
        if is_empty_set(clauses_loc):
          return ("SAT", None, model_loc)
        if has_empty_clause(clauses_loc):
          return ("UNSAT", None, None)

    # Just pure literal elimination to a fixpoint
    while True:
      found_pure, plit = has_literal(clauses_loc)
      if not found_pure:
        break
      clauses_loc = remove_literal(clauses_loc, plit)
      model_loc = model_loc + [plit]
      if is_empty_set(clauses_loc):
        return ("SAT", None, model_loc)
      if has_empty_clause(clauses_loc):
        return ("UNSAT", None, None)

    # Another unit propagation pass after PLE
    changed = True
    while changed:
      changed = False
      found, lit = has_unit_clause(clauses_loc)
      if found:
        clauses_loc = remove_literal(clauses_loc, lit)
        model_loc = model_loc + [lit]
        changed = True
        if is_empty_set(clauses_loc):
          return ("SAT", None, model_loc)
        if has_empty_clause(clauses_loc):
          return ("UNSAT", None, None)

    return ("CONTINUE", clauses_loc, model_loc)

  # Each stack frame stores a deferred branch to try upon backtracking:
  # (clauses_snapshot, model_snapshot, alt_literal)
  stack: List[Tuple[List[List[int]], List[int], int]] = []

  # Lightweight failed-literal detection: unit-propagate quickly to see if a branch is doomed
  def quick_unit_ok(cls: List[List[int]]) -> bool:
    while True:
      found, u = has_unit_clause(cls)
      if not found:
        return not has_empty_clause(cls)
      cls = remove_literal(cls, u)
      if has_empty_clause(cls):
        return False

  while True:
    # 1. Simplify to a local fixpoint
    status, maybe_clauses, maybe_model = simplify(current_clauses, current_model)
    if status == "SAT":
      return ("SAT", maybe_model)
    if status == "UNSAT":
      # 2. Now Backtrack
      if not stack:
        return ("UNSAT", None)
      current_clauses, current_model, alt_lit = stack.pop()
      current_clauses = remove_literal(current_clauses, alt_lit)
      current_model = current_model + [alt_lit]
      continue

    # Continue search with simplified state
    # Assign directly; do not use truthiness fallbacks that could swallow empty lists.
    assert maybe_clauses is not None and maybe_model is not None
    current_clauses = maybe_clauses
    current_model = maybe_model

    # Choose a branching literal and push the alternative branch for later
    lit = select_literal(current_clauses)
    # Simple failed-literal check: if picking lit quickly contradicts, flip polarity
    try_cls = remove_literal(current_clauses, lit)
    if not quick_unit_ok(try_cls):
      lit = -lit
    # Save state to try the opposite polarity later
    stack.append((current_clauses, current_model, -lit))
    # Take the chosen branch immediately
    current_clauses = remove_literal(current_clauses, lit)
    current_model = current_model + [lit]

def solve_cnf(clauses: Iterable[Iterable[int]], num_vars: int) -> Tuple[str, List[int] | None]:
  """
  Implement your SAT solver here.
  Must return:
    ("SAT", model)  where model is a list of ints (DIMACS-style), or
    ("UNSAT", None)
  """
  # Preprocess: remove tautologies
  clauses_no_tauts, _tauts = remove_tautologies(clauses)

  normalized: List[List[int]] = []
  for c in clauses_no_tauts:
    # We use set to remove duplicates
    if not isinstance(c, list):
      c = list(c)
    if len(c) > 1:
      s = set(c)
      if len(s) != len(c):
        normalized.append(list(s))
      else:
        normalized.append(c)
    else:
      normalized.append(c)

  status, model = dp(normalized)
  return status, model
  
