#!/usr/bin/env bash
# Run Non-consecutive Sudoku SAT solver experiments across all heuristics.
# Results are saved under a per-experiment folder: <desc>__<UTC timestamp>/solver-<name>/
# Requires: python3 and project layout a2/evaluate.py and solver files.

set -euo pipefail

# -------- Config (edit as needed) --------
PYTHON_BIN=${PYTHON_BIN:-python3}
BASE_OUT=${BASE_OUT:-"$(cd "$(dirname "$0")"/.. && pwd)/outputs/experiments"}
TIMEOUT_9=${TIMEOUT_9:-120}      # seconds per instance for N=9
TIMEOUT_25=${TIMEOUT_25:-600}    # seconds per instance for N=25
GEN_TIMEOUT=${GEN_TIMEOUT:-60}   # seconds budget for generating a solved grid (suite-mode=sat)
SEED=${SEED:-42}

# Solver list: paths relative to repo root
SOLVERS=(
  "solver.py"
  "solver_JW.py"
  "solver_MOM.py"
  "solver_DLIS.py"
)

# ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_PY="$REPO_ROOT/evaluate.py"
mkdir -p "$BASE_OUT"

_ts() { date -u +%Y%m%dT%H%M%SZ; }

run_experiment() {
  local desc="$1"; shift
  local eval_args=("$@")
  local exp_dir="$BASE_OUT/${desc}__$(_ts)"
  mkdir -p "$exp_dir"

  # Save meta
  {
    echo "description: $desc"
    echo "timestamp_utc: $(_ts)"
    echo "repo_root: $REPO_ROOT"
    echo "python: $($PYTHON_BIN --version 2>&1 | tr -d '\n')"
    echo "evaluate: $EVAL_PY"
    echo "solvers: ${SOLVERS[*]}"
    echo "timeout_9: $TIMEOUT_9"
    echo "timeout_25: $TIMEOUT_25"
    echo "gen_timeout: $GEN_TIMEOUT"
    echo "seed: $SEED"
    echo "args: ${eval_args[*]}"
  } > "$exp_dir/meta.txt"

  for solver_path in "${SOLVERS[@]}"; do
    local solver_name
    solver_name="$(basename "$solver_path" .py)"
    local out_dir="$exp_dir/solver-${solver_name}"
    mkdir -p "$out_dir"

    echo "[RUN] $desc :: $solver_name"
    # Choose timeout by size if present in args
    local timeout="$TIMEOUT_9"
    if printf '%s\n' "${eval_args[@]}" | grep -q -- "--sizes"; then
      # rough parse: if sizes contains 25, use TIMEOUT_25
      if printf '%s\n' "${eval_args[@]}" | grep -q 25; then
        timeout="$TIMEOUT_25"
      fi
    fi

    # Run evaluation
    ( cd "$REPO_ROOT" && \
      "$PYTHON_BIN" "$EVAL_PY" \
        --seed "$SEED" \
        --timeout "$timeout" \
        --gen-timeout "$GEN_TIMEOUT" \
        --outdir "$out_dir" \
        --solver "$solver_path" \
        "${eval_args[@]}" \
      2>&1 | tee "$out_dir/run.log" )
  done

  echo "[DONE] $desc -> $exp_dir"
}

# ------------------- Experiments -------------------
# 1) 100x 9x9 random (default density 0.2)
run_experiment \
  "n9_random_100_d20" \
  --suite-mode random --sizes 9 --instances-per-size 100 --clue-density 0.2

# 2) 100x 9x9 SAT vs UNSAT generated (p=0.5, density 0.2)
run_experiment \
  "n9_sat-vs-unsat_100_d20_p50" \
  --suite-mode sat --sizes 9 --instances-per-size 100 --clue-density 0.2 --unsat-proportion 0.5

# 3) 10x 25x25 random (default density 0.2)
run_experiment \
  "n25_random_10_d20" \
  --suite-mode random --sizes 25 --instances-per-size 10 --clue-density 0.2

# 4) 10x 25x25 SAT vs UNSAT generated (p=0.5, density 0.2)
run_experiment \
  "n25_sat-vs-unsat_10_d20_p50" \
  --suite-mode sat --sizes 25 --instances-per-size 10 --clue-density 0.2 --unsat-proportion 0.5

# 5) Clue density study on 9x9: 10%, 30%, 50% clues
for d in 0.1 0.3 0.5; do
  case "$d" in
    0.1) dtag=10;;
    0.3) dtag=30;;
    0.5) dtag=50;;
    *) dtag="$(echo "$d" | tr -d '.')";;
  esac
  # SAT only
  run_experiment \
    "n9_clues_sat_100_d${dtag}" \
    --suite-mode sat --sizes 9 --instances-per-size 100 --clue-density "$d" --unsat-proportion 0.0
  # UNSAT only
  run_experiment \
    "n9_clues_unsat_100_d${dtag}" \
    --suite-mode sat --sizes 9 --instances-per-size 100 --clue-density "$d" --unsat-proportion 1.0
  # Random
  run_experiment \
    "n9_clues_random_100_d${dtag}" \
    --suite-mode random --sizes 9 --instances-per-size 100 --clue-density "$d"
done

# ---------------------------------------------------

echo "All experiments queued. Results under: $BASE_OUT"
