# Knowledge Representation Assignments

## A1: Non-Consecutive Sudoku -> DIMACS CNF

Rules encoded:
- Exactly one value per cell
- Row/Column: each digit appears exactly once per row and per column
- Box: each digit appears exactly once per B×B box (B=sqrt{N})
- Non-consecutive: orthogonally adjacent cells cannot differ by 1
- Clues are added as unit clauses

### How to run

Use the CLI to convert a puzzle text file into a CNF file:

``` bash
python a1/CODE/main.py --in "a1/EXAMPLE puzzles (input)/example_n9.txt" --out a1/CODE/example_n9.cnf
```

If `--out` is omitted, the CNF will be written to stdout.

Input format: space-separated N×N integers, 0 denotes empty.

Variable mapping (required by assignment): `var(r,c,v) = r*N^2 + c*N + v` with r,c in {0..N-1}, v in {1..N}.

## Validator

I also created small validator for sanity-checking the generated CNF for a given puzzle. It verifies:
- Variable count: num_vars == N^3
- Clause count matches the encoder formula
	- Exactly-one constraints: 4*N^2 groups, each contributes (1 + C(N,2)) clauses
	- Non-consecutive constraints: 2*N*(N-1) orthogonal pairs per axis → total 4*N*(N-1) pairs; each pair contributes 2*(N-1) binary clauses → 4*N*(N-1)^2
	- Clues: K unit clauses
- Unit clauses exactly match the puzzle clues
- All literal ids are within [1..num_vars]

Run the validator like this:

```bash
# 9x9
python a1/CODE/validate_encoder.py --in "a1/EXAMPLE puzzles (input)/example_n9.txt"

# 16x16
python a1/CODE/validate_encoder.py --in "a1/EXAMPLE puzzles (input)/example_n16.txt"
```

Expected output ends with `Result: PASS` if all checks succeed.