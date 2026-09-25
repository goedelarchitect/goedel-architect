# Example problems

`problems.jsonl` has one JSON object per line: `problem_id`, `problem` (the
informal statement, used by `--nl-proof`) and `formal_statement` (a complete
Lean file ending in `:= by sorry`). All statements compile with Lean/Mathlib
v4.27.0.

| `problem_id` | Source |
|---|---|
| `smoke_trivial_1` | warm-up (`n + 0 = n`) |
| `mathd_algebra_478`, `aime_1987_p5`, `imo_1959_p1`, `numbertheory_4x3m7y3neq2003` | [miniF2F](https://github.com/openai/miniF2F) (Zheng et al., 2021), Lean 4 test split |
| `putnam_1988_b1`, `putnam_2016_b3`, `putnam_2025_a1` | [PutnamBench](https://github.com/trishullab/PutnamBench) (Tsoukalas et al., 2024; Apache-2.0), with the final `:= sorry` written as `:= by sorry` |
| `usamo_2026_p1` | USAMO 2026 Problem 1, formalized by the Goedel-Architect authors |

To try your own problems, write them in the same format. The main theorem's
signature is kept verbatim, and every proof is checked against it.
