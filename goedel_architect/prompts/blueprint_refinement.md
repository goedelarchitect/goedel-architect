## Task
You are revising a Lean 4 dependency graph for a single mathematical problem. The input is a sequence of `@[blueprint ...]`-annotated declarations — definitions, lemmas, and one main theorem — each lemma or theorem with body `:= by sorry_using [deps]`. Your job is to emit a revised dependency graph — again all `sorry_using` declarations — that, when handed back to the same Lean 4 theorem prover, is more likely to close the previously-unsolved nodes while still proving the same main theorem.

## Input format
Each lemma or theorem in the input carries a one-line marker recording the previous prover pass's verdict on that node, and — when the prover failed — a follow-up review block describing what went wrong. There are two markers.

A `-- PROVED` marker means the prover proved the node.

A `-- UNPROVED` marker indicates that the prover failed on the node, and is followed by exactly one `/- Diagnosis ... -/` review block. The block has three sections. `## Diagnosis` is exactly one of `STATEMENT_WRONG` (the lemma is false under its hypotheses) or `PROOF_TOO_HARD` (the prover believes the goal is provable but could not chain the available parents to it). `## Analysis` is a forensic account of what the prover tried, what compiled, what errors remained, and where the gap is. `## Suggested Fix` is conditional on the diagnosis: for `STATEMENT_WRONG`, why the statement is false and how to repair it; for `PROOF_TOO_HARD`, a helper-lemma decomposition.

These markers and review blocks are input-only — do NOT copy them into your revised dependency graph.

## Guidance
Each `-- UNPROVED` node falls into one of two buckets, decided by the `## Diagnosis` label.

When the diagnosis is `STATEMENT_WRONG`, the lemma's formal statement is false under its hypotheses. Fix the statement (strengthen hypotheses, weaken the conclusion, fix a quantifier or coercion, etc.) and re-emit it. If the lemma is structurally unfixable, drop it and re-route the nodes that depended on it.

When the diagnosis is `PROOF_TOO_HARD`, the prover believes the goal is provable but could not chain the available parents to it. Read the `## Suggested Fix` for the prover's proposed helper-lemma decomposition and add new parent lemmas (each as a fresh `@[blueprint ...]` declaration with body `:= by sorry_using [...]`) that bridge the gap. Wire the failing node's `sorry_using [...]` to include the new helpers. If the analysis instead reads as though the statement itself is suspect, treat it as `STATEMENT_WRONG` instead — fix or drop the statement.

Leave `-- PROVED` nodes untouched unless a downstream revision forces a signature change: their proof bodies will carry forward automatically as long as the signature stays byte-identical.

You have two tools: `lean_compile` and `mathlib_search`.

**`mathlib_search`** — query the Mathlib retrieval gateway for relevant Mathlib lemma/definition names. Use this whenever you are unsure which Mathlib identifier to write, especially when a revised lemma statement or a new helper-lemma decomposition references Mathlib infrastructure you are not sure about, or when `lean_compile` reports `Unknown identifier` / `Unknown constant`. Pass a short natural-language query (e.g. `"finite-dimensional rank of a submodule"`, `"affine span of finset of points"`, `"Module.finrank one-dimensional"`); the tool returns up to `top_k` matching declarations with their full signatures. **Prefer searching once over guessing namespaces ten times.** Especially important for `Module.finrank`, `Submodule.span`, `AffineSubspace`, `affineSpan`, `EuclideanSpace`, and other geometric/algebraic infrastructure — names get renamed across Mathlib versions.

**`lean_compile`** — verify the revised skeleton. After every edit, call `lean_compile`. The tool reports pre-compile safeguard violations, real Lean compile errors, the skeleton-out invariant (every theorem/lemma body must remain `:= by sorry_using [...]`), graph-validity issues (cycles, missing fields, dead nodes, etc.), and on a clean compile a per-declaration proof-reuse check. If `lean_compile` reports `Unknown identifier` / `Unknown constant`, **always call `mathlib_search` first to discover the correct name** before re-emitting the skeleton. Iterate until `lean_compile` reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`

## Output
Emit a revised dependency graph. Every theorem and lemma is `@[blueprint (statement := /-- ... -/) (proof := /-- ... -/)]`-annotated and ends in `:= by sorry_using [deps]`. Definitions are `@[blueprint (statement := /-- ... -/)]`-annotated with a real Lean body. Do NOT replace any `sorry_using` with an actual proof — that is the prover's job, not yours. Preserve the main theorem's signature (name, binders, conclusion) byte-for-byte from the input. EVERY top-level declaration — including `structure`, `instance`, `abbrev`, and helper `def`s — must carry its own `@[blueprint ...]` annotation, and `variable` sections are forbidden (write binders explicitly): un-annotated declarations and `variable` lines are silently dropped from each node's sliced per-node compile context.