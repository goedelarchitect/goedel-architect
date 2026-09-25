You have exhausted your budget for this proof. The proof attempt is OVER. You MUST NOT call any tool — no `lean_compile`, no `mathlib_search`, no tool of any kind. Output your final assessment in markdown using the section headers below, and nothing else (one and only one final assessment, then stop):

## Diagnosis
<exactly one of STATEMENT_WRONG or PROOF_TOO_HARD here>

## Analysis
A detailed forensic account of what you tried and why you couldn't finish: which approaches you took, what compiled and what didn't, what errors remained unresolved, and where exactly the gap is. At least several paragraphs.

## Suggested Fix
If STATEMENT_WRONG: explain why the theorem is false. Provide a counterexample if you have one. Identify whether the hypothesis is too weak or the conclusion is too strong, and suggest how to strengthen / weaken the statement.

If PROOF_TOO_HARD: propose a lemma decomposition. Describe a set of helper lemmas that together break the proof down. The lemmas form a dependency graph: each lemma should be easy to prove given its parent lemmas, and in particular the main theorem should become easy or trivial given its direct parent lemmas.
