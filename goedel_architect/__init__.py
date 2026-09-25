"""Goedel-Architect: blueprint-guided formal theorem proving in Lean 4.

Stages (each runnable as `python -m goedel_architect.<stage>`):

  nl_proof   optional natural-language proof guidance
  blueprint  blueprint generation (paper §3.1)
  prover     per-lemma theorem proving (§3.2)
  refine     blueprint refinement (§3.3)
  verify     final audit of claimed proofs

`python -m goedel_architect` runs the whole pipeline (pipeline.py).
"""

__version__ = "1.0.0"
