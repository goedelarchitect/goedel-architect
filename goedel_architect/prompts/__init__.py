"""System prompts, one file per role, loaded verbatim.

  blueprint_generation.md              blueprint generation from the formal statement
  blueprint_generation_nl_guided.md    ... guided by a natural-language proof
  theorem_proving.md                   per-lemma prover
  theorem_proving_negation.md          bullet added when the negation path is on
  theorem_proving_forfeit.md           structured post-mortem after a failed lemma
  theorem_proving_disproof_reflection.md  post-mortem after a formal disproof
  blueprint_refinement.md              blueprint refinement
  nl_proof_solver.md                   natural-language prover (solver)
  nl_proof_refiner_addendum.md         appended to the solver prompt for refinement
  nl_proof_grader.md                   natural-language proof grader

Files are read byte-for-byte (a missing or extra trailing newline changes
the prompt), so edit them with care.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the prompt `<name>.md` exactly as stored."""
    with open(_DIR / f"{name}.md", encoding="utf-8", newline="") as f:
        return f.read()


_TWO_TOOLS = "You have two tools: `lean_compile` and `mathlib_search`."
_ONE_TOOL = "You have one tool: `lean_compile`."


def without_search(text: str, edits: tuple[tuple[str, str], ...] = ()) -> str:
    """Variant of a system prompt for runs where `mathlib_search` is off.

    Drops the paragraph that documents `mathlib_search`, rewrites the tool
    count, and applies `edits` (each `old` must occur exactly once). Raises
    if any mention of `mathlib_search` survives, so prompt drift is caught
    instead of silently advertising a tool the model does not have.
    """
    paragraphs = text.split("\n\n")
    paragraphs = [p for p in paragraphs if not p.startswith("**`mathlib_search`**")]
    text = "\n\n".join(paragraphs)
    if _TWO_TOOLS in text:
        text = text.replace(_TWO_TOOLS, _ONE_TOOL, 1)
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"search edit does not match exactly once: {old[:60]!r}")
        text = text.replace(old, new, 1)
    if "mathlib_search" in text:
        raise ValueError("prompt still mentions mathlib_search after removing search")
    return text
