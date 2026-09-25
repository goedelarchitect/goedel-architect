"""
Build a "negation" formal_statement template from an original Lean 4
theorem statement.

Given:

    theorem foo (b1) (b2) ... (bN) : C := by sorry

returns:

    theorem foo_negation : ¬ (∀ b1 b2 ... bN, C) := by sorry

(if there are no binders, just `¬ (C)`).

The header / preamble before the original theorem is preserved, EXCEPT
that all `/- ... -/` and `/-- ... -/` block comments are stripped — those
typically hold the natural-language description of the *positive*
theorem ("Prove that ...") and would mislead the model on the negation
path, where the goal is to disprove. Imports, `set_option`, `open`, and
helper `def` / `structure` declarations are kept verbatim.

The prover offers this `<name>_negation` theorem as a parallel target, so
the model can register a formal disproof of a lemma it believes is false;
the refinement stage then repairs or drops that lemma.
"""

from __future__ import annotations

import re

__all__ = ["build_negation_formal_statement"]


def _strip_block_comments(s: str) -> str:
    """Remove all `/- ... -/` and `/-- ... -/` block comments from `s`.

    Handles nested block comments (Lean 4 supports them). Does not look
    inside string literals — formal-statement headers rarely contain
    any, and conflating string syntax here would only complicate the
    parse. Line breaks inside the stripped span are dropped along with
    the comment; downstream callers normalise whitespace anyway.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if i + 1 < n and s[i] == "/" and s[i + 1] == "-":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and s[i] == "/" and s[i + 1] == "-":
                    depth += 1
                    i += 2
                elif i + 1 < n and s[i] == "-" and s[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _find_sig_colon(s: str) -> int:
    """Return the index of the top-level `:` separating binders from the
    conclusion in a Lean 4 theorem signature, or -1 if not found.

    Tracks `() [] {}` depth and skips `:=` (it would mean we ran past
    the signature into `:= by sorry` without finding a separating colon,
    which means there are no binders *and* no conclusion, i.e.
    parse fail).
    """
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return -1
        elif c == ":" and depth == 0:
            # `:=` at depth 0 means we hit the proof terminator before
            # any signature colon — only legal if the theorem is bodyless,
            # which we don't support here.
            if i + 1 < n and s[i + 1] == "=":
                return -1
            return i
        i += 1
    return -1


_THEOREM_SUFFIX_RE = re.compile(r":=\s*by\s+sorry\s*$")


def build_negation_formal_statement(
    formal_statement: str, theorem_name: str
) -> str | None:
    """Return a `formal_statement`-shaped string whose canonical theorem
    is `<theorem_name>_negation : ¬ (∀ <binders>, <conclusion>)`.

    Returns None if the original signature can't be parsed (e.g. it's a
    term-mode `:= <term>` rather than `:= by sorry`, or the theorem name
    isn't found, or paren depth is unbalanced).

    The header / preamble / helper decls *before* the original `theorem`
    line are preserved verbatim; only the theorem block itself is
    replaced with the negation block.
    """
    # Locate `theorem NAME` (or `lemma NAME`; Lean 4 treats them as
    # synonyms) at the start of a line. Tolerates the same leading
    # modifiers as the blueprint parser's `_DECL_KIND_RE`
    # (private/protected/noncomputable), since `node.signature` carries
    # them verbatim.
    # `(?![\w'])` instead of `\b` — Lean 4 identifiers can carry `'` as a
    # continuation char and Python's `\b` mishandles that. See
    # blueprint_engine.check_theorem_present for the rationale.
    name_re = re.compile(
        r"^\s*(?:private\s+|protected\s+)?(?:noncomputable\s+)?"
        r"(?:theorem|lemma)\s+" + re.escape(theorem_name) + r"(?![\w'])",
        re.MULTILINE,
    )
    m = name_re.search(formal_statement)
    if m is None:
        return None

    prefix = formal_statement[: m.start()]
    # Strip `/- ... -/` / `/-- ... -/` block comments from the prefix.
    # They describe the positive theorem's informal statement and would
    # mislead the model on the negation path. Collapse runs of >2 blank
    # lines that may result from the strip, for tidiness.
    prefix = _strip_block_comments(prefix)
    prefix = re.sub(r"\n{3,}", "\n\n", prefix)
    # m.end() points just past `NAME`; what's after is the binder/sig text.
    rest = formal_statement[m.end():]

    sig_colon = _find_sig_colon(rest)
    if sig_colon < 0:
        return None

    binders = rest[:sig_colon].strip()
    after_colon = rest[sig_colon + 1:]

    suffix_match = _THEOREM_SUFFIX_RE.search(after_colon)
    if suffix_match is None:
        return None
    conclusion = after_colon[: suffix_match.start()].strip()
    if not conclusion:
        return None

    if binders:
        body = f"∀ {binders}, {conclusion}"
    else:
        body = conclusion

    neg_block = (
        f"theorem {theorem_name}_negation : ¬ ({body}) := by sorry"
    )

    # Preserve a single trailing newline like the originals tend to have.
    if not prefix.endswith("\n") and prefix:
        prefix = prefix + "\n"
    return prefix + neg_block + "\n"
