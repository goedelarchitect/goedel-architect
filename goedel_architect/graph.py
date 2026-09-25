"""Blueprint engine: parse, slice, graft, assemble, and prune LeanArchitect blueprints.

A blueprint is a single Lean file whose declarations carry `@[blueprint]`
annotations; each lemma/theorem body is `:= by sorry_using [deps]`, which
records its dependency edges. This module is pure Python (plus a few Lean
server round-trips) and is shared by all stages:

  * `parse_blueprint_file` turns the file into `BlueprintNode`s;
  * `build_dependency_block` builds the per-lemma context the prover sees
    (every definition verbatim, the lemma's direct parents stubbed `by sorry`);
  * `build_safe_code` / `graft_proof_body` rebuild a submission under the
    canonical signature; `assemble_final_file` substitutes proved bodies
    back into the blueprint in place;
  * `parse_axioms_from_result` / `classify_axioms` audit `#print axioms`;
  * `prune_dead_nodes*` drop nodes unreachable from the main theorem.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from .lean import check_lean_async, check_lean_sync

log = logging.getLogger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
#  Blueprint parser
# ---------------------------------------------------------------------------

@dataclass
class BlueprintNode:
    name: str                        # "factorial_decomposition"
    kind: str                        # "theorem"|"lemma"|"def"|"abbrev"|"structure"|"noncomputable_def"|"inductive"|"class"
    is_target: bool                  # True iff it's a theorem/lemma with `sorry_using` body
    raw_source: str                  # full declaration text (attribute + body)
    attr_source: str                 # `@[blueprint ...]` text
    decl_source: str                 # declaration text (no attribute)
    signature: str                   # "theorem foo ... : T" (everything before `:=`)
    body_source: str                 # `:= by sorry_using [...]` portion (may be empty for defs)
    informal_statement: str          # from `(statement := /-- ... -/)`
    informal_proof: str              # from `(proof := /-- ... -/)`
    depends_on: list[str] = field(default_factory=list)
    # File-absolute offsets for in-place substitution:
    attr_start: int = 0
    decl_start: int = 0
    decl_end: int = 0
    # sorry_using token offsets (only set for target nodes):
    sorry_start: int = -1
    sorry_end: int = -1


def _skip_block_comment(code: str, start: int) -> int:
    """Given code[start:start+2] == '/-', return the offset AFTER the
    matching '-/'. Handles Lean's nested block comments. If unterminated
    (depth never reaches 0), returns len(code)."""
    n = len(code)
    i = start + 2
    depth = 1
    while i < n and depth > 0:
        if i + 1 < n and code[i] == '/' and code[i + 1] == '-':
            depth += 1
            i += 2
        elif i + 1 < n and code[i] == '-' and code[i + 1] == '/':
            depth -= 1
            i += 2
        else:
            i += 1
    return i


def _skip_string(code: str, start: int) -> int:
    """Given code[start] == '"', return the offset AFTER the closing quote."""
    n = len(code)
    i = start + 1
    while i < n and code[i] != '"':
        if code[i] == '\\' and i + 1 < n:
            i += 2
        else:
            i += 1
    return i + 1 if i < n else i


def _find_attribute_end(code: str, start: int) -> int | None:
    """Given code[start] == '@' and code[start+1] == '[', find the
    offset AFTER the matching ']'. Respects nested brackets, block
    comments, line comments, and strings. Returns None if unmatched."""
    n = len(code)
    if start + 1 >= n or code[start] != '@' or code[start + 1] != '[':
        return None
    i = start + 2
    depth = 1
    while i < n and depth > 0:
        c = code[i]
        if c == '[':
            depth += 1
            i += 1
        elif c == ']':
            depth -= 1
            i += 1
        elif c == '/' and i + 1 < n and code[i + 1] == '-':
            i = _skip_block_comment(code, i)
        elif c == '-' and i + 1 < n and code[i + 1] == '-':
            while i < n and code[i] != '\n':
                i += 1
        elif c == '"':
            i = _skip_string(code, i)
        else:
            i += 1
    return i if depth == 0 else None


def _find_next_top_level_attr(code: str, start: int) -> int:
    """Return the offset of the next `@[blueprint` that sits at the
    start of a line (after any newline + whitespace), or len(code) if
    none. `start` is the offset to search from."""
    n = len(code)
    i = start
    while i < n:
        nl = code.find('\n', i)
        if nl < 0:
            return n
        j = nl + 1
        while j < n and code[j] in ' \t':
            j += 1
        if j < n and code[j] == '@' and code[j:j + 11] == '@[blueprint':
            return j
        i = nl + 1
    return n


_DECL_KIND_RE = re.compile(
    r'^\s*(?:private\s+|protected\s+)?'
    r'(?:(noncomputable)\s+)?'
    r'(theorem|lemma|def|abbrev|structure|inductive|class)\s+'
    r'(\S+)',
    re.DOTALL,
)

_TARGET_KINDS = {"theorem", "lemma"}
_DEFINITION_KINDS = {
    "def", "abbrev", "structure", "inductive", "class",
    "noncomputable_def",
}


def _parse_decl_head(decl_text: str) -> tuple[str, str] | None:
    """Return (kind, name) from the head of a declaration, or None.

    `kind` is one of: theorem, lemma, def, abbrev, structure, inductive,
    class, noncomputable_def."""
    m = _DECL_KIND_RE.match(decl_text)
    if not m:
        return None
    nc = m.group(1)
    kw = m.group(2)
    name = m.group(3)
    # Strip any binder like `{` that might have stuck to the name
    name = re.split(r'[\s(:,\[{]', name, 1)[0]
    if not name:
        return None
    if nc and kw == "def":
        kind = "noncomputable_def"
    else:
        kind = kw
    return kind, name


def _extract_informal_field(attr_source: str, field_name: str) -> str:
    """Extract the text inside `(<field_name> := /-- ... -/)` from an
    `@[blueprint ...]` attribute source. Returns the inner text
    stripped, or '' if not found."""
    pattern = (
        r'\(\s*' + re.escape(field_name) + r'\s*:=\s*'
        r'/--(.*?)-/'
    )
    m = re.search(pattern, attr_source, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_sorry_using(
    decl_text: str, decl_start_offset: int,
) -> tuple[list[str], int, int]:
    """Find `sorry_using [...]` in a declaration body.

    Returns (deps, absolute_token_start, absolute_token_end). If no
    sorry_using is found, returns ([], -1, -1). `decl_start_offset` is
    the offset of the decl in the enclosing file, used to turn
    decl-local offsets into file-absolute ones for in-place
    substitution."""
    m = re.search(r'\bsorry_using\s*\[([^\]]*)\]', decl_text)
    if not m:
        return [], -1, -1
    raw = m.group(1).strip()
    deps: list[str] = []
    if raw:
        for tok in re.split(r',\s*', raw):
            tok = tok.strip()
            if tok:
                deps.append(tok)
    # 's' of 'sorry_using'
    sorry_start = decl_start_offset + m.start()
    sorry_end = decl_start_offset + m.end()
    return deps, sorry_start, sorry_end


# A `:=` whose preceding text ends in a `let <binder>` is the let-binding,
# not the declaration boundary. The binder is one of: a plain identifier
# (`let x`, `let ψ` — Lean idents are unicode, so Greek/subscripts count),
# a tuple/anonymous-constructor pattern (`let (a, b, c)`, `let ⟨⟨p, q⟩, r⟩`),
# each optionally type-ascribed (`let x : T`). Putnam "find-the-answer" goals
# encode the answer as `let (a, f, g) := <sol>; P`, and map/homomorphism goals
# as `let ψ := <fn>; P`, so without the tuple/⟨⟩ alternatives AND a unicode-safe
# plain-binder class the inner `let` `:=` was mistaken for the proof boundary
# and the goal type was truncated.
_LET_VALUE_COLON_EQ_RE = re.compile(
    r'\blet\s+'
    r'(?:[^\s(),;:=⟨⟩]+|\([^=;]*\)|⟨[^=;]*⟩)'  # ident (unicode) | (tuple) | ⟨anon ctor⟩
    r'(?:\s*:\s*[^=;]+)?'                       # optional `: type`
    r'\s*$',
)


def _split_signature_and_body(decl_text: str) -> tuple[str, str]:
    """Split a declaration into (signature, body) where `body` starts
    with `:=`. If no `:=` is found (e.g. a structure with `where`),
    the whole decl is the signature and body is ''.

    Finds the OUTERMOST `:=` — not just the first.  A naive first-match
    is wrong when the type contains ``let X := by <term>`` or ``let X
    := <value>``, because that inner `:=` is the binding of the let,
    not the declaration boundary.  Same heuristic as
    `_find_real_proof_by`: skip any `:=` immediately preceded by ``let
    <name>[: <type>]``."""
    for m in re.finditer(r':=', decl_text):
        pre = decl_text[:m.start()].rstrip()
        if _LET_VALUE_COLON_EQ_RE.search(pre):
            continue
        return decl_text[:m.start()].rstrip(), decl_text[m.start():]
    return decl_text, ""


def _is_header_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("--"):
        return True
    return s.startswith((
        "import ", "set_option ", "open ", "universe ", "namespace ",
    ))


def split_header_body(code: str) -> tuple[str, str]:
    """Split code into (header, body) by line kind. Header lines are
    imports, set_option, open, universe, and namespace; body starts at
    the first declaration."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if not _is_header_line(line):
            return "\n".join(lines[:i]), "\n".join(lines[i:])
    return code, ""


def parse_blueprint_file(code: str) -> tuple[str, list[BlueprintNode]]:
    """Parse a skeleton-produced blueprint `.lean` file.

    Returns `(header, nodes)` where `header` is the prelude (imports,
    set_option, open statements) and `nodes` is the list of
    `@[blueprint]`-annotated top-level declarations in source order.

    Declarations without `@[blueprint]` are skipped. Declarations with
    unknown kinds are skipped."""
    n = len(code)
    nodes: list[BlueprintNode] = []

    # Phase 1: header by line-kind walk
    header, _ = split_header_body(code)
    # Header offset in code (may differ from len(header) if code doesn't
    # end header with a newline)
    if header:
        header_end = len(header)
        # Skip the trailing newline if present
        if header_end < n and code[header_end] == '\n':
            header_end += 1
    else:
        header_end = 0

    # Phase 2: scan body for @[blueprint ...] declarations
    i = header_end
    while i < n:
        # Find the next `@[blueprint` from position i
        attr_start = code.find('@[blueprint', i)
        if attr_start < 0:
            break
        # Sanity: must be at line start (ignore inline `@[blueprint` in
        # comments/strings)
        line_start = code.rfind('\n', 0, attr_start) + 1
        prefix = code[line_start:attr_start]
        if prefix.strip():
            i = attr_start + 1
            continue

        attr_end = _find_attribute_end(code, attr_start)
        if attr_end is None:
            log.warning("Unbalanced attribute at offset %d", attr_start)
            break

        # Skip whitespace/newlines to find the declaration start
        decl_start = attr_end
        while decl_start < n and code[decl_start] in ' \t\n':
            decl_start += 1
        if decl_start >= n:
            break

        # Find the end of this declaration: next top-level @[blueprint
        decl_end = _find_next_top_level_attr(code, decl_start)
        # Trim trailing whitespace
        while decl_end > decl_start and code[decl_end - 1] in ' \t\n':
            decl_end -= 1
        # Strip trailing comments that belong to the *next* declaration.
        # These can be multi-line doc comments (/-- ... -/) or line
        # comments (-- ...) sitting between the current decl body and
        # the next @[blueprint].  Strategy: find the last top-level
        # `/--` that starts on its own line after the declaration
        # keyword, then check whether everything from there to the end
        # is comment-only.
        _tmp = code[decl_start:decl_end]
        # Find where the actual declaration body ends: last occurrence
        # of sorry_using/`:= by`/`:=` line, or the last non-comment
        # content.  Simplest: repeatedly strip a trailing block comment
        # or trailing line comments.
        _changed = True
        while _changed:
            _changed = False
            _s = _tmp.rstrip()
            # Strip trailing multi-line block comment: /-- ... -/ or /- ... -/
            if _s.endswith('-/'):
                # Find the matching opening /- or /--
                idx = _s.rfind('\n/-')
                if idx >= 0:
                    # Check: is everything from idx+1 onward a block comment?
                    candidate = _s[idx + 1:]
                    if candidate.lstrip().startswith('/-'):
                        _tmp = _s[:idx].rstrip()
                        _changed = True
                        continue
                # Also handle /-- at start of string (no preceding newline)
                if _s.lstrip().startswith('/-'):
                    _tmp = ''
                    _changed = True
                    continue
            # Strip trailing line comments (-- ...)
            last_nl = _s.rfind('\n')
            if last_nl >= 0:
                last_line = _s[last_nl + 1:].strip()
                if last_line.startswith('--'):
                    _tmp = _s[:last_nl].rstrip()
                    _changed = True
        decl_end = decl_start + len(_tmp)

        attr_source = code[attr_start:attr_end]
        decl_source = code[decl_start:decl_end]
        raw_source = code[attr_start:decl_end]

        head = _parse_decl_head(decl_source)
        if head is None:
            log.warning(
                "Could not parse declaration head at offset %d: %r",
                decl_start, decl_source[:60],
            )
            i = decl_end
            continue
        kind, name = head

        signature, body_source = _split_signature_and_body(decl_source)
        informal_statement = _extract_informal_field(attr_source, "statement")
        informal_proof = _extract_informal_field(attr_source, "proof")

        depends_on: list[str] = []
        sorry_start = sorry_end = -1
        is_target = False
        if kind in _TARGET_KINDS:
            deps, ss, se = _extract_sorry_using(decl_source, decl_start)
            if ss >= 0:
                depends_on = deps
                sorry_start = ss
                sorry_end = se
                is_target = True
            # else: theorem without sorry_using (already proved?) —
            # treat as definition-like (include verbatim as dep)

        node = BlueprintNode(
            name=name,
            kind=kind,
            is_target=is_target,
            raw_source=raw_source,
            attr_source=attr_source,
            decl_source=decl_source,
            signature=signature,
            body_source=body_source,
            informal_statement=informal_statement,
            informal_proof=informal_proof,
            depends_on=depends_on,
            attr_start=attr_start,
            decl_start=decl_start,
            decl_end=decl_end,
            sorry_start=sorry_start,
            sorry_end=sorry_end,
        )
        nodes.append(node)
        i = decl_end

    return header, nodes


# ---------------------------------------------------------------------------
#  Comment handling
# ---------------------------------------------------------------------------

def check_balanced_comments(code: str) -> tuple[bool, str | None]:
    """Pre-check: count `/-` / `-/` depth respecting strings and line
    comments. Returns `(True, None)` if balanced, or
    `(False, error_message)` if the code ends with unbalanced open
    block comments.

    This catches the failure mode where the model writes a malformed
    doc-comment like `/-- foo` that has no matching `-/` — the regular
    sanitizer `strip_lean_comments_and_strings` silently swallows the
    rest of the file in that case, making downstream checks like
    `check_theorem_present` return False even when the theorem is
    present in the RAW source."""
    n = len(code)
    i = 0
    depth = 0
    open_count = 0
    close_count = 0
    while i < n:
        c = code[i]
        # String literal
        if c == '"':
            i = _skip_string(code, i)
            continue
        # Line comment
        if c == '-' and i + 1 < n and code[i + 1] == '-':
            while i < n and code[i] != '\n':
                i += 1
            continue
        # Block comment open
        if c == '/' and i + 1 < n and code[i + 1] == '-':
            depth += 1
            open_count += 1
            i += 2
            continue
        # Block comment close
        if c == '-' and i + 1 < n and code[i + 1] == '/':
            if depth > 0:
                depth -= 1
                close_count += 1
            # If depth == 0 here it's a stray `-/`, ignored (not a
            # comment close).
            i += 2
            continue
        i += 1
    if depth != 0:
        return False, (
            f"Unbalanced `/- ... -/` block comment. Found {open_count} "
            f"opening `/-` and {close_count} closing `-/`, leaving "
            f"{depth} comment(s) unterminated. This often happens when "
            f"a `/-- ... -/` doc-comment is missing its closing `-/`. "
            f"Check every `/-` has a matching `-/` before its "
            f"declaration."
        )
    return True, None


def strip_lean_comments_and_strings(code: str) -> str:
    """Remove comments and string literals, preserving line structure.

    NOTE: If the code has an unbalanced `/- ... -/` block comment,
    this function will greedily consume the rest of the file as
    comment content. Callers should `check_balanced_comments` first
    and handle the error before calling this."""
    result: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        if code[i] == '"':
            result.append(' ')
            i += 1
            while i < n and code[i] != '"':
                if code[i] == '\\' and i + 1 < n:
                    result.append(' ')
                    i += 1
                if i < n:
                    result.append('\n' if code[i] == '\n' else ' ')
                    i += 1
            if i < n:
                result.append(' ')
                i += 1
            continue
        if i + 1 < n and code[i] == '/' and code[i + 1] == '-':
            depth = 1
            result.append(' ')
            result.append(' ')
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and code[i] == '/' and code[i + 1] == '-':
                    depth += 1
                    result.append(' ')
                    result.append(' ')
                    i += 2
                elif i + 1 < n and code[i] == '-' and code[i + 1] == '/':
                    depth -= 1
                    result.append(' ')
                    result.append(' ')
                    i += 2
                else:
                    result.append('\n' if code[i] == '\n' else ' ')
                    i += 1
            continue
        if i + 1 < n and code[i] == '-' and code[i + 1] == '-':
            while i < n and code[i] != '\n':
                result.append(' ')
                i += 1
            continue
        result.append(code[i])
        i += 1
    return ''.join(result)


def check_theorem_present(code: str, theorem_name: str) -> bool:
    balanced, _ = check_balanced_comments(code)
    if not balanced:
        return False
    sanitised = strip_lean_comments_and_strings(code)
    # Lean 4 identifiers can include `'` as a continuation char. Python's
    # `\b` is a `\w`/non-`\w` transition and treats `'` as non-`\w`, so a
    # trailing `\b` would (a) miss `name` followed by space/colon (the
    # apostrophe→space transition is non-word→non-word, no boundary) and
    # (b) falsely match `name'_other` or `name'2`. Use a negative
    # lookahead on `[\w']` instead — same pattern is used for the other
    # theorem-name regexes here and across the prover stages.
    pattern = (
        r"\b(?:theorem|lemma)\s+" + re.escape(theorem_name) + r"(?![\w'])"
    )
    return bool(re.search(pattern, sanitised))


# ---------------------------------------------------------------------------
#  Axiom classification
# ---------------------------------------------------------------------------

FOUNDATIONAL_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def parse_axioms_from_result(result: dict) -> set[str]:
    axioms: set[str] = set()
    response = result.get("response", {})
    if response is None:
        return axioms
    for msg in response.get("messages", []) or []:
        if msg.get("severity") != "info":
            continue
        data = msg.get("data", "")
        m = re.search(r"depends on axioms: \[([^\]]*)\]", data)
        if m:
            axiom_str = m.group(1).strip()
            if axiom_str:
                axioms = set(a.strip() for a in axiom_str.split(","))
    return axioms


def classify_axioms(axioms: set[str]) -> tuple[str, set[str]]:
    """Bucket a set of axiom dependencies.

    Buckets (most-rejected first):
      USES_SORRY    — sorryAx in axioms (proof is incomplete)
      CUSTOM_AXIOM  — non-foundational axioms (extra returned)
      VACUOUS       — no axioms (e.g. proof by rfl)
      LEGIT         — exactly the foundational set
      PARTIAL       — strict, non-empty subset of foundational
    """
    if "sorryAx" in axioms:
        return "USES_SORRY", set()
    # ALLOW_NATIVE_DECIDE=1 admits Lean.ofReduceBool (the axiom native_decide
    # introduces) as foundational so honest native_decide proofs count as
    # solves. Kernel-trusted-via-compiler and benchmark-accepted;
    # verify_submission.py still FLAGS every such proof for human review.
    import os as _os
    foundational = FOUNDATIONAL_AXIOMS
    if _os.environ.get("ALLOW_NATIVE_DECIDE") == "1":
        foundational = FOUNDATIONAL_AXIOMS | {"Lean.ofReduceBool"}
    extra = axioms - foundational
    if extra:
        return "CUSTOM_AXIOM", extra
    if not axioms:
        return "VACUOUS", set()
    if axioms == foundational:
        return "LEGIT", set()
    return "PARTIAL", set()


# ---------------------------------------------------------------------------
#  Dependency block builder
# ---------------------------------------------------------------------------

def _make_sorry_body(signature: str) -> str:
    """Return `<signature> := by sorry` — canonical sorry'd form."""
    sig = signature.rstrip()
    return f"{sig} := by sorry"


def _inline_comment(text: str) -> str:
    """Turn a multi-line informal text into a single-line Lean comment
    ready to prepend to a declaration."""
    if not text:
        return ""
    single = re.sub(r'\s+', ' ', text.strip())
    return f"-- {single}"


def build_dependency_block(
    target_node: BlueprintNode,
    all_nodes: list[BlueprintNode],
    formal_header: str,
) -> tuple[str, str]:
    """Build the dependency context for a target node.

    Layout (same for prompt and rebuild):
        <formal_header>

        -- <def N informal>
        def P : ... := ...

        ... (all defs in source order) ...

        -- L1 (already proven; do not reprove)
        theorem L1 : T1 := by sorry

        ... (direct-parent lemmas, sorry'd) ...

    Returns `(prompt_block, rebuild_block)`. They are currently
    identical (the dep block has no model-facing/compile-facing
    divergence); the tuple is kept in case the prompt version should
    carry extra informal comments later."""
    parts = [formal_header.rstrip()]

    node_by_name = {n.name: n for n in all_nodes}

    # All definition-like nodes in source order. Target-kind nodes
    # (theorem / lemma) never land here — grafted and sorry_using alike,
    # they are addressed by the direct-parent loop below as `by sorry`
    # stubs. This keeps per-node context uniform across iterations: the
    # target node sees only the signature of each parent, never a
    # grafted real proof that happened to arrive from an earlier stage.
    for n in all_nodes:
        if n.name == target_node.name:
            continue
        if n.kind in _DEFINITION_KINDS:
            comment = _inline_comment(n.informal_statement)
            body = n.decl_source.strip()
            if comment:
                parts.append(f"{comment}\n{body}")
            else:
                parts.append(body)

    # Direct-parent theorem nodes, uniformly sorry'd regardless of
    # whether the input has them as `sorry_using` or already grafted.
    direct_deps = [
        node_by_name[d] for d in target_node.depends_on
        if d in node_by_name
        and node_by_name[d].kind in _TARGET_KINDS
    ]
    for dep in direct_deps:
        comment_text = dep.informal_statement or dep.name
        squashed = re.sub(r'\s+', ' ', comment_text.strip())
        comment = f"-- {squashed} (already proven; do not reprove)"
        sorry_body = _make_sorry_body(dep.signature)
        parts.append(f"{comment}\n{sorry_body}")

    block = "\n\n".join(parts)
    return block, block


# ---------------------------------------------------------------------------
#  Per-target dep-block compile check (blueprint acceptance gate)
#
#  Some blueprints compile cleanly as a whole file but the per-node context
#  the prover extracts from them is broken — e.g. a `noncomputable section
#  ... end` wrapper that `split_header_body` doesn't recognise as a header
#  line and silently drops, or a lemma referenced inside a def's body but
#  not itself a `@[blueprint]` target (so the dep block omits it and Lean
#  fires `Unknown identifier`). When that happens every prover attempt
#  forfeits against the same broken context and the refinement loop burns
#  its budget on a sample that can never succeed.
#
#  This check runs the same extraction the prover uses (parse +
#  `build_dependency_block`), appends `<sig> := by sorry`, and submits each
#  target's dep block to Lean. An error on any target means the prover would
#  face the same failure, so blueprint generation and refinement reject the
#  sample (after a few repair turns) instead of wasting downstream budget.
# ---------------------------------------------------------------------------

async def check_dep_blocks_compile(
    code: str,
    lean_server_url: str,
    lean_timeout: int,
    concurrency: int = 8,
) -> tuple[str, list[dict]] | None:
    """Return None if every target's dep block compiles cleanly under
    `:= by sorry`; otherwise return `(failing_target_name, errors)` for
    the first failure, where `errors` is the list of `severity=="error"`
    message dicts from Lean.

    Mirrors the extraction in `prover.run_problem` — including
    stripping `import Architect` from the header — so a failure here is
    exactly what the downstream prover would face.

    On Lean transport errors (gateway 5xx, timeout, etc.) the check
    treats that target as "no finding" rather than a failure: a
    transient Lean outage should not trash the skeleton.
    """
    try:
        header, nodes = parse_blueprint_file(code)
    except Exception:
        # If parsing fails the upstream graph-validity / alignment gate
        # would already have rejected the code; we shouldn't override
        # that here.
        return None
    header = "\n".join(
        ln for ln in header.split("\n") if ln.strip() != "import Architect"
    )
    targets = [n for n in nodes if n.is_target]
    if not targets:
        return None

    sem = asyncio.Semaphore(concurrency)

    async def check_one(node: "BlueprintNode") -> tuple[str, list[dict]]:
        _, rebuild = build_dependency_block(node, nodes, header)
        sorry_stub = f"{node.signature.rstrip()} := by sorry"
        safe = rebuild.rstrip() + "\n\n" + sorry_stub
        async with sem:
            try:
                resp = await check_lean_async(
                    safe, lean_server_url, lean_timeout,
                )
            except Exception:
                # Treat Lean outages as no-finding — let the outer
                # pipeline handle transient errors on its own retries.
                return node.name, []
        msgs = (resp.get("response") or {}).get("messages") or []
        errors = [m for m in msgs if m.get("severity") == "error"]
        return node.name, errors

    results = await asyncio.gather(*(check_one(t) for t in targets))
    for name, errors in results:
        if errors:
            return name, errors
    return None


# ---------------------------------------------------------------------------
#  Safe code rebuild + proof grafting
# ---------------------------------------------------------------------------

def extract_lean_code_from_text(text: str) -> str | None:
    m = re.search(r"```lean4?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _find_real_proof_by(text: str):
    """Find the `:= by` that starts the REAL proof body, skipping any
    `:= by` that appears as a ``let <name>[: <type>] := by <term>``
    construct inside the type signature.  Uses the same heuristic as
    `_split_signature_and_body`: skip any `:=` immediately preceded by
    ``let <name>[: <type>]``."""
    for m in re.finditer(r':=\s*by\b', text):
        pre = text[:m.start()].rstrip()
        if _LET_VALUE_COLON_EQ_RE.search(pre):
            continue
        return m
    return None


def graft_proof_body(
    model_code: str, node: BlueprintNode,
) -> tuple[str | None, str | None]:
    """Extract the model's tactic body for `node.name` and attach it
    to the node's canonical signature (from the skeleton).

    Returns `(grafted_declaration, proof_body)`. `grafted_declaration`
    is the full theorem ready to compile: `<signature> := by <body>`.
    `proof_body` is just the tactic block after `:= by` (for final
    assembly substitution). Returns `(None, None)` on failure."""
    # `(?![\w'])` instead of `\b` — see check_theorem_present for why.
    pattern = (
        r"\b(?:theorem|lemma)\s+" + re.escape(node.name) + r"(?![\w'])"
    )
    m = re.search(pattern, model_code)
    if not m:
        # Fallback: last theorem/lemma keyword
        matches = list(re.finditer(
            r'\b(?:theorem|lemma)\s+\S+', model_code,
        ))
        if not matches:
            return None, None
        m = matches[-1]
    rest = model_code[m.end():]
    by_match = _find_real_proof_by(rest)
    if not by_match:
        return None, None
    proof_body = rest[by_match.end():]
    # Drop trailing #print axioms
    proof_body = re.sub(
        r'\n*#print\s+axioms\s+\S+\s*$', '', proof_body,
    ).rstrip()
    # Strip leading whitespace/newlines
    proof_body = proof_body.lstrip('\n').rstrip()
    if not proof_body:
        return None, None
    sig = node.signature.rstrip()
    grafted = f"{sig} := by\n{_reindent_tactics(proof_body, '  ')}"
    return grafted, proof_body


def _reindent_tactics(body: str, target_indent: str) -> str:
    """Strip common leading whitespace from `body` and re-indent every
    non-empty line with `target_indent`."""
    # Dedent to remove whatever the model used
    dedented = textwrap.dedent(body)
    out_lines = []
    for line in dedented.split('\n'):
        if line.strip():
            out_lines.append(target_indent + line)
        else:
            out_lines.append('')
    return '\n'.join(out_lines)


def build_safe_code(
    model_code: str,
    rebuild_dependency_block: str,
    node: BlueprintNode,
) -> tuple[str | None, str | None]:
    """Rebuild safe compile code for a target node.

    Layout:
        <rebuild_dependency_block>  ← skeleton header + defs + sorry'd parents

        <grafted target declaration>  ← skeleton signature + model's proof

        #print axioms <name>

    Returns `(safe_code, proof_body)`.  The model's header (imports /
    set_option / open) is DISCARDED — the rebuild_dependency_block
    already carries the canonical skeleton header, and using it alone
    guarantees the node-level compile environment matches the final
    assembly.  Without this, the model can add ``open Finset`` and
    succeed locally while the assembled file fails with unqualified
    name errors."""
    grafted, proof_body = graft_proof_body(model_code, node)
    if grafted is None:
        return None, None

    parts = [rebuild_dependency_block.strip(), grafted]
    safe_code = "\n\n".join(parts)
    safe_code += f"\n\n#print axioms {node.name}\n"
    return safe_code, proof_body


def target_has_sorry(
    code: str, target_name: str, sorries: list[dict],
) -> bool:
    """Return True if any sorry appears on or after the target's
    declaration line."""
    for i, line in enumerate(code.split('\n'), 1):
        # `(?![\w'])` instead of `\b` — see check_theorem_present for why.
        if re.search(
            rf"\b(?:theorem|lemma)\s+{re.escape(target_name)}(?![\w'])",
            line,
        ):
            return any(
                s.get('pos', {}).get('line', 0) >= i for s in sorries
            )
    return False


def target_has_sorry_using(code: str, target_name: str) -> bool:
    """Return True if a `sorry_using` token appears on or after the
    target's declaration line.

    Unlike bare `sorry`, the Architect `sorry_using` macro elaborates
    via an axiomatic admit and does NOT emit a Lean sorry warning, so
    `analyse_result(...)['sorries']` won't catch it. We have to scan
    the source. Parents grafted into the dependency block come BEFORE
    the target's declaration line, so they're correctly excluded."""
    lines = code.split('\n')
    decl_line = -1
    for i, line in enumerate(lines, 1):
        # `(?![\w'])` instead of `\b` — see check_theorem_present for why.
        if re.search(
            rf"\b(?:theorem|lemma)\s+{re.escape(target_name)}(?![\w'])",
            line,
        ):
            decl_line = i
            break
    if decl_line < 0:
        return False
    return any(
        re.search(r'\bsorry_using\b', ln)
        for ln in lines[decl_line - 1:]
    )


# ---------------------------------------------------------------------------
#  Concurrency pool
# ---------------------------------------------------------------------------

class CountingSemaphore:
    """asyncio.Semaphore wrapper that exposes in_flight / idle counts.

    The accessor surface (`total`, `in_flight`, `idle`) lets callers
    introspect pool occupancy — useful for speculative-retry policies
    that launch extra parallel attempts when the pool has free slots.
    Behaves like a standard `asyncio.Semaphore` if no caller reads the
    counters.
    """

    def __init__(self, total: int):
        self._sem = asyncio.Semaphore(total)
        self._total = total
        self._in_flight = 0

    async def __aenter__(self):
        await self._sem.acquire()
        self._in_flight += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._in_flight -= 1
        self._sem.release()

    @property
    def total(self) -> int:
        return self._total

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def idle(self) -> int:
        return self._total - self._in_flight


# ---------------------------------------------------------------------------
#  Final assembly: textual in-place substitution
# ---------------------------------------------------------------------------

def replace_sorry_using(
    code: str, sorry_start: int, sorry_end: int, proof_body: str,
) -> str:
    """Replace `sorry_using [...]` at [sorry_start, sorry_end) with
    `proof_body`, preserving indentation.

    Two cases:
      1. sorry_using is on its own indented line under `:= by`:
         Replace the entire line, indent every body line to the
         sorry_using column.
      2. sorry_using is inline after `:= by` on the signature line:
         Put the first tactic on that same line (replacing sorry_using).
         For multi-line bodies, move everything to a new indented block
         below `:= by` so Lean sees a proper tactic sequence.
    """
    line_start = code.rfind('\n', 0, sorry_start) + 1
    leading = code[line_start:sorry_start]
    own_line = leading.strip() == ""

    body = proof_body.strip('\n').rstrip()
    if not body:
        body = "sorry"
    lines = body.split('\n')

    # Strip common leading whitespace from the proof body
    non_empty = [l for l in lines if l.strip()]
    if non_empty:
        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
        lines = [l[min_indent:] if l.strip() else l for l in lines]

    if own_line:
        # Case 1: sorry_using on its own line — indent to its column
        indent = leading
        parts_out = []
        for i, l in enumerate(lines):
            if l.strip():
                parts_out.append(indent + l)
            else:
                parts_out.append('')
        new_text = '\n'.join(parts_out)
        return code[:line_start] + new_text + code[sorry_end:]
    else:
        # Case 2: inline after `:= by` on the signature line.
        # For single-line proofs, just drop the tactic in place.
        if len(lines) == 1:
            return code[:sorry_start] + lines[0] + code[sorry_end:]
        # Multi-line: put everything on indented lines below `:= by`.
        # Find the `:= by` that precedes sorry_using on this line and
        # compute the indent from the line start.
        by_col = leading.rfind(':= by')
        if by_col >= 0:
            # Indent body to 2 spaces past the line start indent
            base_indent = len(leading) - len(leading.lstrip())
            indent = ' ' * (base_indent + 2)
        else:
            indent = ' ' * (sorry_start - line_start + 2)
        # Remove sorry_using and the `:= by ` prefix (keep `:= by\n`)
        # We replace from the `:= by sorry_using [...]` with `:= by\n<indented body>`
        by_pos = code.rfind(':= by', line_start, sorry_start)
        if by_pos >= 0:
            cut_start = by_pos + len(':= by')
        else:
            cut_start = sorry_start
        parts_out = []
        for l in lines:
            if l.strip():
                parts_out.append(indent + l)
            else:
                parts_out.append('')
        new_text = '\n' + '\n'.join(parts_out)
        return code[:cut_start] + new_text + code[sorry_end:]


def assemble_final_file(
    skeleton_code: str,
    target_nodes: list[BlueprintNode],
    node_results: dict[str, dict],
) -> str:
    """Produce the final proved blueprint file by in-place substitution.

    For every successfully-proved target node, replace its
    `sorry_using [...]` token with the real tactic body. Definitions
    and `@[blueprint ...]` attributes are preserved byte-for-byte from
    the skeleton."""
    # Sort by sorry_start DESCENDING so earlier substitutions don't
    # invalidate later offsets.
    targets_sorted = sorted(
        target_nodes, key=lambda n: n.sorry_start, reverse=True,
    )
    code = skeleton_code
    for node in targets_sorted:
        result = node_results.get(node.name, {})
        if not result.get("success"):
            continue
        proof_body = result.get("proof_body")
        if not proof_body:
            # Try to extract from the final grafted code
            grafted = result.get("grafted")
            if grafted:
                g_proof = None
                by_match = re.search(r':=\s*by\b', grafted)
                if by_match:
                    g_proof = grafted[by_match.end():].rstrip()
                    g_proof = re.sub(
                        r'\n*#print\s+axioms\s+\S+\s*$', '', g_proof,
                    ).rstrip()
                    g_proof = g_proof.lstrip('\n')
                proof_body = g_proof
        if not proof_body:
            continue
        if node.sorry_start < 0 or node.sorry_end < 0:
            continue
        code = replace_sorry_using(
            code, node.sorry_start, node.sorry_end, proof_body,
        )
    return code


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------

def get_theorem_name(formal_statement: str) -> str | None:
    m = re.search(r'^\s*theorem\s+(\S+)', formal_statement, re.MULTILINE)
    if m:
        name = m.group(1)
        # Strip any trailing punctuation (e.g., binder `(`)
        name = re.split(r'[\s(:,\[{]', name, 1)[0]
        return name
    return None


def load_skeleton_traces(
    skeleton_results_dir: Path,
) -> dict[str, dict]:
    """Read a skeleton prover's `traces.jsonl` and return a dict
    `{problem_id: best_trace}` picking the FIRST successful sample
    per problem. A trace's payload includes `final_code`, `sample`,
    `formal_statement`, and metadata from the skeleton run."""
    traces_path = skeleton_results_dir / "traces.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(
            f"Skeleton traces not found: {traces_path}"
        )

    chosen: dict[str, dict] = {}
    seen_any: set[str] = set()
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            pid = t.get("problem_id")
            if not pid:
                continue
            seen_any.add(pid)
            if not t.get("success"):
                continue
            if pid in chosen:
                # Prefer the earliest successful sample
                existing = chosen[pid]
                if t.get("sample", 0) < existing.get("sample", 0):
                    chosen[pid] = t
            else:
                chosen[pid] = t
    log.info(
        "Loaded skeleton traces: %d problems total, %d with a "
        "successful sample",
        len(seen_any), len(chosen),
    )
    return chosen


# ---------------------------------------------------------------------------
#  Dead-node pruning
#
#  Nodes that the main theorem does not (transitively) use are dropped
#  after each stage. Two analyzers compute reachability:
#
#    * `analyze_blueprint_from_trace` (used after proving): builds the dep
#      graph from the prover's own per-node records plus a
#      Lean-identifier-aware regex scan of each node's declaration text and
#      proof body. This avoids a blind spot of `Architect.collectUsed`, which
#      walks the elaborated `Expr` only and misses lemma references inside
#      `simp [X]` / `simpa [X]` / `simp only [X]`.
#    * `analyze_skeleton` (used after blueprint generation and refinement):
#      asks Lean via LeanArchitect's `collectUsed` on the compiled file.
# ---------------------------------------------------------------------------

# Marker used by the appended `#eval` to delimit the dep-graph JSON in
# the Lean server's `info` message stream.
_DEPS_MARKER = "ARCHITECT_DEPS:"

_DEPS_PROBE = '''

open Lean Architect in
#eval show CoreM Unit from do
  let env ← getEnv
  let mut deps : Array (Name × Array Name × Array Name) := #[]
  for (n, _) in blueprintExt.getEntries env do
    let (typeU, valU) ← collectUsed n
    deps := deps.push (n, typeU.toArray, valU.toArray)
  IO.println s!"''' + _DEPS_MARKER + '''{(toJson deps).compress}"
'''


@dataclass
class AnalysisResult:
    """Summary of reachability analysis on a blueprint partial assembly.

    All fields are sets of node names. `effective_deps` maps every node
    name to its computed dep set (debug / logging aid).
    """

    reachable: set[str]         = field(default_factory=set)
    unsolved_targets: set[str]  = field(default_factory=set)
    dead: set[str]              = field(default_factory=set)
    frozen: set[str]            = field(default_factory=set)
    effective_deps: dict[str, set[str]] = field(default_factory=dict)


_IDENT_CHAR = r"[\w']"  # Lean identifier chars: \w plus prime


def strip_lean_comments(text: str) -> str:
    """Strip `--`, `/- ... -/`, and `/-- ... -/` from `text`.

    Block comments nest (Lean spec); depth is tracked. Line comments
    end at the next newline. Removed regions are dropped entirely;
    positions don't matter — we only care about identifier matches.
    """
    if not text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        if depth == 0:
            if text[i:i + 2] == "/-":
                depth = 1
                i += 2
                continue
            if text[i:i + 2] == "--":
                eol = text.find("\n", i)
                if eol < 0:
                    break
                i = eol
                continue
            out.append(text[i])
            i += 1
        else:
            if text[i:i + 2] == "-/":
                depth -= 1
                i += 2
                continue
            if text[i:i + 2] == "/-":
                depth += 1
                i += 2
                continue
            i += 1
    return "".join(out)


def _regex_used_names(text: str, candidates: set[str]) -> set[str]:
    """Subset of `candidates` that appear as Lean identifiers in `text`.

    Uses negative lookaround on `[\\w']` so names ending in `'` (e.g.
    `foo'`) match correctly — Python's `\\b` treats `'` as non-word and
    would fail at the boundary.

    `.` is a namespace separator, not an identifier char, so `M.foo` IS
    a reference to `foo` and we match it. False positives from string
    literals (we strip comments but not strings) only over-keep, never
    drop, so they're safe.
    """
    if not text:
        return set()
    text = strip_lean_comments(text)
    out: set[str] = set()
    for name in candidates:
        pat = rf"(?<!{_IDENT_CHAR}){re.escape(name)}(?!{_IDENT_CHAR})"
        if re.search(pat, text):
            out.add(name)
    return out


def analyze_blueprint_from_trace(
    trace: dict,
    main_name: str,
    *,
    skeleton: str | None = None,
) -> AnalysisResult:
    """Compute reachability from `main_name` using the prover's trace.

    For every blueprint node parsed from `skeleton` (or
    `trace['skeleton_code']` / `trace['upstream_skeleton']` if `skeleton`
    is None), the effective dep set is the union of blueprint-node names
    that textually appear in:

      - the declaration's signature (skeleton's
        attr_start..sorry_start for targets, attr_start..decl_end for
        non-targets such as `def`), and
      - the prover's `proof_body` for that node, if any.

    When a target node has no `proof_body` (prover failed) but does have
    a declared `depends_on` list (from `sorry_using [...]`), fall back
    to the declared list — it's the most conservative answer and
    matches what the skeleton author intended.

    Raises ValueError if `main_name` isn't a parsed node.
    """
    skel = skeleton or trace.get("skeleton_code") or trace.get("upstream_skeleton") or ""
    nr = trace.get("node_results") or {}

    try:
        _, parsed = parse_blueprint_file(skel)
    except Exception:
        parsed = []
    name_to_node = {n.name: n for n in parsed}
    all_names: set[str] = set(name_to_node.keys()) | set(nr.keys())

    if main_name not in all_names:
        raise ValueError(
            f"analyze_blueprint_from_trace: main `{main_name}` not in "
            f"parsed nodes (got {sorted(all_names)})"
        )

    solved_set: set[str] = set()
    unsolved_set: set[str] = set()
    for n in parsed:
        if not n.is_target:
            solved_set.add(n.name)
        elif isinstance(nr.get(n.name), dict) and nr[n.name].get("success"):
            solved_set.add(n.name)
        else:
            unsolved_set.add(n.name)

    effective: dict[str, set[str]] = {}
    for name in all_names:
        candidates = all_names - {name}
        node = name_to_node.get(name)

        sig_text = ""
        if node is not None:
            if node.is_target and node.sorry_start >= 0:
                sig_text = skel[node.attr_start:node.sorry_start]
            else:
                sig_text = skel[node.attr_start:node.decl_end]

        r = nr.get(name) if isinstance(nr.get(name), dict) else None
        body = (r or {}).get("proof_body") or ""
        scan_text = (sig_text + "\n" + body) if (sig_text or body) else ""

        is_target = node.is_target if node is not None else (name in nr)
        declared_raw = (r or {}).get("depends_on")
        body_attempted = bool(body)

        if scan_text:
            effective[name] = _regex_used_names(scan_text, candidates)
        elif is_target and declared_raw is not None and not body_attempted:
            effective[name] = set(declared_raw) & candidates
        else:
            effective[name] = candidates

    reachable: set[str] = set()
    stack = [main_name]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for d in effective.get(cur, ()):
            if d not in reachable:
                stack.append(d)

    return AnalysisResult(
        reachable=reachable,
        unsolved_targets=reachable & unsolved_set,
        dead=all_names - reachable,
        frozen=reachable & solved_set,
        effective_deps=effective,
    )


def prune_dead_nodes_from_trace(
    *,
    trace: dict,
    code: str,
    main_theorem_name: str,
    skeleton: str | None = None,
    solved_nodes: list[str] | None = None,
    unsolved_nodes: list[str] | None = None,
) -> tuple[str, str | None, list[str] | None, list[str] | None, AnalysisResult | None]:
    """Trace-based counterpart to `prune_dead_nodes`. Mirrors the same
    return shape so it's a drop-in for callers that have a trace dict
    in scope. Doesn't need a Lean roundtrip.
    """
    if not code or not main_theorem_name:
        return (code, skeleton, solved_nodes, unsolved_nodes, None)

    try:
        ana = analyze_blueprint_from_trace(
            trace, main_theorem_name, skeleton=skeleton,
        )
    except Exception:
        return (code, skeleton, solved_nodes, unsolved_nodes, None)

    if not ana.dead:
        return (code, skeleton, solved_nodes, unsolved_nodes, ana)

    code_pruned = prune_skeleton(code, ana.dead)
    skel_pruned = (
        prune_skeleton(skeleton, ana.dead) if skeleton is not None else None
    )
    solved_pruned = (
        [n for n in solved_nodes if n not in ana.dead]
        if solved_nodes is not None else None
    )
    unsolved_pruned = (
        [n for n in unsolved_nodes if n not in ana.dead]
        if unsolved_nodes is not None else None
    )
    return (code_pruned, skel_pruned, solved_pruned, unsolved_pruned, ana)


def _extract_deps_marker(messages: list[dict]) -> str | None:
    """Return the first `ARCHITECT_DEPS:...` payload found in the
    gateway's message stream, or None."""
    for m in messages:
        data = m.get("data") or ""
        if data.startswith(_DEPS_MARKER):
            return data[len(_DEPS_MARKER):]
        # Some REPL versions wrap stdout in extra prefix lines; try a
        # secondary search for the marker anywhere inside.
        idx = data.find(_DEPS_MARKER)
        if idx >= 0:
            return data[idx + len(_DEPS_MARKER):].splitlines()[0]
    return None


def _fetch_lean_dep_graph(
    code: str,
    lean_server_url: str,
    timeout: int,
) -> dict[str, set[str]]:
    """Send `code + dep-probe elab` to the gateway and return a
    {name -> dep-name-set} map sourced from `Architect.collectUsed`.

    Raises RuntimeError if the file fails to compile (the probe never
    runs) or the marker is missing (Architect API drift / wrong gateway).
    """
    augmented = code + _DEPS_PROBE
    result = check_lean_sync(augmented, lean_server_url, timeout)
    response = result.get("response") or {}
    messages = response.get("messages") or []

    # Lean recovers past elaboration errors and may still run the
    # appended `#eval`, so a present marker does NOT imply a successful
    # compile. Treat any error-severity message as a hard stop — the
    # resulting dep graph would be incomplete (failing nodes never get
    # registered with `blueprintExt`) and silently feeding it to the
    # pruner would strip valid nodes.
    errs = [m for m in messages if m.get("severity") == "error"]
    if errs:
        preview = (errs[0].get("data") or "")[:200].replace("\n", " ")
        raise RuntimeError(
            f"analyze_skeleton: Lean compile failed ({len(errs)} "
            f"error(s)); first: {preview!r}. Prune must be called "
            f"post-success — the dep graph from a partially-elaborated "
            f"file is unreliable."
        )

    payload = _extract_deps_marker(messages)
    if payload is None:
        raise RuntimeError(
            "analyze_skeleton: gateway returned no ARCHITECT_DEPS marker "
            "and no compile errors. Check that the Architect package "
            "exposes `Architect.blueprintExt` and `Architect.collectUsed` "
            "(public meta) and that the Lean server's project includes LeanArchitect."
        )

    raw = json.loads(payload)
    # Wire format: each entry is `[name, [typeU_array, valU_array]]`
    # because Lean's `(Name × Array Name × Array Name)` right-associates
    # and `ToJson` on a product emits `[a, b]`.
    deps: dict[str, set[str]] = {}
    for entry in raw:
        name, (typeU, valU) = entry[0], entry[1]
        deps[name] = set(typeU) | set(valU)
    return deps


def analyze_skeleton(
    partial_assembly: str,
    main_name: str,
    *,
    lean_server_url: str,
    timeout: int = 300,
) -> AnalysisResult:
    """Compute reachability from the main theorem.

    Edges come from `Architect.collectUsed` evaluated on the live Lean
    environment (the input is sent to `lean_server_url` with a small
    `#eval` block appended that prints the dep graph as JSON). Both the
    statement (typeUsed) and the proof body (valueUsed) contribute
    edges; axioms / non-blueprint constants are filtered out by
    intersecting with the parsed-node name set.

    Raises ValueError if `main_name` is absent from the parsed nodes.
    Raises RuntimeError if the gateway compile fails or the dep-probe
    marker is missing — there is intentionally no string-scrape
    fallback so latent disagreements between Lean and Python surface
    immediately rather than masquerade as silent prune bugs.
    """
    _, nodes = parse_blueprint_file(partial_assembly)
    all_names = {n.name for n in nodes}

    if main_name not in all_names:
        raise ValueError(
            f"analyze_skeleton: main theorem `{main_name}` not found in "
            f"partial assembly (parsed nodes: {sorted(all_names)})"
        )

    solved_set:   set[str] = {n.name for n in nodes if not n.is_target}
    unsolved_set: set[str] = {n.name for n in nodes if n.is_target}

    raw_graph = _fetch_lean_dep_graph(
        partial_assembly, lean_server_url, timeout,
    )

    # Filter to known blueprint nodes only — drop axioms (`Quot.sound`,
    # `Classical.choice`, `propext`, `sorryAx`, …) and any other
    # non-blueprint constants Lean traversed transitively.
    effective: dict[str, set[str]] = {}
    for name in all_names:
        deps = raw_graph.get(name, set())
        effective[name] = (deps & all_names) - {name}

    reachable: set[str] = set()
    stack: list[str] = [main_name]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for dep in effective.get(cur, ()):
            if dep not in reachable:
                stack.append(dep)

    return AnalysisResult(
        reachable=reachable,
        unsolved_targets=reachable & unsolved_set,
        dead=all_names - reachable,
        frozen=reachable & solved_set,
        effective_deps=effective,
    )


def _absorb_leading_comments(code: str, attr_start: int, lower_bound: int) -> int:
    """Return the offset where deletion of a node should begin so that
    blank lines and immediately-preceding `--` line comments and
    `/- ... -/` / `/-- ... -/` block comments are removed along with the
    declaration. Walks line-by-line from `lower_bound` to `attr_start`,
    classifying each line as code or comment-only, and returns the start
    of the longest comment-only suffix. The bp parser already trims
    leading comments out of the *previous* node's `decl_end`, so passing
    `prev.decl_end` (or 0 for the first node) as `lower_bound` keeps us
    from clobbering live content.
    """
    if lower_bound >= attr_start:
        return attr_start

    attr_line_start = code.rfind("\n", 0, attr_start) + 1
    if attr_line_start <= lower_bound:
        return attr_start

    # Snap lower_bound up to the next line boundary — we only absorb
    # whole lines. (lower_bound==0 is already at a line start.)
    if lower_bound == 0:
        line_lower = 0
    else:
        nl = code.find("\n", lower_bound, attr_line_start)
        if nl < 0:
            return attr_start
        line_lower = nl + 1
        if line_lower >= attr_line_start:
            return attr_start

    line_starts: list[int] = []
    pos = line_lower
    while pos < attr_line_start:
        line_starts.append(pos)
        nl = code.find("\n", pos, attr_line_start)
        if nl < 0:
            break
        pos = nl + 1
    line_starts.append(attr_line_start)  # sentinel

    block_depth = 0
    has_code: list[bool] = []
    for i in range(len(line_starts) - 1):
        s = line_starts[i]
        e = line_starts[i + 1] - 1  # exclude trailing '\n'
        if e < s:
            e = s
        line = code[s:e]
        line_has_code = False
        j = 0
        while j < len(line):
            if block_depth > 0:
                nxt_open = line.find("/-", j)
                nxt_close = line.find("-/", j)
                if nxt_close < 0:
                    j = len(line)
                elif nxt_open < 0 or nxt_close < nxt_open:
                    block_depth -= 1
                    j = nxt_close + 2
                else:
                    block_depth += 1
                    j = nxt_open + 2
            else:
                nxt_open = line.find("/-", j)
                nxt_lc = line.find("--", j)
                if nxt_open >= 0 and (nxt_lc < 0 or nxt_open <= nxt_lc):
                    if line[j:nxt_open].strip():
                        line_has_code = True
                    block_depth += 1
                    j = nxt_open + 2
                elif nxt_lc >= 0:
                    if line[j:nxt_lc].strip():
                        line_has_code = True
                    j = len(line)  # rest of line is `--` line comment
                else:
                    if line[j:].strip():
                        line_has_code = True
                    j = len(line)
        has_code.append(line_has_code)

    k = len(has_code)
    while k > 0 and not has_code[k - 1]:
        k -= 1
    return line_starts[k]


def prune_skeleton(code: str, dead_names: set[str]) -> str:
    """Delete dead nodes and scrub dead names from surviving
    `sorry_using [...]` lists.

    Every surviving target's `sorry_using` list must be scrubbed,
    otherwise the Architect `sorry_using` macro fails to elaborate
    against a now-undefined name.

    Each dead node's deletion range is extended backward to absorb any
    immediately-preceding doc-comment block (`/-- ... -/`), ordinary
    block comment, or `--` line comments — otherwise those orphan lines
    sit in front of an unrelated declaration (or the next attribute) and
    fail to parse.

    Informal natural-language text is untouched.

    Idempotent: `prune_skeleton(code, set())` returns `code`.
    """
    if not dead_names:
        return code

    _, nodes = parse_blueprint_file(code)

    sorted_nodes = sorted(nodes, key=lambda n: n.attr_start)
    edits: list[tuple[int, int, str]] = []

    for idx, n in enumerate(sorted_nodes):
        if n.name in dead_names:
            lower = sorted_nodes[idx - 1].decl_end if idx > 0 else 0
            start = _absorb_leading_comments(code, n.attr_start, lower)
            edits.append((start, n.decl_end, ""))
        elif n.is_target and n.sorry_start >= 0:
            kept = [d for d in n.depends_on if d not in dead_names]
            if len(kept) != len(n.depends_on):
                new_tok = f"sorry_using [{', '.join(kept)}]"
                edits.append((n.sorry_start, n.sorry_end, new_tok))

    edits.sort(key=lambda e: -e[0])
    out = code
    for start, end, repl in edits:
        out = out[:start] + repl + out[end:]

    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def prune_dead_nodes(
    *,
    code: str,
    main_theorem_name: str,
    lean_server_url: str,
    skeleton: str | None = None,
    solved_nodes: list[str] | None = None,
    unsolved_nodes: list[str] | None = None,
    timeout: int = 300,
) -> tuple[str, str | None, list[str] | None, list[str] | None, AnalysisResult | None]:
    """Reachability-prune `code` and any companion artifacts.

    Reachability is computed against `code` (the most-information-rich
    form — pass a partial assembly with grafted bodies when available;
    otherwise pass the pure skeleton itself). The resulting dead set is
    applied uniformly to `code`, `skeleton` (if given), and the optional
    name lists, keeping all four artifacts in lockstep.

    `lean_server_url` is required: edges come from the Lean gateway via
    `Architect.collectUsed`. Call this only after the input compiled
    cleanly — see `analyze_skeleton` for the failure modes.

    Returns: (code', skeleton', solved', unsolved', analysis).
    `analysis` is None when inputs are empty; otherwise an
    AnalysisResult — empty `dead` means everything was already reachable
    and the inputs are returned unchanged.

    Raises ValueError if `main_theorem_name` is absent from the parsed
    nodes; bp parser and gateway exceptions propagate. Call sites
    should wrap in try/except so a parse hiccup never wedges the
    surrounding stage.
    """
    if not code or not main_theorem_name:
        return (code, skeleton, solved_nodes, unsolved_nodes, None)

    ana = analyze_skeleton(
        code, main_name=main_theorem_name,
        lean_server_url=lean_server_url, timeout=timeout,
    )

    if not ana.dead:
        return (code, skeleton, solved_nodes, unsolved_nodes, ana)

    code_pruned = prune_skeleton(code, ana.dead)
    skel_pruned = (
        prune_skeleton(skeleton, ana.dead) if skeleton is not None else None
    )
    solved_pruned = (
        [n for n in solved_nodes if n not in ana.dead]
        if solved_nodes is not None else None
    )
    unsolved_pruned = (
        [n for n in unsolved_nodes if n not in ana.dead]
        if unsolved_nodes is not None else None
    )
    return (code_pruned, skel_pruned, solved_pruned, unsolved_pruned, ana)
