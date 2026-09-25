"""Blueprint generation (paper §3.1).

Given the formal statement of the target theorem, the model designs a
dependency graph of definitions and lemmas and emits it as ONE Lean file in
which every node is a `@[blueprint]`-annotated declaration (LeanArchitect).
Lemma and theorem bodies stay unproved as `:= by sorry_using [parents]`,
which records the graph's edges; the main theorem keeps the original
signature. Optionally, a natural-language proof guides the decomposition
(`--nl-proof-decompose`).

The model iterates against the Lean compiler through a `lean_compile` tool
(and optionally `mathlib_search`). A blueprint is accepted only when:

  * pre-compile safeguards pass (balanced comments, required imports,
    canonical main signature, every theorem/lemma annotated and bodied with
    `sorry_using`, no `axiom` / `opaque` / `native_decide`);
  * the file compiles;
  * the graph is valid (statement/proof fields present, deps resolve, no
    cycles or self-loops, one main theorem, every node reachable from it);
  * every lemma's sliced per-node context compiles on its own (the exact
    context the prover will see).

Accepted blueprints are pruned of nodes unreachable from the main theorem.

Input:  JSONL rows with `problem_id`, `formal_statement` (and `nl_proof` in
        `--nl-proof-decompose` mode).
Output: `traces.jsonl` (one row per sample; `final_code` / `final_skeleton`
        hold the blueprint), `summary.json`, `session.log` under `--output`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from . import prompts
from .graph import (
    check_dep_blocks_compile,
    extract_lean_code_from_text,
    parse_blueprint_file,
    pass_at_k,
    prune_dead_nodes,
    strip_lean_comments_and_strings,
)
from .lean import (
    DEFAULT_LEAN_SERVER,
    analyse_result,
    check_lean_async,
    extract_tool_code_arg,
    format_compile_response,
)
from .llm import (
    DEFAULT_MAX_MODEL_LEN,
    add_model_args,
    build_extra_body,
    chat_with_retry,
    compute_max_tokens,
    count_prompt_tokens,
    default_headers,
    estimate_tools_overhead,
    load_tokenizer,
    read_api_key,
    setup_logger,
)
from .search import (
    DEFAULT_SEARCH_SERVER,
    MATHLIB_SEARCH_TOOL,
    format_search_response,
    resolve_search_server,
    search_mathlib_async,
)

# Fixed name (not __name__): when run with `python -m`, __name__ is "__main__".
log = logging.getLogger("goedel_architect.blueprint")


# ---------------------------------------------------------------------------
#  Tool definitions
#
#  The `lean_compile` description below is kept verbatim from the runs
#  reported in the paper. It still mentions the "input DAG" of an earlier
#  pipeline variant; the model sees it in both modes of this stage.
# ---------------------------------------------------------------------------

LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_compile",
        "description": (
            "Compile a Lean 4 code snippet against Mathlib + LeanArchitect and return compilation feedback. The code must be a complete, self-contained Lean 4 file including all imports (e.g. `import Mathlib` and `import Architect`). Returns whether compilation succeeded and any error messages with their positions; on a clean compile, also runs an alignment check against the input DAG and reports any mismatches (missing node, kind mismatch, or `sorry_using` deps that differ from the DAG's `deps`). For the SKELETON stage, sorries from `sorry_using` are EXPECTED and do not count as failure — only real compile errors and DAG-alignment mismatches do."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Lean 4 LeanArchitect blueprint code. Must include `import Mathlib` and `import Architect`. Each node from the input DAG must appear as a bare `@[blueprint]`-annotated declaration with the same `name` as the DAG node (used as the Lean identifier); every theorem/lemma must end in `:= by sorry_using [d1, d2, ...]` listing the DAG node's `deps`. Do NOT add `(statement := ...)` or `(proof := ...)` fields to the `@[blueprint]` annotations — those are post-processed in from the DAG. The main theorem must use the exact original Lean signature provided in the user prompt."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

TOOLS = [LEAN_COMPILE_TOOL, MATHLIB_SEARCH_TOOL]


# ---------------------------------------------------------------------------
#  System prompts (goedel_architect/prompts/blueprint_generation*.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SELF_DECOMPOSE = prompts.load("blueprint_generation")
SYSTEM_PROMPT_NL_PROOF_DECOMPOSE = prompts.load("blueprint_generation_nl_guided")

# Extra edit (beyond dropping the `mathlib_search` paragraph) applied when
# the run has no Mathlib search; see `prompts.without_search`.
_NO_SEARCH_EDITS = (
    (" — if it's an `Unknown identifier` / `Unknown constant` error, "
     "**always call `mathlib_search` first to discover the correct name** — "
     "and call `lean_compile` again.",
     " and call `lean_compile` again."),
)


def system_prompt_for(nl_proof_decompose: bool, with_search: bool) -> str:
    """System prompt for the requested mode, with or without search."""
    prompt = (SYSTEM_PROMPT_NL_PROOF_DECOMPOSE if nl_proof_decompose
              else SYSTEM_PROMPT_SELF_DECOMPOSE)
    if with_search:
        return prompt
    return prompts.without_search(prompt, _NO_SEARCH_EDITS)


# ---------------------------------------------------------------------------
#  Helpers: theorem name, headers, comment stripping
# ---------------------------------------------------------------------------

def get_theorem_name(formal_statement: str) -> str | None:
    m = re.search(r"^\s*theorem\s+(\S+)", formal_statement, re.MULTILINE)
    return m.group(1) if m else None


def extract_theorem_line(formal_statement: str, theorem_name: str) -> str:
    # `(?![\w'])` instead of `\b` so apostrophe-ending Lean identifiers
    # match correctly. Python's `\b` is `\w`/non-`\w` and treats `'` as
    # non-`\w`, which both misses `name` followed by space (false negative)
    # and falsely matches `name'_other` (false positive). Same fix is
    # applied to every other theorem-name regex in this file.
    pattern = r"\btheorem\s+" + re.escape(theorem_name) + r"(?![\w'])"
    m = re.search(pattern, formal_statement)
    if m:
        return formal_statement[m.start():].strip()
    return formal_statement.strip()


def _is_header_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith("--"):
        return True
    return s.startswith(("import ", "set_option ", "open "))


def split_header_body(code: str) -> tuple[str, str]:
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if not _is_header_line(line):
            return "\n".join(lines[:i]), "\n".join(lines[i:])
    return code, ""


def find_unbalanced_block_comment(code: str) -> tuple[int, int, str] | None:
    """Return (line, col, kind) if `/- ... -/` nesting is unbalanced, else None.

    `kind` is 'unclosed' (an open with no matching close, pointing at the
    outermost unclosed open) or 'stray_close' (a `-/` with no matching open).
    Catches the common `/-)` vs `-/)` typo at field closers before
    `strip_lean_comments_and_strings` silently swallows the rest of the file
    and downstream checks misreport the real error as "missing theorem".
    """
    i = 0
    n = len(code)
    line = 1
    line_start = 0
    depth = 0
    open_stack: list[tuple[int, int]] = []
    while i < n:
        ch = code[i]
        if ch == '\n':
            line += 1
            line_start = i + 1
            i += 1
            continue
        if depth == 0:
            if ch == '"':
                i += 1
                while i < n and code[i] != '"':
                    if code[i] == '\\' and i + 1 < n:
                        if code[i + 1] == '\n':
                            line += 1
                            line_start = i + 2
                        i += 2
                        continue
                    if code[i] == '\n':
                        line += 1
                        line_start = i + 1
                    i += 1
                if i < n:
                    i += 1
                continue
            if i + 1 < n and ch == '-' and code[i + 1] == '-':
                while i < n and code[i] != '\n':
                    i += 1
                continue
        if i + 1 < n and ch == '/' and code[i + 1] == '-':
            open_stack.append((line, i - line_start))
            depth += 1
            i += 2
            continue
        if i + 1 < n and ch == '-' and code[i + 1] == '/':
            if depth == 0:
                return (line, i - line_start, 'stray_close')
            depth -= 1
            open_stack.pop()
            i += 2
            continue
        i += 1
    if depth > 0:
        l, c = open_stack[0]
        return (l, c, 'unclosed')
    return None


# ---------------------------------------------------------------------------
#  Unified safeguard rendering: every pre-compile rejection produces one or
#  more violation dicts of shape {"display": str, "message": str},
#  rendered into `Safeguard rejected with 1 violation.` /
#  `with N violations.` (singular/plural matched per the spec) + per-violation
#  `Violation N:` blocks containing a fenced lean4 code block (with
#  `<error></error>` markers around the offending span — synthetic for
#  absence cases) and a `Reason:` line. The status word is distinct
#  from `Compilation FAILED` so the model can tell pre-compile
#  rejection from a real Lean error.
# ---------------------------------------------------------------------------

def _highlight_line_span(line_text: str, col: int, end_col: int) -> str:
    """Wrap `line_text[col:end_col]` in `<error></error>` markers."""
    return (
        f"{line_text[:col]}<error>{line_text[col:end_col]}</error>"
        f"{line_text[end_col:]}"
    )


def _violation_at_offsets(code: str, start: int, end: int, message: str) -> dict:
    """Build a {display, message} violation dict highlighting `code[start:end]`
    on its source line. Spans crossing a newline are clipped to end-of-line."""
    line_start = code.rfind('\n', 0, start) + 1
    line_end_off = code.find('\n', start)
    if line_end_off == -1:
        line_end_off = len(code)
    line_text = code[line_start:line_end_off]
    col = start - line_start
    end_col = min(end - line_start, len(line_text))
    return {"display": _highlight_line_span(line_text, col, end_col),
            "message": message}


def _violation_synthetic(display: str, message: str) -> dict:
    """Build a violation dict with a pre-rendered display block. Pass an
    empty string for `display` to omit the top-level fenced block entirely
    (used when there's no real offending span and the required content is
    embedded in a nested ```lean4 block inside `message` instead)."""
    return {"display": display, "message": message}


def _render_violation_blocks(violations: list[dict]) -> str:
    """Render a list of {display, message} violations into the unified
    `Safeguard rejected ...` response. A falsy `display` skips the
    top-level fenced block — the violation reduces to `Violation N:` +
    `Reason: ...`, with any required code embedded inside the message."""
    n = len(violations)
    head = (f"Safeguard rejected with {n} violation."
            if n == 1
            else f"Safeguard rejected with {n} violations.")
    parts = [head, ""]
    for i, v in enumerate(violations, 1):
        parts.append(f"Violation {i}:")
        if v["display"]:
            parts.append("```lean4")
            parts.append(v["display"])
            parts.append("```")
        parts.append(f"Reason: {v['message']}")
        parts.append("")
    return '\n'.join(parts).rstrip()


# ---------------------------------------------------------------------------
#  Pre-compile checks. Each returns either None (passed) or a non-empty
#  list[dict] of {display, message} violations ready for
#  `_render_violation_blocks`.
# ---------------------------------------------------------------------------

def check_unbalanced_block_comment(code: str) -> list[dict] | None:
    info = find_unbalanced_block_comment(code)
    if info is None:
        return None
    line, col, kind = info
    code_lines = code.split('\n')
    line_idx = line - 1
    ctx_start = max(0, line_idx - 2)
    ctx_end = min(len(code_lines), line_idx + 3)
    rendered: list[str] = []
    for ln in range(ctx_start, ctx_end):
        if ln >= len(code_lines):
            break
        if ln == line_idx:
            # Highlight the offending 2-char `/-` or `-/` token.
            rendered.append(_highlight_line_span(code_lines[ln], col, col + 2))
        else:
            rendered.append(code_lines[ln])
    display = '\n'.join(rendered)
    if kind == 'unclosed':
        message = (
            f"Your code has an unclosed block comment `/-` opening at line "
            f"{line}, col {col}, with no matching `-/`. Everything after "
            f"this point is being treated as comment content, so any "
            f"theorem or declaration that follows is invisible to the "
            f"compiler. Find the offending `/-` and make sure every "
            f"`/--` or `/-` opening has a matching `-/` closing."
        )
    else:
        message = (
            f"Your code has a stray `-/` at line {line}, col {col} with "
            f"no matching `/-` opening before it. Either remove the extra "
            f"`-/` or add the missing `/-`."
        )
    return [_violation_synthetic(display, message)]


def check_theorem_present(code: str, theorem_name: str) -> bool:
    sanitised = strip_lean_comments_and_strings(code)
    pattern = r"\btheorem\s+" + re.escape(theorem_name) + r"(?![\w'])"
    return bool(re.search(pattern, sanitised))


def check_missing_theorem(
    code: str, theorem_name: str | None, formal_statement: str,
) -> list[dict] | None:
    if not theorem_name or check_theorem_present(code, theorem_name):
        return None
    orig_thm = extract_theorem_line(formal_statement, theorem_name)
    message = (
        f"Your code does not contain the required theorem "
        f"'{theorem_name}'. Add it at the end of the file with EXACTLY "
        f"this signature (body must be `:= by sorry_using [...]`):\n\n"
        f"```lean4\n{orig_thm}\n```"
    )
    return [_violation_synthetic("", message)]


_AXIOM_RE = re.compile(r'\baxiom\b')
_NATIVE_DECIDE_RE = re.compile(r'\bnative_decide\b')
# `opaque` introduces an abstract constant with no reducible body. In the
# skeleton stage it is functionally identical to `axiom` — a custom type
# left as `opaque Strategy : Sort u_1` compiles cleanly but every
# downstream blueprint node fails
# because the type can never be unfolded. Force the skeleton author to
# materialize the type with `def` / `structure` / `inductive` instead.
_OPAQUE_RE = re.compile(r'\bopaque\b')


def check_forbidden_constructs(code: str) -> list[dict] | None:
    """Reject `axiom` declarations and `native_decide` uses. Highlights
    each occurrence with `<error></error>` on its source line."""
    sanitised = strip_lean_comments_and_strings(code)
    violations: list[dict] = []
    for m in _AXIOM_RE.finditer(sanitised):
        violations.append(_violation_at_offsets(
            code, m.start(), m.end(),
            "'axiom' declarations are not allowed. The skeleton must be "
            "self-contained without axioms.",
        ))
    if os.environ.get("ALLOW_NATIVE_DECIDE") != "1":
        for m in _NATIVE_DECIDE_RE.finditer(sanitised):
            violations.append(_violation_at_offsets(
                code, m.start(), m.end(),
                "'native_decide' is not allowed. Use `decide`, `norm_num`, "
                "or other tactics instead.",
            ))
    for m in _OPAQUE_RE.finditer(sanitised):
        violations.append(_violation_at_offsets(
            code, m.start(), m.end(),
            "'opaque' declarations are not allowed in the skeleton — they "
            "leave the type with no reducible body, so downstream blueprint "
            "nodes that need to unfold it will fail. Materialize the type "
            "concretely with `def`, `structure`, or `inductive` instead.",
        ))
    return violations or None


def count_blueprint_decls(code: str) -> int:
    """Count `@[blueprint ...]` annotations (outside comments/strings)."""
    sanitised = strip_lean_comments_and_strings(code)
    return len(re.findall(r'@\[blueprint\b', sanitised))


def _normalise_signature(sig: str) -> str:
    return re.sub(r'\s+', ' ', sig.strip())


def _extract_main_signature(text: str, theorem_name: str) -> str | None:
    pattern = r"theorem\s+" + re.escape(theorem_name) + r"(?![\w'])"
    m = re.search(pattern, text)
    if not m:
        return None
    rest = text[m.start():]
    by_match = re.search(r':=\s*by\b', rest)
    if not by_match:
        return None
    return rest[:by_match.start()]


def _highlight_theorem_line(code: str, theorem_name: str) -> str | None:
    """Return the model's `theorem <name> ...` line with everything from
    the `theorem` keyword onward wrapped in `<error></error>`. Returns
    None if the keyword isn't found in `code`."""
    pattern = r"\btheorem\s+" + re.escape(theorem_name) + r"(?![\w'])"
    m = re.search(pattern, code)
    if not m:
        return None
    line_start = code.rfind('\n', 0, m.start()) + 1
    line_end = code.find('\n', m.end())
    if line_end == -1:
        line_end = len(code)
    line_text = code[line_start:line_end]
    col = m.start() - line_start
    return f"{line_text[:col]}<error>{line_text[col:]}</error>"


def check_main_signature_matches(
    code: str, formal_statement: str, theorem_name: str | None,
) -> list[dict] | None:
    if not theorem_name:
        return None
    orig_sig = _extract_main_signature(formal_statement, theorem_name)
    if orig_sig is None:
        return None  # can't verify — assume OK

    model_sig = _extract_main_signature(code, theorem_name)
    if model_sig is None:
        # The theorem keyword exists (else `check_missing_theorem` would
        # have fired earlier), but no `:= by` body. Highlight the line.
        display = (_highlight_theorem_line(code, theorem_name)
                   or f"<error>theorem {theorem_name} ... := by sorry_using [...]</error>")
        message = (
            f"theorem '{theorem_name}' (with `:= by ...` body) not found "
            f"in the code. The body must be `:= by sorry_using [...]`."
        )
        return [_violation_synthetic(display, message)]

    if _normalise_signature(orig_sig) != _normalise_signature(model_sig):
        display = (_highlight_theorem_line(code, theorem_name)
                   or f"<error>theorem {theorem_name} ...</error>")
        message = (
            f"theorem '{theorem_name}' signature differs from the "
            f"original signature.\n\n"
            f"Required:\n```lean4\n{orig_sig.strip()}\n```\n\n"
            f"Found in your code:\n```lean4\n{model_sig.strip()}\n```\n\n"
            f"You must use the original signature exactly. Do not rename "
            f"hypotheses, change types, or alter the conclusion."
        )
        return [_violation_synthetic(display, message)]
    return None


def check_imports(code: str) -> list[dict] | None:
    """Ensure the file imports both Mathlib and Architect. The Reason
    embeds the missing imports in a nested ```lean4 block — no top-level
    display is emitted (no real offending span exists)."""
    sanitised = strip_lean_comments_and_strings(code)
    has_mathlib = bool(re.search(r'^\s*import\s+Mathlib\b', sanitised, re.MULTILINE))
    has_architect = bool(re.search(r'^\s*import\s+Architect\b', sanitised, re.MULTILINE))
    if has_mathlib and has_architect:
        return None
    missing_lines: list[str] = []
    msg_parts: list[str] = []
    if not has_mathlib:
        missing_lines.append("import Mathlib")
        msg_parts.append("`import Mathlib`")
    if not has_architect:
        missing_lines.append("import Architect")
        msg_parts.append(
            "`import Architect` (required for `@[blueprint]` and `sorry_using`)"
        )
    missing_str = " and ".join(msg_parts)
    block = '\n'.join(missing_lines)
    message = (
        f"Missing {missing_str}. The file MUST start with both "
        f"`import Mathlib` and `import Architect`:\n\n"
        f"```lean4\n{block}\n```"
    )
    return [_violation_synthetic("", message)]


_BLUEPRINT_ATTR_START_RE = re.compile(r'@\[blueprint\b')
_DECL_KIND_RE = re.compile(r'\b(theorem|lemma|def|structure|abbrev|instance)\b(?:\s+([A-Za-z_][\w\']*))?')


def extract_blueprint_blocks(sanitised: str) -> list[dict]:
    """Enumerate `@[blueprint ...]` attribute blocks and the declaration that
    follows each one. Operates on already-sanitised code so that doc-comment
    content like `proof := "..."` inside `/-- ... -/` can't be mistaken for
    a real attribute field.
    """
    results: list[dict] = []
    n = len(sanitised)
    for m in _BLUEPRINT_ATTR_START_RE.finditer(sanitised):
        start = m.start()
        i = start + 2
        depth = 1
        while i < n and depth > 0:
            if sanitised[i] == '[':
                depth += 1
            elif sanitised[i] == ']':
                depth -= 1
            i += 1
        if depth != 0:
            break
        end = i
        block_text = sanitised[start:end]
        tail = sanitised[end:]
        kind = None
        name = None
        km = _DECL_KIND_RE.search(tail)
        if km:
            next_attr = tail.find('@[')
            if next_attr == -1 or km.start() < next_attr:
                kind = km.group(1)
                name = km.group(2)
        results.append({
            'start': start,
            'end': end,
            'block_text': block_text,
            'decl_kind': kind,
            'decl_name': name,
        })
    return results


# ---------------------------------------------------------------------------
#  Enforcement: no bare `sorry`, all theorems @[blueprint]-annotated
# ---------------------------------------------------------------------------

_THEOREM_DECL_RE = re.compile(
    r'\b(theorem|lemma)\s+([A-Za-z_][\w\']*)\b',
)


def _iter_theorem_bodies(sanitised: str) -> list[dict]:
    """Return list of {name, kind, decl_start, kw_start, kw_end,
    body_first_token, body_token_start, body_token_end} for every
    theorem/lemma declaration. Offsets are in the sanitised string, which
    `strip_lean_comments_and_strings` keeps byte-aligned with the
    original code so the same offsets work against `code`."""
    results: list[dict] = []
    for m in _THEOREM_DECL_RE.finditer(sanitised):
        kw_start = m.start()
        kw_end = m.end(2)  # end of the theorem name
        kind = m.group(1)
        name = m.group(2)
        tail_start = m.end()
        tail = sanitised[tail_start:]
        by_m = re.search(r':=\s*by\b', tail)
        if not by_m:
            continue
        i = tail_start + by_m.end()
        while i < len(sanitised) and sanitised[i] in ' \t\n':
            i += 1
        body_token_start = i
        tok_m = re.match(r"[A-Za-z_][\w']*", sanitised[i:])
        body_first_token = tok_m.group(0) if tok_m else ''
        body_token_end = body_token_start + len(body_first_token)
        results.append({
            'name': name,
            'kind': kind,
            'decl_start': kw_start,  # alias retained for annotation check
            'kw_start': kw_start,
            'kw_end': kw_end,
            'body_first_token': body_first_token,
            'body_token_start': body_token_start,
            'body_token_end': body_token_end,
        })
    return results


def check_no_bare_sorry(code: str) -> list[dict] | None:
    """Reject any theorem/lemma whose body is bare `sorry` (must be
    `sorry_using [...]`). Highlights the `sorry` token in place."""
    sanitised = strip_lean_comments_and_strings(code)
    violations: list[dict] = []
    for d in _iter_theorem_bodies(sanitised):
        if d['body_first_token'] != 'sorry':
            continue
        violations.append(_violation_at_offsets(
            code, d['body_token_start'], d['body_token_end'],
            (
                f"theorem `{d['name']}` uses bare `sorry` as its body. "
                f"Every theorem/lemma in the skeleton MUST end with "
                f"`:= by sorry_using [dep1, dep2, ...]` listing its "
                f"direct-parent dependencies (use `[]` for a theorem "
                f"that depends only on definitions). Bare `sorry` is "
                f"not allowed — it breaks DAG tracking and prevents "
                f"the blueprint prover from recognising the theorem "
                f"as a target."
            ),
        ))
    return violations or None


def check_body_is_sorry_using(code: str) -> list[dict] | None:
    """Reject any theorem/lemma whose `:= by` body opens with a real
    tactic instead of `sorry_using [...]`. Bare `sorry` is left to
    `check_no_bare_sorry`; this gate catches the case where the model
    inlines an actual proof (e.g. `:= by exact le_antisymm ...`).
    Highlights the offending leading token."""
    sanitised = strip_lean_comments_and_strings(code)
    violations: list[dict] = []
    for d in _iter_theorem_bodies(sanitised):
        first = d['body_first_token']
        if first in ('sorry', 'sorry_using'):
            continue
        # Empty token means body starts with punctuation (e.g. `(`); still
        # flag, highlighting one character so the marker is visible.
        end = d['body_token_end']
        if end == d['body_token_start']:
            end = d['body_token_start'] + 1
        shown = first or sanitised[d['body_token_start']:end]
        violations.append(_violation_at_offsets(
            code, d['body_token_start'], end,
            (
                f"`{d['kind']} {d['name']}` has a real proof body "
                f"starting with `{shown}` instead of `sorry_using [...]`. "
                f"Every theorem/lemma in the skeleton MUST end with "
                f"`:= by sorry_using [dep1, dep2, ...]` listing its "
                f"direct-parent dependencies (use `[]` for a theorem "
                f"that depends only on definitions). Real proofs belong "
                f"in the prover stage; in the skeleton stage every body "
                f"is a `sorry_using` placeholder that records the "
                f"dependency edges for DAG tracking."
            ),
        ))
    return violations or None


def check_all_theorems_annotated(code: str) -> list[dict] | None:
    """Reject any theorem/lemma not preceded by an `@[blueprint ...]`
    attribute. Highlights the `theorem <name>` keyword + name span."""
    sanitised = strip_lean_comments_and_strings(code)
    blocks = extract_blueprint_blocks(sanitised)
    attr_ranges: list[int] = []
    for b in blocks:
        end = b['end']
        while end < len(sanitised) and sanitised[end] in ' \t\n':
            end += 1
        attr_ranges.append(end)
    attr_set = set(attr_ranges)

    violations: list[dict] = []
    for d in _iter_theorem_bodies(sanitised):
        if d['decl_start'] in attr_set:
            continue
        violations.append(_violation_at_offsets(
            code, d['kw_start'], d['kw_end'],
            (
                f"`{d['kind']} {d['name']}` is NOT preceded by an "
                f"`@[blueprint]` attribute. Every theorem and lemma "
                f"in the skeleton MUST have a bare `@[blueprint]` "
                f"attribute on the line directly before it. "
                f"Unannotated theorems are invisible to the blueprint "
                f"prover and carry through to the final file with their "
                f"bodies unproved."
            ),
        ))
    return violations or None


# ---------------------------------------------------------------------------
#  Highlight helpers for post-compile issue rendering
# ---------------------------------------------------------------------------

def _highlight_decl_line(code: str, name: str) -> str | None:
    """Return the line containing `theorem|lemma|def|... <name>` with the
    `kind name` keyword span wrapped in `<error></error>`. Returns None
    if no such decl is found (caller falls back to a synthetic display).
    """
    pattern = (
        r"\b(theorem|lemma|def|abbrev|structure|instance|inductive|class)"
        r"\s+" + re.escape(name) + r"(?![\w'])"
    )
    m = re.search(pattern, code)
    if m is None:
        return None
    line_start = code.rfind('\n', 0, m.start()) + 1
    line_end = code.find('\n', m.end())
    if line_end == -1:
        line_end = len(code)
    line_text = code[line_start:line_end]
    col = m.start() - line_start
    end_col = m.end() - line_start
    return _highlight_line_span(line_text, col, end_col)


def _highlight_sorry_using_line(code: str, name: str) -> str | None:
    """Find the `sorry_using [...]` token associated with the decl
    `<name>` and return its source line with the bracketed list wrapped
    in `<error></error>`. Returns None if not located."""
    pattern = (
        r"\b(?:theorem|lemma)\s+" + re.escape(name)
        + r"(?![\w'])[\s\S]*?sorry_using\s*(\[[^\]]*\])"
    )
    m = re.search(pattern, code)
    if m is None:
        return None
    list_start = m.start(1)
    list_end = m.end(1)
    line_start = code.rfind('\n', 0, list_start) + 1
    line_end = code.find('\n', list_end)
    if line_end == -1:
        line_end = len(code)
    line_text = code[line_start:line_end]
    col = list_start - line_start
    end_col = list_end - line_start
    return _highlight_line_span(line_text, col, end_col)


# ---------------------------------------------------------------------------
#  Post-compile graph validation.
#
#  After a clean Lean compile, parse the model-authored `@[blueprint]`
#  declarations and run the structural checks:
#    * every node has a non-empty `(statement := /-- ... -/)` field;
#    * every Lemma/Theorem has a non-empty `(proof := /-- ... -/)` field;
#    * no `sorry_using` self-loops; deps reference declared names;
#    * `sorry_using` graph is acyclic;
#    * exactly one main Theorem with the canonical name;
#    * every node is reachable in reverse from the main Theorem.
#
#  Issues are returned as {display, message} dicts and rendered with
#  the same fenced-lean4 + `<error></error>` + `Reason:` shape used by
#  the other feedback gates so all rejection responses read consistently.
# ---------------------------------------------------------------------------

def validate_blueprint_graph(
    code: str,
    blueprint_nodes: list,
    theorem_name: str | None,
) -> list[dict] | None:
    """Run structural validation on a model-authored blueprint graph.

    `blueprint_nodes` must come from `graph.parse_blueprint_file` on
    `code`. Returns a list of {display, message} issues or None if the
    graph is valid.
    """
    if not blueprint_nodes:
        return [_violation_synthetic(
            "<error>(no @[blueprint] declarations found)</error>",
            (
                "The Lean file has no `@[blueprint]`-annotated "
                "declarations. Every node of your decomposition "
                "(definitions, lemmas, and the main theorem) must be a "
                "`@[blueprint]`-annotated declaration."
            ),
        )]

    nodes_by_name: dict[str, object] = {n.name: n for n in blueprint_nodes}
    target_kinds = {"theorem", "lemma"}
    issues: list[dict] = []

    # --- Statement / proof field presence on each node ------------------
    for n in blueprint_nodes:
        if not (n.informal_statement or "").strip():
            display = (_highlight_decl_line(code, n.name)
                       or f"<error>@[blueprint] {n.kind} {n.name}</error>")
            issues.append(_violation_synthetic(
                display,
                (
                    f"`{n.kind} {n.name}` is missing a non-empty "
                    f"`(statement := /-- ... -/)` field in its "
                    f"`@[blueprint]` annotation. Every node must carry "
                    f"an NL statement field, e.g. `@[blueprint "
                    f"(statement := /-- claim text -/)]`."
                ),
            ))
        if n.kind in target_kinds and not (n.informal_proof or "").strip():
            display = (_highlight_decl_line(code, n.name)
                       or f"<error>@[blueprint] {n.kind} {n.name}</error>")
            issues.append(_violation_synthetic(
                display,
                (
                    f"`{n.kind} {n.name}` is missing a non-empty "
                    f"`(proof := /-- ... -/)` field in its "
                    f"`@[blueprint]` annotation. Every Lemma and the "
                    f"Theorem must carry an NL proof sketch field, e.g. "
                    f"`@[blueprint (statement := /-- claim -/), "
                    f"(proof := /-- sketch -/)]`."
                ),
            ))

    # --- Self-loops -----------------------------------------------------
    for n in blueprint_nodes:
        if n.name in (n.depends_on or []):
            display = (_highlight_sorry_using_line(code, n.name)
                       or f"<error>{n.kind} {n.name} ... "
                          f"sorry_using [..., {n.name}, ...]</error>")
            issues.append(_violation_synthetic(
                display,
                (
                    f"Node `{n.name}` lists itself in its `sorry_using "
                    f"[...]` (self-loop). A node cannot depend on itself."
                ),
            ))

    # --- Undeclared deps ------------------------------------------------
    for n in blueprint_nodes:
        for d in (n.depends_on or []):
            if d == n.name:
                continue  # already reported as self-loop
            if d not in nodes_by_name:
                display = (_highlight_sorry_using_line(code, n.name)
                           or f"<error>sorry_using [..., {d}, ...]</error>")
                issues.append(_violation_synthetic(
                    display,
                    (
                        f"`{n.kind} {n.name}`'s `sorry_using [...]` "
                        f"references `{d}`, which is not the name of any "
                        f"`@[blueprint]`-annotated declaration in this "
                        f"file. Either add a `@[blueprint]` decl named "
                        f"`{d}` or drop it from `sorry_using`."
                    ),
                ))

    # --- Cycle detection (Kahn's algorithm) -----------------------------
    out_adj: dict[str, list[str]] = {}
    in_adj: dict[str, list[str]] = {}
    for n in blueprint_nodes:
        for d in (n.depends_on or []):
            if d in nodes_by_name and d != n.name:
                out_adj.setdefault(d, []).append(n.name)
                in_adj.setdefault(n.name, []).append(d)
    indeg = {nm: len(in_adj.get(nm, [])) for nm in nodes_by_name}
    work = dict(indeg)
    queue = [nm for nm, k in work.items() if k == 0]
    visited: list[str] = []
    while queue:
        t = queue.pop()
        visited.append(t)
        for nb in out_adj.get(t, []):
            work[nb] -= 1
            if work[nb] == 0:
                queue.append(nb)
    if len(visited) != len(nodes_by_name):
        cycle_members = sorted(set(nodes_by_name) - set(visited))
        if cycle_members:
            display_lines = []
            for nm in cycle_members[:6]:
                hl = _highlight_sorry_using_line(code, nm)
                if hl:
                    display_lines.append(hl)
                else:
                    node = nodes_by_name[nm]
                    display_lines.append(
                        f"<error>{node.kind} {nm}</error>"
                    )
            display = "\n".join(display_lines)
        else:
            display = "<error>(cyclic sorry_using graph)</error>"
        issues.append(_violation_synthetic(
            display,
            (
                f"Cyclic `sorry_using` dependencies through "
                f"{cycle_members}. The graph must be acyclic — break the "
                f"cycle by removing one or more dep edges."
            ),
        ))

    # --- Theorem existence + reachability -------------------------------
    if theorem_name is not None:
        thm_node = nodes_by_name.get(theorem_name)
        if thm_node is None:
            issues.append(_violation_synthetic(
                f"<error>theorem {theorem_name} ... "
                f":= by sorry_using [...]</error>",
                (
                    f"No `@[blueprint]`-annotated declaration named "
                    f"`{theorem_name}` (the canonical theorem "
                    f"identifier). The blueprint must include the main "
                    f"Theorem under its canonical name."
                ),
            ))
        elif thm_node.kind not in target_kinds:
            display = (_highlight_decl_line(code, theorem_name)
                       or f"<error>{thm_node.kind} {theorem_name}</error>")
            issues.append(_violation_synthetic(
                display,
                (
                    f"`{theorem_name}` is declared as `{thm_node.kind}` "
                    f"but the canonical Lean identifier expects a "
                    f"`theorem` (or `lemma`). Re-emit it with the right "
                    f"keyword."
                ),
            ))
        else:
            reachable = {theorem_name}
            stack = [theorem_name]
            while stack:
                t = stack.pop()
                for d in in_adj.get(t, []):
                    if d not in reachable:
                        reachable.add(d)
                        stack.append(d)
            dead = sorted(nm for nm in nodes_by_name if nm not in reachable)
            if dead:
                display_lines: list[str] = []
                for nm in dead[:10]:
                    hl = _highlight_decl_line(code, nm)
                    if hl:
                        display_lines.append(hl)
                    else:
                        node = nodes_by_name[nm]
                        display_lines.append(
                            f"<error>{node.kind} {nm}</error>"
                        )
                display = "\n".join(display_lines)
                issues.append(_violation_synthetic(
                    display,
                    (
                        f"{len(dead)} node(s) not reachable from the "
                        f"main Theorem `{theorem_name}`: {dead}. The "
                        f"Theorem must transitively depend on every "
                        f"other node — wire each dead node into some "
                        f"path that reaches the Theorem (via the "
                        f"`sorry_using` deps), or remove it."
                    ),
                ))

    return issues or None


def check_blueprint_graph_validity(
    code: str, theorem_name: str | None,
) -> list[dict] | None:
    """Post-compile graph-validity gate.

    Parses the blueprint nodes from `code` (using
    `graph.parse_blueprint_file`) and runs `validate_blueprint_graph`.
    Returns a list of {display, message} issues or None if the graph is
    valid.

    Run only after `check_lean_async` succeeded — `parse_blueprint_file`
    relies on well-formed Lean syntax and should never raise on code the
    Lean gateway accepted.
    """
    try:
        _, blueprint_nodes = parse_blueprint_file(code)
    except Exception as e:
        return [_violation_synthetic(
            f"<error>{type(e).__name__}: {str(e)[:200]}</error>",
            (
                "Could not parse the skeleton's `@[blueprint]` decls "
                "for the graph-validity check. The Lean compiler "
                "accepted the file, but the blueprint parser did not — "
                "fix any non-standard `@[blueprint ...]` syntax and "
                "retry."
            ),
        )]
    return validate_blueprint_graph(code, blueprint_nodes, theorem_name)


def _render_validation_blocks(issues: list[dict]) -> str:
    """Render the `Compilation SUCCESSFUL. Validation FAILED with N
    issue.` response. Singular/plural matched per
    never the `(s)` shortcut."""
    n_i = len(issues)
    head = (
        f"Compilation SUCCESSFUL. Validation FAILED with {n_i} issue."
        if n_i == 1
        else
        f"Compilation SUCCESSFUL. Validation FAILED with {n_i} issues."
    )
    parts = [head, ""]
    for i, v in enumerate(issues, 1):
        parts.append(f"Issue {i}:")
        parts.append("```lean4")
        parts.append(v["display"])
        parts.append("```")
        parts.append(f"Reason: {v['message']}")
        parts.append("")
    return '\n'.join(parts).rstrip()


# ---------------------------------------------------------------------------
#  Safe code rebuild: graft canonical main-theorem signature
# ---------------------------------------------------------------------------

def graft_main_signature(
    code: str, formal_statement: str, theorem_name: str,
) -> str:
    """Replace the model's main-theorem signature with the canonical
    one from `formal_statement`, preserving the `@[blueprint]`
    attribute that precedes it and the `:= by sorry_using [...]` body
    the model wrote."""
    canonical_sig = _extract_main_signature(formal_statement, theorem_name)
    if canonical_sig is None:
        return code

    pattern = r"\btheorem\s+" + re.escape(theorem_name) + r"(?![\w'])"
    m = re.search(pattern, code)
    if not m:
        return code

    tail_start = m.start()
    tail = code[tail_start:]
    by_m = re.search(r':=\s*by\b', tail)
    if not by_m:
        return code

    return (
        code[:tail_start]
        + canonical_sig.rstrip()
        + ' '
        + tail[by_m.start():]
    )


# ---------------------------------------------------------------------------
#  User prompts
# ---------------------------------------------------------------------------


def _build_user_prompt_self_decompose(
    formal_statement: str, theorem_name: str | None,
) -> str:
    """User prompt for the default mode (formal statement only).

    The model receives only the canonical Lean signature and must design
    the dependency graph itself, embedding NL `statement` / `proof`
    fields directly into the `@[blueprint ...]` annotations.
    """
    _, body = split_header_body(formal_statement)
    body = body.strip("\n")
    name_clause = (
        f" The main theorem's name MUST be `{theorem_name}`."
        if theorem_name else ""
    )
    return (
        "## Task\n"
        f"Produce a blueprint skeleton for the theorem below. Decompose the proof yourself into a dependency graph of definitions, lemmas, and a single main theorem.{name_clause} Translate the graph into a compilable Lean 4 file where every node is a `@[blueprint ...]`-annotated declaration carrying its natural language `(statement := /-- ... -/)` field (and, for Lemmas/Theorems, a natural language `(proof := /-- ... -/)` field); every theorem and lemma body is `:= by sorry_using [...]` listing its declared deps. You do not prove anything in this stage.\n\n"
        "Use the `lean_compile` tool to verify your skeleton compiles. Iterate until `lean_compile` reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`\n\n"
        "## Original theorem\n"
        "Use this signature exactly for the main theorem. At `lean_compile`, your code is rebuilt with this targeted signature, and the validation check requires the main theorem's name to match.\n\n"
        "```lean4\n"
        f"{body}\n"
        "```\n"
    )


def _build_user_prompt_nl_proof_decompose(
    formal_statement: str, theorem_name: str | None, nl_proof: str,
) -> str:
    """User prompt for --nl-proof-decompose mode.

    The model receives the canonical Lean signature plus a
    natural-language proof, and uses the proof as its decomposition
    blueprint while still authoring the dependency graph itself
    (embedding NL `statement` / `proof` fields directly into the
    `@[blueprint ...]` annotations).
    """
    _, body = split_header_body(formal_statement)
    body = body.strip("\n")
    name_clause = (
        f" The main theorem's name MUST be `{theorem_name}`."
        if theorem_name else ""
    )
    nl_proof_text = (nl_proof or "").strip()
    return (
        "## Task\n"
        f"Produce a blueprint skeleton for the theorem below. A natural-language proof has been supplied — use it as your decomposition blueprint, breaking the argument into definitions, lemmas, and a single main theorem.{name_clause} Translate the graph into a compilable Lean 4 file where every node is a `@[blueprint ...]`-annotated declaration carrying its natural language `(statement := /-- ... -/)` field (and, for Lemmas/Theorems, a natural language `(proof := /-- ... -/)` field); every theorem and lemma body is `:= by sorry_using [...]` listing its declared deps. You do not prove anything in this stage.\n\n"
        "Use the `lean_compile` tool to verify your skeleton compiles. Iterate until `lean_compile` reports `Compilation SUCCESSFUL. Validation SUCCESSFUL.`\n\n"
        "## Original theorem\n"
        "Use this signature exactly for the main theorem. At `lean_compile`, your code is rebuilt with this targeted signature, and the validation check requires the main theorem's name to match.\n\n"
        "```lean4\n"
        f"{body}\n"
        "```\n\n"
        "## Natural-language proof\n"
        "Use this proof as the structural guide for your decomposition. Identify each substantive step and emit it as a Lemma; identify each helper construction (auxiliary function, set, predicate) and emit it as a Definition; let the main Theorem combine them via `sorry_using`. Do not transcribe the proof verbatim — extract its structure, restate every hypothesis as a closed proposition on each Lemma, and split coarse steps into intermediate lemmas where the per-lemma triviality bar requires it.\n\n"
        f"{nl_proof_text}\n"
    )


# ---------------------------------------------------------------------------
#  Single conversation (async)
# ---------------------------------------------------------------------------

async def run_problem(
    client: AsyncOpenAI,
    model_name: str,
    problem: dict,
    lean_server_url: str,
    semaphore: asyncio.Semaphore,
    tokenizer,
    tools_overhead: int = 0,
    max_turns: int = 64,
    temperature: float = 1.0,
    lean_timeout: int = 300,
    sample_idx: int = 0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_output_tokens: int | None = None,
    patience: int | None = None,
    extra_body: dict | None = None,
    prune_dead: bool = True,
    nl_proof_decompose: bool = False,
    search_server_url: str | None = None,
) -> dict:
    async with semaphore:
        return await _run_problem_inner(
            client, model_name, problem, lean_server_url,
            tokenizer, tools_overhead, max_turns, temperature,
            lean_timeout, sample_idx, max_model_len, max_output_tokens,
            patience, extra_body, prune_dead,
            nl_proof_decompose, search_server_url,
        )


async def _run_problem_inner(
    client: AsyncOpenAI,
    model_name: str,
    problem: dict,
    lean_server_url: str,
    tokenizer,
    tools_overhead: int,
    max_turns: int,
    temperature: float,
    lean_timeout: int,
    sample_idx: int,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_output_tokens: int | None = None,
    patience: int | None = None,
    extra_body: dict | None = None,
    prune_dead: bool = True,
    nl_proof_decompose: bool = False,
    search_server_url: str | None = None,
) -> dict:
    problem_id = problem["problem_id"]
    formal_statement = problem["formal_statement"]
    # The model authors its own decomposition graph in both modes; they
    # differ only in the system + user prompts (`nl_proof_decompose`
    # additionally supplies a natural-language proof as guidance).
    theorem_name = get_theorem_name(formal_statement)
    tag = f"{problem_id}/s{sample_idx}"

    active_tools = [LEAN_COMPILE_TOOL]
    if search_server_url:
        active_tools.append(MATHLIB_SEARCH_TOOL)
    system_prompt = system_prompt_for(nl_proof_decompose, bool(search_server_url))

    if nl_proof_decompose:
        nl_proof = problem.get("nl_proof") or problem.get("stage1_proof") or ""
        user_prompt = _build_user_prompt_nl_proof_decompose(
            formal_statement, theorem_name, nl_proof,
        )
        log.info(
            "[%s] Start (max_turns=%d, mode=nl_proof_decompose, "
            "nl_proof_chars=%d)",
            tag, max_turns, len(nl_proof),
        )
    else:
        user_prompt = _build_user_prompt_self_decompose(
            formal_statement, theorem_name,
        )
        log.info(
            "[%s] Start (max_turns=%d, mode=formal_statement_only)",
            tag, max_turns,
        )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    trace = {
        "problem_id": problem_id,
        "sample": sample_idx,
        "formal_statement": formal_statement,
        "nl_proof_decompose": nl_proof_decompose,
        "system_prompt": system_prompt,
        "tools": active_tools,
        "user_prompt": user_prompt,
        "max_turns": max_turns,
        "patience": patience,
        "turns": [],
        "success": False,
        "complete": False,
        "total_model_tokens": {"input": 0, "output": 0},
        "total_cached_tokens": 0,
        "total_cost_usd": 0.0,
        "total_lean_time": 0.0,
        "total_search_calls": 0,
    }

    t0 = time.time()

    min_errors: int | None = None
    last_improve_turn: int = 0

    for turn_idx in range(max_turns):
        turn_record = {"turn": turn_idx + 1}

        if patience is not None and (turn_idx + 1) - last_improve_turn > patience:
            log.info("[%s] turn %d: patience exceeded (no improvement in %d turns, min_errors=%s)",
                     tag, turn_idx + 1, patience, min_errors)
            turn_record["status"] = "patience_exceeded"
            trace["turns"].append(turn_record)
            break

        prompt_tokens = count_prompt_tokens(tokenizer, messages, tools_overhead)
        max_tokens = compute_max_tokens(prompt_tokens, max_model_len, max_output_tokens)
        turn_record["prompt_tokens"] = prompt_tokens
        turn_record["max_tokens"] = max_tokens

        if max_tokens == 0:
            log.warning("[%s] Context full on turn %d (prompt=%d tokens), stopping",
                        tag, turn_idx + 1, prompt_tokens)
            turn_record["status"] = "context_full"
            trace["turns"].append(turn_record)
            break

        try:
            response = await chat_with_retry(
                client,
                tag=tag,
                model=model_name,
                messages=messages,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as e:
            log.error("[%s] Model error turn %d: %s", tag, turn_idx + 1, e)
            turn_record["status"] = "model_error"
            turn_record["error"] = str(e)
            trace["turns"].append(turn_record)
            break

        msg = response.choices[0].message
        content = msg.content or ""
        # OpenRouter exposes provider reasoning via msg.reasoning (or
        # reasoning_content for some providers); capture whichever exists.
        reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or ""
        )
        finish_reason = response.choices[0].finish_reason

        if response.usage:
            pt = response.usage.prompt_tokens or 0
            ct = response.usage.completion_tokens or 0
            cached = getattr(response.usage, "cached_tokens", 0) or 0
            cost = getattr(response.usage, "cost_usd", 0.0) or 0.0
            trace["total_model_tokens"]["input"] += pt
            trace["total_model_tokens"]["output"] += ct
            trace["total_cached_tokens"] += cached
            trace["total_cost_usd"] += cost
            turn_record["billed_prompt_tokens"] = pt
            turn_record["cached_tokens"] = cached
            turn_record["cost_usd"] = cost

        if reasoning and content:
            turn_record["raw_content"] = f"{reasoning}\n\n{content}"
        else:
            turn_record["raw_content"] = reasoning or content
        turn_record["content"] = content
        turn_record["reasoning_content"] = reasoning
        turn_record["finish_reason"] = finish_reason

        # Build assistant message for conversation history. Including
        # `reasoning` makes OpenRouter forward it to the provider so the
        # model sees its own prior chain-of-thought across turns.
        assistant_msg: dict = {"role": "assistant", "content": content}
        if reasoning:
            assistant_msg["reasoning"] = reasoning
        if msg.tool_calls:
            tc_dicts = []
            for tc in msg.tool_calls:
                d = {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                ec = getattr(tc, "extra_content", None)
                if ec:  # Gemini thought_signature — required round-trip
                    d["extra_content"] = ec
                tc_dicts.append(d)
            assistant_msg["tool_calls"] = tc_dicts
        messages.append(assistant_msg)

        if not msg.tool_calls:
            code = extract_lean_code_from_text(content)
            if code:
                turn_record["tool_call"] = {
                    "name": "lean_compile",
                    "code": code,
                    "from_fallback": True,
                }
                tool_response_text = await _handle_lean_compile(
                    code, lean_server_url, lean_timeout,
                    formal_statement, theorem_name,
                    tag, turn_idx, turn_record, trace,
                )
                messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{tool_response_text}\n</tool_response>",
                })
                turn_record["tool_response"] = tool_response_text
                turn_record["tool_calls"] = [{
                    "name": "lean_compile",
                    "arguments": {"code": code},
                    "response": tool_response_text,
                }]
                if turn_record.get("status") == "success":
                    trace["turns"].append(turn_record)
                    break
                trace["turns"].append(turn_record)
                if trace.get("dep_block_fail_abort"):
                    break
                ne = turn_record.get("n_errors")
                if ne is not None and (min_errors is None or ne < min_errors):
                    min_errors = ne
                    last_improve_turn = turn_idx + 1
                continue

            log.warning("[%s] No tool call on turn %d — ending rollout",
                        tag, turn_idx + 1)
            turn_record["status"] = "no_tool_call"
            trace["turns"].append(turn_record)
            break

        solved_this_turn = False
        tool_calls_record = []

        for call in msg.tool_calls:
            if call.function.name == "mathlib_search":
                if not search_server_url:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "mathlib_search is not enabled on this run.",
                    })
                    continue
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {"query": call.function.arguments}
                query = args.get("query", "")
                top_k = args.get("top_k", 5)
                log.info(
                    "[%s] mathlib_search turn %d: %r (top_k=%d)",
                    tag, turn_idx + 1, query, top_k,
                )
                try:
                    search_result = await search_mathlib_async(
                        query, search_server_url, top_k,
                    )
                    response_text = format_search_response(search_result)
                except Exception as e:
                    log.error(
                        "[%s] Search error turn %d: %s",
                        tag, turn_idx + 1, e,
                    )
                    response_text = f"Mathlib search error: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": response_text,
                })
                tool_calls_record.append({
                    "name": "mathlib_search",
                    "arguments": {"query": query, "top_k": top_k},
                    "response": response_text,
                })
                trace["total_search_calls"] += 1

            elif call.function.name == "lean_compile":
                code = extract_tool_code_arg(call.function.arguments)

                turn_record["tool_call"] = {
                    "name": "lean_compile",
                    "code": code,
                }

                tool_response_text = await _handle_lean_compile(
                    code, lean_server_url, lean_timeout,
                    formal_statement, theorem_name,
                    tag, turn_idx, turn_record, trace,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_response_text,
                })
                turn_record["tool_response"] = tool_response_text
                tool_calls_record.append({
                    "name": "lean_compile",
                    "arguments": {"code": code},
                    "response": tool_response_text,
                })

                if turn_record.get("status") == "success":
                    solved_this_turn = True

            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"Unknown tool: {call.function.name}",
                })

        turn_record["tool_calls"] = tool_calls_record

        if "status" not in turn_record:
            turn_record["status"] = "no_lean_compile"

        trace["turns"].append(turn_record)

        ne = turn_record.get("n_errors")
        if ne is not None and (min_errors is None or ne < min_errors):
            min_errors = ne
            last_improve_turn = turn_idx + 1

        if solved_this_turn:
            break
        if trace.get("dep_block_fail_abort"):
            break

    trace["elapsed_seconds"] = round(time.time() - t0, 2)
    trace["total_turns"] = len(trace["turns"])

    if not trace["success"]:
        for turn in reversed(trace["turns"]):
            tc = turn.get("tool_call", {})
            if tc.get("code"):
                trace["final_code"] = tc["code"]
                trace["final_skeleton"] = tc["code"]
                break

    # Post-sketch prune: drop scaffolding nodes the sketcher introduced
    # but never wired into the main theorem's `sorry_using [...]` graph.
    if prune_dead and trace.get("final_code") and theorem_name:
        try:
            pruned_code, _, _, _, ana = prune_dead_nodes(
                code=trace["final_code"],
                main_theorem_name=theorem_name,
                lean_server_url=lean_server_url,
                timeout=lean_timeout,
            )
            if ana is not None and ana.dead:
                log.info(
                    "[%s] post-sketch pruned %d dead nodes: %s "
                    "(reachable=%d unsolved=%d)",
                    tag, len(ana.dead), sorted(ana.dead),
                    len(ana.reachable), len(ana.unsolved_targets),
                )
                trace["final_code"] = pruned_code
                trace["final_skeleton"] = pruned_code
        except Exception as e:
            log.warning("[%s] post-sketch prune skipped: %s", tag, e)

    status = "SOLVED" if trace["success"] else "FAIL"
    log.info("[%s] %s — turns=%d elapsed=%.1fs cost=$%.4f",
             tag, status, trace["total_turns"], trace["elapsed_seconds"],
             trace["total_cost_usd"])
    return trace


# Number of feedback turns the model gets to fix a dep-block failure
# before the sample is abandoned. 0 discards without feedback. Without
# feedback, blueprints whose un-annotated preamble declarations are dropped
# by the per-node slicer fail identically on every sample, with the model
# never shown the error.
_DEP_BLOCK_FEEDBACK_TURNS = int(os.environ.get("DEP_BLOCK_FEEDBACK_TURNS", "2"))

# Lean idents are unicode; match "name = anything up to whitespace or a
# delimiter" (anonymous instances simply do not match - fine).
_TOP_DECL_RE = re.compile(
    r"^(?:private\s+|protected\s+|noncomputable\s+|local\s+)*"
    r"(?:def|abbrev|structure|inductive|class|instance|theorem|lemma|opaque)"
    r"[ \t]+([^\s(\[{:]+)",
    re.MULTILINE,
)


def _dep_block_feedback(safe_code: str, fail_name: str, fail_errors: list,
                        attempt: int, max_feedback: int) -> str:
    """Actionable tool response for a dep-block failure: explain the
    sliced-context contract and name the file's violations (un-annotated
    top-level decls, `variable` lines) so the model can repair the file
    instead of the sample being silently discarded."""
    try:
        _, nodes = parse_blueprint_file(safe_code)
        annotated = {n.name for n in nodes}
    except Exception:
        annotated = set()
    seen: set[str] = set()
    unannotated = []
    for m in _TOP_DECL_RE.finditer(safe_code):
        name = m.group(1)
        if name not in annotated and name not in seen:
            seen.add(name)
            unannotated.append(name)
    var_lines = [i + 1 for i, l in enumerate(safe_code.split("\n"))
                 if l.lstrip().startswith("variable")]
    errs = "\n".join(f"  - {e.get('data', '')[:200]}" for e in fail_errors[:3])
    parts = [
        f"Dep-block check FAILED for node `{fail_name}` "
        f"(feedback {attempt}/{max_feedback}; this sample is abandoned "
        f"when feedback turns run out).",
        "Your file compiles as a whole, but every node must ALSO compile "
        "in its sliced per-node context, which contains ONLY: header "
        "lines (`import`/`open`/`set_option`/`universe`/`namespace`), "
        "the `@[blueprint]`-annotated definition-kind declarations, and "
        "the signatures of the node's `sorry_using` parents. Every other "
        "line of the file is dropped from that context.",
        f"First errors in `{fail_name}`'s sliced context:\n{errs}",
    ]
    if unannotated:
        shown = ", ".join(f"`{x}`" for x in unannotated[:12])
        parts.append(
            f"Top-level declarations WITHOUT `@[blueprint]` (dropped from "
            f"every sliced context): {shown}. Annotate EVERY declaration "
            f"with a bare `@[blueprint]` line.")
    if var_lines:
        parts.append(
            f"`variable` line(s) at line {var_lines} are also dropped "
            f"from sliced contexts. Remove them and write the binders "
            f"explicitly in each declaration that needs them.")
    parts.append("Fix the complete file and resubmit it via lean_compile.")
    return "\n\n".join(parts)


async def _dep_block_acceptance_gate(
    safe_code: str,
    lean_server_url: str,
    lean_timeout: int,
    tag: str,
    turn_idx: int,
    turn_record: dict,
    trace: dict,
) -> str | None:
    """Run the per-target dep-block compile check; on failure give the
    model up to `_DEP_BLOCK_FEEDBACK_TURNS` repair turns, then mark the
    sample for abandonment and return the placeholder tool response.

    Returns None when every target's dep block compiles cleanly (caller
    proceeds to the success branch). Returns a non-empty string when at
    least one target fails — caller must return that string directly;
    the outer turn loop continues (feedback) or breaks (abort flag set).
    """
    dep_fail = await check_dep_blocks_compile(
        safe_code, lean_server_url, lean_timeout,
    )
    if dep_fail is None:
        return None
    fail_name, fail_errors = dep_fail
    err_data = [e.get("data", "")[:120] for e in fail_errors[:3]]
    turn_record["status"] = "dep_block_fail"
    turn_record["dep_block_fail_target"] = fail_name
    turn_record["dep_block_fail_errors"] = fail_errors
    n_fails = trace.get("dep_block_fail_count", 0) + 1
    trace["dep_block_fail_count"] = n_fails
    if n_fails <= _DEP_BLOCK_FEEDBACK_TURNS:
        log.warning(
            "[%s] turn %d: skeleton compiles but per-node dep block for "
            "`%s` does not (%d %s) — feedback %d/%d. First errors: %s",
            tag, turn_idx + 1, fail_name, len(fail_errors),
            "error" if len(fail_errors) == 1 else "errors",
            n_fails, _DEP_BLOCK_FEEDBACK_TURNS, "; ".join(err_data),
        )
        return _dep_block_feedback(
            safe_code, fail_name, fail_errors,
            n_fails, _DEP_BLOCK_FEEDBACK_TURNS,
        )
    log.warning(
        "[%s] turn %d: skeleton compiles but per-node dep block for "
        "`%s` does not (%d %s) — feedback exhausted, abandoning sample. "
        "First errors: %s",
        tag, turn_idx + 1, fail_name, len(fail_errors),
        "error" if len(fail_errors) == 1 else "errors",
        "; ".join(err_data),
    )
    trace["dep_block_fail_abort"] = True
    trace["dep_block_fail_target"] = fail_name
    return (
        f"Skeleton compiles but the per-node dependency block for "
        f"`{fail_name}` fails to compile. Abandoning this skeleton "
        f"sample (the next num_sample retry will start fresh)."
    )


async def _handle_lean_compile(
    code: str,
    lean_server_url: str,
    lean_timeout: int,
    formal_statement: str,
    theorem_name: str | None,
    tag: str,
    turn_idx: int,
    turn_record: dict,
    trace: dict,
) -> str:
    """Blueprint-generation lean_compile handler.

    Pre-compile checks (each rejected via the unified `Safeguard rejected`
    status line so the model can distinguish pre-compile rejection from
    real Lean errors):
      0. `/- ... -/` block-comment balance
      1. theorem name is present
      2. forbidden constructs (axiom, opaque, native_decide)
      3. imports are present (Mathlib + Architect)
      4. theorem signature matches the original (verbatim modulo whitespace)
      5. no bare `sorry`
      6. every theorem/lemma body is `:= by sorry_using [...]`
         (no inlined real proofs)
      7. every theorem/lemma is `@[blueprint]`-annotated

    If all checks pass, send the code to the Lean gateway.

    Post-compile gate: the parsed blueprint decls are checked for
    structural validity (no cycles, no self-loops, deps resolve, single
    main Theorem with the canonical name, every node reachable from the
    Theorem, NL `statement` / `proof` fields present). Any failure is
    reported as `Compilation SUCCESSFUL. Validation FAILED with N issue.`
    + `Issue N:` blocks, rendered with the same fenced-lean4 +
    `<error></error>` + `Reason:` shape as the pre-compile safeguards.
    Then every target's sliced per-node context must compile
    (`_dep_block_acceptance_gate`).

    Success criterion: compilation has no ERRORS, AND the chosen
    post-compile gate passes. Sorries from `sorry_using` are expected
    and do NOT count as failure.
    """

    # Each pre-compile check returns None (passed) or a non-empty list
    # of {display, message} violations ready for the unified renderer.
    pre_checks: list[tuple[str, list[dict] | None]] = [
        ("unbalanced_block_comment",
         check_unbalanced_block_comment(code)),
        ("missing_theorem",
         check_missing_theorem(code, theorem_name, formal_statement)),
        ("safeguard_precheck",
         check_forbidden_constructs(code)),
        ("missing_imports",
         check_imports(code)),
        ("signature_mismatch",
         check_main_signature_matches(code, formal_statement, theorem_name)),
        ("bare_sorry",
         check_no_bare_sorry(code)),
        ("non_sorry_using_body",
         check_body_is_sorry_using(code)),
        ("unannotated_theorem",
         check_all_theorems_annotated(code)),
    ]
    for status, violations in pre_checks:
        if not violations:
            continue
        n_v = len(violations)
        log.warning("[%s] %s on turn %d (%d %s)",
                    tag, status, turn_idx + 1, n_v,
                    "violation" if n_v == 1 else "violations")
        turn_record["status"] = status
        turn_record["compilation_pass"] = False
        turn_record["compilation_complete"] = False
        if status == "safeguard_precheck":
            # Tag axiom/native_decide types for downstream telemetry.
            tags: list[str] = []
            for v in violations:
                # The display text contains <error>...</error>; recover
                # the offending construct from the unsanitised display
                # by checking for the keyword.
                disp = v.get("display", "")
                if "axiom" in disp:
                    tags.append("axiom")
                elif "native_decide" in disp:
                    tags.append("native_decide")
                elif "opaque" in disp:
                    tags.append("opaque")
                else:
                    tags.append("forbidden")
            turn_record["safeguard_violations"] = tags
        return _render_violation_blocks(violations)

    # --- Safe code rebuild: graft canonical main signature BEFORE compile ---
    if theorem_name:
        safe_code = graft_main_signature(code, formal_statement, theorem_name)
    else:
        safe_code = code
    turn_record["safe_code"] = safe_code

    # --- Compile via Lean gateway ---
    code_lines = len(safe_code.split("\n"))
    log.info("[%s] turn %d: lean_compile (%d lines, %d blueprint decls)",
             tag, turn_idx + 1, code_lines, count_blueprint_decls(safe_code))
    log.info("[%s] turn %d: === SAFE CODE ===\n%s",
             tag, turn_idx + 1, safe_code)

    try:
        lean_result = await check_lean_async(
            safe_code, lean_server_url, lean_timeout,
        )
        analysis = analyse_result(lean_result)

        trace["total_lean_time"] += analysis["time"]
        turn_record["lean_result"] = lean_result
        turn_record["compilation_pass"] = analysis["pass_"]
        turn_record["compilation_complete"] = analysis["complete"]
        turn_record["n_errors"] = 0 if analysis["pass_"] else len(analysis["errors"])
        lean_time = analysis["time"]

        if not analysis["pass_"]:
            turn_record["status"] = "compile_fail"
            first_err = analysis["errors"][0] if analysis["errors"] else {}
            err_line = first_err.get("pos", {}).get("line", "?")
            err_msg = first_err.get("data", "")[:120]
            n_err = len(analysis["errors"])
            log.info(
                "[%s] turn %d: COMPILE FAIL — %d %s, %.1fs, first@L%s: %s",
                tag, turn_idx + 1, n_err,
                "error" if n_err == 1 else "errors",
                lean_time, err_line, err_msg,
            )
            tool_response_text = format_compile_response(
                safe_code, lean_result,
            )
            log.info("[%s] turn %d: === FEEDBACK ===\n%s",
                     tag, turn_idx + 1, tool_response_text)
            return tool_response_text

        # Lean accepted the file. Now run the post-compile graph-validity
        # gate.
        sorry_count = len(analysis["sorries"])

        issues = check_blueprint_graph_validity(
            safe_code, theorem_name,
        )
        turn_record["validation_issues"] = (
            len(issues) if issues else 0
        )
        if issues:
            n_i = len(issues)
            log.info(
                "[%s] turn %d: COMPILE OK but graph validation "
                "FAILED — %d %s, %.1fs (sorry_count=%d)",
                tag, turn_idx + 1, n_i,
                "issue" if n_i == 1 else "issues",
                lean_time, sorry_count,
            )
            turn_record["status"] = "validation_fail"
            tool_response_text = _render_validation_blocks(issues)
            log.info("[%s] turn %d: === FEEDBACK ===\n%s",
                     tag, turn_idx + 1, tool_response_text)
            return tool_response_text

        # Final acceptance gate: per-target dep block must also compile
        # (mirrors what the downstream prover extracts from the
        # blueprint). On failure the model gets a few repair turns
        # before the sample is abandoned (`_dep_block_acceptance_gate`).
        dep_fail = await _dep_block_acceptance_gate(
            safe_code, lean_server_url, lean_timeout,
            tag, turn_idx, turn_record, trace,
        )
        if dep_fail:
            return dep_fail

        # Both gates + dep-block check passed → blueprint is accepted.
        log.info("[%s] turn %d: SUCCESS — skeleton compiles and "
                 "graph is valid (sorry_count=%d, %.1fs)",
                 tag, turn_idx + 1, sorry_count, lean_time)
        turn_record["status"] = "success"
        trace["success"] = True
        trace["complete"] = analysis["complete"]
        trace["final_code"] = safe_code
        trace["final_skeleton"] = safe_code
        return "Compilation SUCCESSFUL. Validation SUCCESSFUL."

    except Exception as e:
        log.error("[%s] Lean error turn %d: %s", tag, turn_idx + 1, e)
        turn_record["status"] = "lean_error"
        turn_record["error"] = str(e)
        return f"Lean 4 server error: {e}"


# ---------------------------------------------------------------------------
#  Main async loop
# ---------------------------------------------------------------------------

async def async_main():
    parser = argparse.ArgumentParser(
        description=(
            "Blueprint generation: decompose each formal statement into a "
            "`@[blueprint]`-annotated Lean 4 file of definitions and "
            "lemmas with `sorry_using` bodies, iterating against the Lean "
            "compiler until the file compiles and the graph is valid."
        )
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL — each line must have "
                             "{problem_id, formal_statement} (plus nl_proof "
                             "with --nl-proof-decompose)")
    parser.add_argument("--output", required=True, help="Output directory")
    add_model_args(parser, x_title="goedel-architect-blueprint")
    parser.add_argument("--lean-server", default=DEFAULT_LEAN_SERVER,
                        help="Lean server check endpoint (kimina-lean-server "
                             "API) with Mathlib + LeanArchitect")
    parser.add_argument("--search-server", default=DEFAULT_SEARCH_SERVER,
                        help="Mathlib search endpoint for the "
                             "`mathlib_search` tool; 'none' disables search")
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Independent conversations per problem")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Max concurrent conversations. If unset, "
                             "auto-picks min(len(problems), 128).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lean-timeout", type=int, default=300)
    parser.add_argument("--tokenizer-path", default=None,
                        help="Optional HF tokenizer path/name for prompt-token "
                             "counting; falls back to char estimator")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                        help="Max context length the remote model accepts. "
                             "DeepSeek-V4 supports up to 1M; default 256K is "
                             "the agentic sweet spot.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Optional cap on output tokens per response. If "
                             "unset, the per-turn max is whatever budget "
                             "remains in --max-model-len after the prompt.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Stop if min_error_count hasn't decreased in N "
                             "turns (default: off)")
    parser.add_argument("--problem-max-attempts", type=int, default=3,
                        help="Retry a problem this many times if run_problem "
                             "raises an unexpected exception. Default: 3.")
    parser.add_argument("--limit", type=int, help="Process only first N problems")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (problem, sample) pairs already in traces.jsonl")
    parser.add_argument("--early-stop", action="store_true",
                        help="Stop sampling a problem after first success "
                             "(sequential attempts)")
    parser.add_argument(
        "--prune-dead", action=argparse.BooleanOptionalAction, default=True,
        help="After sketch, prune nodes unreachable from the main theorem in "
             "the emitted skeleton. ON by default; use --no-prune-dead to disable.",
    )
    parser.add_argument(
        "--nl-proof-decompose", action="store_true",
        help="Author the blueprint graph guided by a natural-language "
             "proof. The model receives the formal signature plus the "
             "supplied NL proof and uses the proof as its decomposition "
             "blueprint (still emitting its own `(statement := ...)` / "
             "`(proof := ...)` fields). Input rows need `problem_id`, "
             "`formal_statement`, and `nl_proof` (alias `stage1_proof`).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    log_path = setup_logger(out_dir)
    log.info("Output directory: %s", out_dir)
    log.info("Session log: %s", log_path)
    log.info("Model URL: %s", args.model_url)
    log.info("Model name: %s", args.model_name)
    log.info("Max model len: %d", args.max_model_len)
    log.info("Max output tokens: %s",
             args.max_output_tokens if args.max_output_tokens is not None
             else "<unset — context-window remainder>")

    api_key = read_api_key(args)

    tokenizer = load_tokenizer(args.tokenizer_path)
    tools_overhead = estimate_tools_overhead(tokenizer, TOOLS)

    problems = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if args.limit:
        problems = problems[:args.limit]
    log.info("Loaded %d problems from %s", len(problems), args.input)

    # Input rows need `problem_id` + `formal_statement`; with
    # --nl-proof-decompose also `nl_proof` (alias `stage1_proof`).
    missing_field_pids: list[tuple[str, str]] = []
    if args.nl_proof_decompose:
        log.info(
            "nl_proof_decompose=True — input rows need problem_id + "
            "formal_statement + nl_proof (alias stage1_proof)",
        )
    for p in problems:
        # Accept `uuid` as a fallback alias for `problem_id`.
        if "problem_id" not in p and "uuid" in p:
            p["problem_id"] = p["uuid"]
        for field in ("problem_id", "formal_statement"):
            if field not in p:
                missing_field_pids.append(
                    (p.get("problem_id", "<unknown>"), field)
                )
        pid = p.get("problem_id", "<unknown>")
        if args.nl_proof_decompose:
            nl_proof = (
                p.get("nl_proof") or p.get("stage1_proof") or ""
            )
            if not isinstance(nl_proof, str) or not nl_proof.strip():
                missing_field_pids.append(
                    (pid, "nl_proof (or stage1_proof)")
                )
                continue
            p["nl_proof"] = nl_proof
    if missing_field_pids:
        for pid, field in missing_field_pids[:10]:
            log.error("Problem %s missing field %r", pid, field)
        log.error("Aborting: %d problems missing required fields",
                  len(missing_field_pids))
        sys.exit(1)

    num_samples = args.num_samples
    if args.concurrency is None:
        concurrency = min(len(problems), 128)
        log.info("Concurrency: auto -> %d (= min(len(problems)=%d, 128))",
                 concurrency, len(problems))
    else:
        concurrency = args.concurrency
        log.info("Concurrency: %d (explicit)", concurrency)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.model_url,
        default_headers=default_headers(args),
    )
    extra_body = build_extra_body(args)
    search_server_url = resolve_search_server(args.search_server)

    semaphore = asyncio.Semaphore(concurrency)
    traces_path = out_dir / "traces.jsonl"
    write_lock = asyncio.Lock()

    # Resume bookkeeping
    done_keys: set[tuple[str, int]] = set()
    existing_traces: list[dict] = []
    solved_pids: set[str] = set()
    if args.resume and traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                existing_traces.append(entry)
                done_keys.add((entry["problem_id"], entry.get("sample", 0)))
                if entry.get("success"):
                    solved_pids.add(entry["problem_id"])
        log.info("Resumed %d completed conversations from %s",
                 len(done_keys), traces_path)
        if args.early_stop:
            log.info("Early-stop resume: %d problems already solved, will skip",
                     len(solved_pids))

    if not args.resume and traces_path.exists() and traces_path.stat().st_size > 0:
        log.error(
            "Output %s already contains results. Pass --resume to continue "
            "or choose a new --output dir.",
            traces_path,
        )
        sys.exit(1)
    if not args.resume:
        traces_path.write_text("")

    completed = 0
    total_convos = 0

    async def run_and_save(problem: dict, sample_idx: int) -> dict:
        nonlocal completed
        pid = problem["problem_id"]
        tag = f"{pid}/s{sample_idx}"
        max_attempts = max(1, args.problem_max_attempts)
        last_exc: Exception | None = None
        trace: dict | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                trace = await run_problem(
                    client=client,
                    model_name=args.model_name,
                    problem=problem,
                    lean_server_url=args.lean_server,
                    semaphore=semaphore,
                    tokenizer=tokenizer,
                    tools_overhead=tools_overhead,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    lean_timeout=args.lean_timeout,
                    sample_idx=sample_idx,
                    max_model_len=args.max_model_len,
                    max_output_tokens=args.max_output_tokens,
                    patience=args.patience,
                    extra_body=extra_body,
                    prune_dead=args.prune_dead,
                    nl_proof_decompose=args.nl_proof_decompose,
                    search_server_url=search_server_url,
                )
                # Treat "no progress" traces (every turn = model_error)
                # as retriable so transient transport storms don't bury
                # an entire conversation as a 1-turn fail.
                turns = trace.get("turns", []) if trace else []
                no_progress = (
                    not turns
                    or all(t.get("status") == "model_error" for t in turns)
                )
                if no_progress and attempt < max_attempts:
                    last_status = turns[-1].get("error", "<no turns>") if turns else "<no turns>"
                    backoff = min(2 ** (attempt - 1) * 5, 60)
                    log.warning("[%s] attempt %d/%d: no-progress trace "
                                "(every turn=model_error, last=%s); "
                                "retrying in %ds",
                                tag, attempt, max_attempts,
                                str(last_status)[:120], backoff)
                    await asyncio.sleep(backoff)
                    continue
                if attempt > 1:
                    log.info("[%s] succeeded on retry attempt %d/%d",
                             tag, attempt, max_attempts)
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                if attempt >= max_attempts:
                    log.error("[%s] gave up after %d attempts: %s: %s",
                              tag, max_attempts, type(e).__name__, e)
                    break
                backoff = min(2 ** (attempt - 1) * 5, 60)
                log.warning("[%s] attempt %d/%d failed (%s: %s); "
                            "retrying in %ds",
                            tag, attempt, max_attempts,
                            type(e).__name__, e, backoff)
                await asyncio.sleep(backoff)

        if trace is None:
            trace = {
                "problem_id": pid,
                "sample": sample_idx,
                "success": False,
                "complete": False,
                "total_turns": 0,
                "elapsed_seconds": 0.0,
                "total_model_tokens": {"input": 0, "output": 0},
                "total_cached_tokens": 0,
                "total_cost_usd": 0.0,
                "total_lean_time": 0.0,
                "turns": [],
                "error": (
                    f"{type(last_exc).__name__}: {last_exc}"
                    if last_exc is not None else "unknown error"
                ),
                "attempts": max_attempts,
            }

        async with write_lock:
            with open(traces_path, "a") as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
            completed += 1
            if completed % 10 == 0:
                log.info("Progress: %d conversations saved", completed)

        return trace

    t_start = time.time()

    if args.early_stop:
        problems_to_run = [p for p in problems if p["problem_id"] not in solved_pids]
        log.info(
            "Early-stop: scheduling %d problems (up to %d samples each, "
            "%d already solved), concurrency=%d",
            len(problems_to_run), num_samples, len(solved_pids), concurrency,
        )

        async def run_problem_early_stop(problem: dict) -> list[dict]:
            pid = problem["problem_id"]
            traces_for_problem = []
            for si in range(num_samples):
                if (pid, si) in done_keys:
                    continue
                trace = await run_and_save(problem, si)
                traces_for_problem.append(trace)
                if trace.get("success"):
                    log.info("[%s] Solved on sample %d (attempt %d/%d)",
                             pid, si, si + 1, num_samples)
                    break
            return traces_for_problem

        results_nested = await asyncio.gather(
            *(run_problem_early_stop(p) for p in problems_to_run),
            return_exceptions=True,
        )

        all_traces = list(existing_traces)
        for result in results_nested:
            if isinstance(result, Exception):
                log.error("Problem-level exception: %s", result)
            else:
                all_traces.extend(result)

    else:
        tasks_to_run: list[tuple[dict, int]] = []
        for problem in problems:
            pid = problem["problem_id"]
            for si in range(num_samples):
                if (pid, si) not in done_keys:
                    tasks_to_run.append((problem, si))

        total_convos = len(tasks_to_run)
        log.info(
            "Scheduling %d conversations (%d problems x %d samples, %d resumed), concurrency=%d",
            total_convos, len(problems), num_samples, len(done_keys), concurrency,
        )

        results = await asyncio.gather(
            *(run_and_save(p, si) for p, si in tasks_to_run),
            return_exceptions=True,
        )

        for (problem, sample_idx), result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                pid = problem["problem_id"]
                log.error("[%s/s%d] Exception: %s", pid, sample_idx, result)
                error_trace = {
                    "problem_id": pid, "sample": sample_idx,
                    "success": False, "complete": False,
                    "total_turns": 0, "elapsed_seconds": 0,
                    "total_model_tokens": {"input": 0, "output": 0},
                    "total_cached_tokens": 0, "total_cost_usd": 0.0,
                    "total_lean_time": 0,
                    "turns": [],
                    "error": str(result),
                }
                with open(traces_path, "a") as f:
                    f.write(json.dumps(error_trace, ensure_ascii=False, default=str) + "\n")

        all_traces: list[dict] = list(existing_traces)
        for result in results:
            if not isinstance(result, Exception):
                all_traces.append(result)

    # ── Group by problem ─────────────────────────────────────────────────
    per_problem_traces: dict[str, list[dict]] = {}
    for t in all_traces:
        pid = t["problem_id"]
        per_problem_traces.setdefault(pid, []).append(t)

    elapsed_total = time.time() - t_start
    n_problems = len(per_problem_traces)

    problem_success_counts: dict[str, int] = {}
    all_summaries: dict[str, list[dict]] = {}
    for pid, traces in per_problem_traces.items():
        sums = []
        for t in traces:
            sums.append({
                "problem_id": pid,
                "sample": t.get("sample", 0),
                "success": t.get("success", False),
                "complete": t.get("complete", False),
                "turns": t.get("total_turns", 0),
                "elapsed": t.get("elapsed_seconds", 0),
                "tokens_in": t.get("total_model_tokens", {}).get("input", 0),
                "tokens_out": t.get("total_model_tokens", {}).get("output", 0),
                "cached_tokens": t.get("total_cached_tokens", 0),
                "cost_usd": t.get("total_cost_usd", 0.0),
                "lean_time": t.get("total_lean_time", 0),
            })
        all_summaries[pid] = sums
        problem_success_counts[pid] = sum(1 for s in sums if s.get("success", False))

    total_conversations = sum(len(s) for s in all_summaries.values())

    pass_at_k_results = {}
    for k in sorted({1, 2, 4, num_samples} & set(range(1, num_samples + 1))):
        eligible = [(pid, len(all_summaries[pid]), problem_success_counts[pid])
                    for pid in per_problem_traces if len(all_summaries[pid]) >= k]
        if not eligible:
            continue
        total = sum(pass_at_k(n_actual, c, k) for _, n_actual, c in eligible)
        rate = total / n_problems if n_problems else 0
        pass_at_k_results[k] = {
            "rate": round(rate, 6),
            "solved_estimate": round(rate * n_problems, 1),
            "total": n_problems,
            "eligible": len(eligible),
        }

    solved_any = sum(1 for c in problem_success_counts.values() if c > 0)
    total_input_tokens = sum(
        t.get("total_model_tokens", {}).get("input", 0) for t in all_traces
    )
    total_output_tokens = sum(
        t.get("total_model_tokens", {}).get("output", 0) for t in all_traces
    )
    total_cached_tokens = sum(
        t.get("total_cached_tokens", 0) for t in all_traces
    )
    total_cost_usd = sum(t.get("total_cost_usd", 0.0) for t in all_traces)
    cache_hit_rate = (
        total_cached_tokens / total_input_tokens
        if total_input_tokens else 0.0
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "stage": "skeleton",
        "input": args.input,
        "model_url": args.model_url,
        "model_name": args.model_name,
        "lean_server": args.lean_server,
        "max_turns": args.max_turns,
        "num_samples": num_samples,
        "concurrency": concurrency,
        "temperature": args.temperature,
        "tool_calling": "openrouter (streaming) + skeleton_safeguard",
        "search_server": search_server_url,
        "nl_proof_decompose": args.nl_proof_decompose,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "provider": args.provider,
        "total_problems": n_problems,
        "total_conversations": total_conversations,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_tokens": total_cached_tokens,
        "cache_hit_rate": round(cache_hit_rate, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "solved_any_sample": solved_any,
        "solve_rate_any": solved_any / n_problems if n_problems else 0,
        "pass@k": pass_at_k_results,
        "elapsed_seconds": round(elapsed_total, 2),
        "per_problem_results": all_summaries,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info(
        "DONE: %d/%d skeletons valid (%.1f%%) — %d samples x %d turns, concurrency=%d",
        solved_any, n_problems,
        100 * solved_any / n_problems if n_problems else 0,
        num_samples, args.max_turns, concurrency,
    )
    for k, info in pass_at_k_results.items():
        log.info("  pass@%d = %.1f%% (~%.0f/%d)",
                 k, info["rate"] * 100, info["solved_estimate"], n_problems)
    log.info("Tokens: input=%d (cached=%d, %.1f%% hit) output=%d",
             total_input_tokens, total_cached_tokens,
             100 * cache_hit_rate, total_output_tokens)
    log.info("Total cost (OpenRouter billed): $%.4f", total_cost_usd)
    log.info("Total wall time: %.1fs", elapsed_total)
    log.info("Results: %s", summary_path)


def main():
    # Line-buffer output so progress shows up promptly when piped.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
