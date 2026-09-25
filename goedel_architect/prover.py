"""Theorem proving (paper §3.2): prove every lemma of a blueprint, in parallel.

Each lemma of the blueprint is dispatched to its own agentic conversation.
The prover sees only that lemma and the definitions and lemmas it declared
as dependencies (`graph.build_dependency_block`: every definition verbatim,
direct parents stubbed `:= by sorry` as available facts). It calls
`lean_compile` (and optionally `mathlib_search`) until it closes the goal or
exhausts its turn budget:

  * a submission containing the target is rebuilt under the canonical
    signature (only the `:= by` body is kept) and must pass the safeguards,
    compile, contain no `sorry` / `sorry_using`, and use only foundational
    axioms (`#print axioms`);
  * a snippet without the target is compiled as-is (exploration);
  * with `--allow-negation`, a proof of `<lemma>_negation` registers a
    formal disproof instead;
  * with `--allow-forfeit`, a failed rollout ends with a structured
    diagnosis (STATEMENT_WRONG / PROOF_TOO_HARD, analysis, suggested fix);
    a disproof ends with a STATEMENT_WRONG reflection.

Each lemma is retried sequentially up to `--node-retries` times. Proved
bodies are substituted into the blueprint and the assembled file is checked
end to end. The per-lemma verdicts and diagnoses are the input to blueprint
refinement (`refine.py`).

Input:  a blueprint-generation or refinement output dir (`--skeleton-results`).
Output: `traces.jsonl` (one row per problem sample, with `node_results` and
        the assembled `final_code`), `node_results.jsonl` (per-lemma resume
        checkpoints), `proved_blueprints/` / `unproved_blueprints/` `.lean`
        files, `summary.json`, `session.log` under `--output`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from . import graph
from . import prompts
from .graph import (
    BlueprintNode,
    CountingSemaphore,
    parse_blueprint_file,
    build_dependency_block,
    build_safe_code,
    graft_proof_body,
    parse_axioms_from_result,
    classify_axioms,
    target_has_sorry,
    target_has_sorry_using,
    assemble_final_file,
    get_theorem_name,
    load_skeleton_traces,
    check_balanced_comments,
    strip_lean_comments_and_strings,
    check_theorem_present,
    extract_lean_code_from_text,
    pass_at_k,
    prune_dead_nodes_from_trace,
)
from .lean import (
    DEFAULT_LEAN_SERVER,
    LEAN_TOOL_CEILING_MULTIPLIER,
    TIMEOUT_BUDGET,
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
from .negation import build_negation_formal_statement
from .search import (
    DEFAULT_SEARCH_SERVER,
    MATHLIB_SEARCH_TOOL,
    format_search_response,
    resolve_search_server,
    search_mathlib_async,
)

# Fixed name (not __name__): when run with `python -m`, __name__ is "__main__".
log = logging.getLogger("goedel_architect.prover")


# Hard ceiling on a single streaming chat.completions.create call: the
# timer that kills a silently-hung stream. asyncio.wait_for cancels the
# coroutine, the stream's finally-block closes the response, and
# asyncio.TimeoutError propagates up to `llm.chat_with_retry`, which counts
# it as a transient and retries on a fresh request. Sized at ~7× the median per-turn duration we've
# observed (5–90s) and ~3× the upper end of legitimate heavy-reasoning
# turns (≤3 min): safely above any real turn, comfortably below the
# multi-hour silent hangs we want to terminate.
STREAM_HARD_TIMEOUT_S = 600.0

# ─────────────────────────────────────────────────────────────────────
# Per-node checkpoint: a sidecar `node_results.jsonl` saving each
# `prove_node` winner as soon as it returns, so a killed run can
# resume node-by-node within an in-progress problem instead of
# re-running every node from scratch. `traces.jsonl` is still the
# canonical per-problem output; this file is purely a finer-grained
# resume index.
#
# Format: one JSON object per line, with keys
#   {"problem_id":..., "sample":..., "node_name":..., "result": {…}}
# where `result` is the exact dict `prove_node` returns.
# ─────────────────────────────────────────────────────────────────────
_node_results_path: "Path | None" = None
_node_results_cache: "dict[tuple[str, int, str], dict]" = {}
_node_results_lock = asyncio.Lock()


def _load_node_results(path: "Path") -> int:
    """Populate `_node_results_cache` from an existing node_results.jsonl."""
    if not path.exists():
        return 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                key = (
                    entry["problem_id"],
                    entry["sample"],
                    entry["node_name"],
                )
            except KeyError:
                continue
            _node_results_cache[key] = entry["result"]
    return len(_node_results_cache)


async def _save_node_result(
    problem_id: str, sample_idx: int, node_name: str, result: dict,
) -> None:
    """Append one node-result entry to the sidecar and update the cache."""
    if _node_results_path is None:
        return
    entry = {
        "problem_id": problem_id,
        "sample": sample_idx,
        "node_name": node_name,
        "result": result,
    }
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    async with _node_results_lock:
        with open(_node_results_path, "a") as f:
            f.write(line)
    _node_results_cache[(problem_id, sample_idx, node_name)] = result


# ---------------------------------------------------------------------------
#  Tool definitions
# ---------------------------------------------------------------------------

LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_compile",
        "description": (
            "Compile a Lean 4 code snippet against Mathlib and return "
            "compilation feedback. The code must be a complete, "
            "self-contained Lean 4 file including all imports (e.g. "
            "`import Mathlib`). The system inspects what you submit:\n"
            "- If the snippet contains the target theorem/lemma "
            "declaration with a `:= by` proof body, the system extracts "
            "only your proof body (everything after `:= by`) and grafts "
            "it onto the canonical signature; everything else you wrote "
            "(imports, `set_option`, `open`, definitions, helper or "
            "parent lemmas) is discarded and replaced by the canonical "
            "skeleton header and dependency block. A passing compile "
            "registers a solve. `axiom`, `native_decide`, `sorry`, and "
            "`sorry_using` are not accepted in the proof body.\n"
            "- If the snippet does NOT contain the target theorem (e.g. "
            "`#check`, `example`, helper prototypes), it is compiled "
            "as-given and the raw feedback is returned (including "
            "`#check` / `#print` / `#eval` output). This is exploration "
            "only and cannot register a solve."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Lean 4 code, including all necessary "
                        "imports."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

TOOLS = [LEAN_COMPILE_TOOL, MATHLIB_SEARCH_TOOL]


# ---------------------------------------------------------------------------
#  System prompt (goedel_architect/prompts/theorem_proving*.md)
#
#  The `<error></error>` / `<info></info>` legend lives in the system prompt
#  and is never repeated in tool responses.
#
#  When `allow_negation` is on AND the negation builder produced a
#  parseable template for the current node, `_blueprint_system_prompt`
#  splices a third bullet into the `lean_compile` description naming the
#  parallel `<node>_negation` registration path. With negation off, the
#  prompt is byte-identical to the base prompt.
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = prompts.load("theorem_proving")

# Optional third bullet inserted into the `## Tool use` block when the
# `--allow-negation` pathway is active for the current node.
_NEGATION_BULLET = prompts.load("theorem_proving_negation")

# Extra edit (beyond dropping search mentions) applied when the run has no
# Mathlib search; see `prompts.without_search`.
_NO_SEARCH_EDITS = (
    ("Use `mathlib_search` to find relevant Mathlib lemmas when you need "
     "them. The compiler is a stronger signal source than search — call "
     "`lean_compile` early, even with a partial proof.",
     "Call `lean_compile` early, even with a partial proof."),
)


def _blueprint_system_prompt(*, with_negation: bool = False,
                             with_search: bool = True) -> str:
    """Return the system prompt, optionally including the negation bullet.

    When `with_negation` is False (default), this is byte-identical to
    the base prompt. When True, the `## Tool use` lead-in is
    updated from "two cases" to "three cases", the original-theorem
    bullet's "Only this path can register a solve" is softened to "This
    path registers a solve" (so the two solve-registering bullets don't
    contradict each other), and the negation bullet is spliced in
    between the explore bullet and the `# Feedback format` section.
    `with_search=False` drops the `mathlib_search` guidance.
    """
    base = _BASE_SYSTEM_PROMPT
    if not with_search:
        base = prompts.without_search(base, _NO_SEARCH_EDITS)
    if not with_negation:
        return base
    base = base.replace(
        "two cases automatically", "three cases automatically", 1,
    )
    base = base.replace(
        "Only this path can register a solve",
        "This path registers a solve",
        1,
    )
    marker = "\n\n# Feedback format"
    return base.replace(marker, "\n" + _NEGATION_BULLET + marker, 1)


# The base prompt, recorded on problem-level traces.
SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
#  Unified safeguard rendering — every pre/post-compile rejection
#  produces a list[dict] of {display, message} violations rendered into
#  `Safeguard rejected with 1 violation.` / `with N violations.` plus
#  per-violation `Violation N:` blocks. Same shape as blueprint
#  generation.
# ---------------------------------------------------------------------------

def _highlight_line_span(line_text: str, col: int, end_col: int) -> str:
    """Wrap `line_text[col:end_col]` in `<error></error>` markers."""
    return (
        f"{line_text[:col]}<error>{line_text[col:end_col]}</error>"
        f"{line_text[end_col:]}"
    )


def _violation_at_offsets(
    code: str, start: int, end: int, message: str,
) -> dict:
    """Build a {display, message} violation dict highlighting
    `code[start:end]` on its source line. Spans crossing a newline are
    clipped to end-of-line."""
    line_start = code.rfind('\n', 0, start) + 1
    line_end = code.find('\n', start)
    if line_end == -1:
        line_end = len(code)
    line_text = code[line_start:line_end]
    col = start - line_start
    end_col = min(end - line_start, len(line_text))
    return {
        "display": _highlight_line_span(line_text, col, end_col),
        "message": message,
    }


def _violation_synthetic(display: str, message: str) -> dict:
    """Build a violation dict with a pre-rendered display block. Used
    for absence cases (missing theorem / build_fail / unbalanced
    comments etc.) where there's no offending span in the user's code,
    or for post-compile rejections that need a synthetic display."""
    return {"display": display, "message": message}


def _render_violation_blocks(violations: list[dict]) -> str:
    """Render a list of {display, message} violations into the unified
    `Safeguard rejected ...` response."""
    n = len(violations)
    head = (
        f"Safeguard rejected with {n} violation."
        if n == 1
        else f"Safeguard rejected with {n} violations."
    )
    parts = [head, ""]
    for i, v in enumerate(violations, 1):
        parts.append(f"Violation {i}:")
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
    balanced, msg = check_balanced_comments(code)
    if balanced:
        return None
    # Highlight the first non-blank line of the file as a placeholder
    # location — the imbalance is global and `check_balanced_comments`
    # doesn't pin a specific line. The message names the diagnostic.
    code_lines = code.split('\n')
    first_idx = next(
        (i for i, l in enumerate(code_lines) if l.strip()), 0,
    )
    line_text = code_lines[first_idx] if code_lines else ""
    display = _highlight_line_span(line_text, 0, min(2, len(line_text)))
    return [_violation_synthetic(display, msg or "Unbalanced block comment.")]


def check_missing_theorem_node(
    code: str, theorem_name: str,
) -> list[dict] | None:
    if check_theorem_present(code, theorem_name):
        return None
    code_lines = code.rstrip().split('\n')
    tail_lines = code_lines[-3:] if len(code_lines) > 3 else code_lines
    error_block = (
        f"<error>theorem {theorem_name} ... := by <tactics></error>"
    )
    display = '\n'.join(tail_lines + [error_block])
    message = (
        f"Your code does not contain the required theorem "
        f"`{theorem_name}`. It must appear with the canonical "
        f"signature given in the task and a `:= by <tactics>` body."
    )
    return [_violation_synthetic(display, message)]


_AXIOM_RE = re.compile(r'\baxiom\b')
_NATIVE_DECIDE_RE = re.compile(r'\bnative_decide\b')


def check_forbidden_constructs(code: str) -> list[dict] | None:
    """Reject `axiom` and `native_decide` outside comments/strings.
    Highlights each occurrence with `<error></error>` on its source
    line."""
    sanitised = strip_lean_comments_and_strings(code)
    violations: list[dict] = []
    for m in _AXIOM_RE.finditer(sanitised):
        violations.append(_violation_at_offsets(
            code, m.start(), m.end(),
            "`axiom` declarations are not allowed. Prove the theorem "
            "with real tactics.",
        ))
    if os.environ.get("ALLOW_NATIVE_DECIDE") != "1":
        for m in _NATIVE_DECIDE_RE.finditer(sanitised):
            violations.append(_violation_at_offsets(
                code, m.start(), m.end(),
                "`native_decide` is not allowed. Use `decide`, `norm_num`, "
                "or `omega` instead.",
            ))
    return violations or None


# ---------------------------------------------------------------------------
#  Silently-dropped top-level decls
#
#  `build_safe_code` extracts ONLY the proof body from `theorem
#  <node.name> ... := by <body>` and grafts it onto the canonical
#  `rebuild_dependency_block + signature`. Everything else the model
#  wrote — extra top-level `lemma`, `def`, auxiliary `theorem`, etc. —
#  is silently discarded. If the proof body then references the dropped
#  helper, Lean replies with a misleading `unknown identifier` and the
#  model spends turns debugging Lean instead of moving the helper. Catch
#  pre-compile and reject with the same `Safeguard rejected` shape.
# ---------------------------------------------------------------------------

_HELPER_DECL_RE = re.compile(
    r"^\s*(?:noncomputable\s+|private\s+|protected\s+|@\[\w+[^\]]*\]\s*)*"
    r"(lemma|def|abbrev|structure|inductive|class|instance|theorem)\s+"
    r"(\w+)",
    re.MULTILINE,
)

_BY_KIND_TO_LOCAL_BINDING = {
    "lemma": "have",
    "theorem": "have",
    "def": "let",
    "abbrev": "let",
    "structure": None,
    "inductive": None,
    "class": None,
    "instance": None,
}


def names_in_dep_block(rebuild_block: str) -> frozenset[str]:
    """Names of every top-level decl already present in the canonical
    `rebuild_dependency_block` (defs + sorry'd target parents). A model
    snippet that redeclares one of these is harmless — `graft_proof_body`
    only keeps the target's body, so the redeclaration is silently
    dropped while the canonical version still supplies the same name.
    Used to whitelist parent-stub copies in `scan_dropped_top_level`.
    """
    sanitised = strip_lean_comments_and_strings(rebuild_block)
    return frozenset(m.group(2) for m in _HELPER_DECL_RE.finditer(sanitised))


def scan_dropped_top_level(
    code: str, theorem_name: str,
    allow_names: frozenset[str] = frozenset(),
) -> list[dict] | None:
    """Detect non-header content above the canonical theorem that
    `build_safe_code` would silently discard, plus any extra
    `theorem`/`lemma`/`def`/etc. AFTER the canonical theorem (also
    discarded since `graft_proof_body` strips trailing `#print axioms`
    and re-anchors at the canonical signature). Returns violations or
    None.

    Names in `allow_names` (typically the canonical dep-block decls
    via `names_in_dep_block`) are exempt: redeclaring them is harmless
    because the rebuild reintroduces the same name, so the model's
    body still resolves. Foreign top-level decls remain rejected.
    """
    sanitised = strip_lean_comments_and_strings(code)

    # `(?![\w'])` instead of `\b` — see graph.check_theorem_present
    # for why; Lean 4 identifiers can carry `'` as a continuation char.
    main_re = re.compile(
        r"\b(?:theorem|lemma)\s+" + re.escape(theorem_name) + r"(?![\w'])"
    )
    m_main = main_re.search(sanitised)
    if not m_main:
        # Caught earlier by `check_missing_theorem_node`; skip.
        return None

    # End of import/set_option/open header.
    lines = sanitised.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("--"):
            continue
        if s.startswith(("import ", "set_option ", "open ", "namespace ",
                         "universe ")):
            continue
        header_end = sum(len(l) + 1 for l in lines[:i])
        break

    pre_main = sanitised[header_end:m_main.start()]

    # Skip attribute / modifier lines immediately preceding the main
    # theorem (e.g. `@[simp]\ntheorem T …`).
    attr_only_tail = re.search(
        r"(?:@\[\w+[^\]]*\]\s*|noncomputable\s+|private\s+|protected\s+)+\Z",
        pre_main,
    )
    if attr_only_tail:
        pre_main = pre_main[:attr_only_tail.start()]

    violations: list[dict] = []
    saw_helper_decl = False
    for m in _HELPER_DECL_RE.finditer(pre_main):
        saw_helper_decl = True
        kind, name = m.group(1), m.group(2)
        if kind in ("theorem", "lemma") and name == theorem_name:
            continue
        if name in allow_names:
            # Redeclaration of a dep-block name (a parent stub the
            # model copied from the prompt, or a def). The rebuild
            # supplies the canonical version, so the model's copy is
            # silently dropped without breaking the proof body.
            continue
        offset = header_end + m.start()
        # Locate the keyword span for highlighting.
        kw_local = m.group(0).rfind(kind)
        kw_start = offset + (kw_local if kw_local >= 0 else 0)
        kw_end = kw_start + len(f"{kind} {name}")
        local = _BY_KIND_TO_LOCAL_BINDING.get(kind)
        if local in ("have", "let"):
            if local == "have":
                alt = (
                    f"Move helpers inside the proof body using "
                    f"`have {name} : … := by …`."
                )
            else:
                alt = (
                    f"Move helpers inside the proof body using "
                    f"`let {name} : … := …` or `have {name} : … := …`."
                )
        else:
            alt = (
                "Move it inside the proof body, or restructure to "
                "avoid the top-level declaration."
            )
        violations.append(_violation_at_offsets(
            code, kw_start, kw_end,
            (
                f"top-level `{kind} {name}` is not allowed in submit "
                f"snippets — the safeguard rebuild keeps only the main "
                f"theorem's `:= by` body and supplies everything else "
                f"(imports, `set_option`, `open`, definitions, parent "
                f"lemmas) from the canonical skeleton. {alt}"
            ),
        ))

    # Fallback: non-empty pre_main with no helper-decl matches
    # (e.g. raw `noncomputable section`, `variable …`, `notation …`).
    # Only fires when no helper-decl was seen at all — otherwise the
    # model wrote allowed dep-block redeclarations and we should not
    # flag the same lines as "top-level content other than...".
    if not violations and not saw_helper_decl and pre_main.strip():
        first_token = pre_main.strip().split()[0]
        offset = header_end + pre_main.find(first_token)
        end_off = offset + len(first_token)
        violations.append(_violation_at_offsets(
            code, offset, end_off,
            (
                "top-level content before the main theorem is "
                "discarded by the safeguard rebuild — only the main "
                "theorem's `:= by` body is kept; everything else "
                "(imports, `set_option`, `open`, definitions, parent "
                "lemmas) comes from the canonical skeleton. Move it "
                "inside the proof body, or remove it."
            ),
        ))

    # Also reject any extra `theorem`/`lemma`/etc. AFTER the canonical
    # theorem keyword: `graft_proof_body` cuts at the canonical body's
    # `:= by` and only keeps everything before, so additional decls
    # past that point are dropped on rebuild too.
    post_main = sanitised[m_main.end():]
    by_match = re.search(r":=\s*by\b", post_main)
    if by_match:
        body_start = m_main.end() + by_match.end()
        post_body = sanitised[body_start:]
        body_offset = body_start
    else:
        post_body = post_main
        body_offset = m_main.end()
    for m in _HELPER_DECL_RE.finditer(post_body):
        kind, name = m.group(1), m.group(2)
        if kind in ("theorem", "lemma") and name == theorem_name:
            continue
        if name in allow_names:
            continue
        offset = body_offset + m.start()
        kw_local = m.group(0).rfind(kind)
        kw_start = offset + (kw_local if kw_local >= 0 else 0)
        kw_end = kw_start + len(f"{kind} {name}")
        violations.append(_violation_at_offsets(
            code, kw_start, kw_end,
            (
                f"top-level `{kind} {name}` after the main theorem is "
                f"not allowed — the safeguard rebuild trims everything "
                f"past the canonical `:= by` body. Either inline this "
                f"helper inside the proof body, or remove it."
            ),
        ))

    return violations or None


# ---------------------------------------------------------------------------
#  Extra `import` / `open` detection. Same fail-loud rationale as
#  `scan_dropped_top_level`: `build_safe_code` rebuilds from
#  `rebuild_dependency_block` (which carries the canonical skeleton
#  header) plus only the model's `:= by` body, so any extra `import`
#  or `open` line in the model's snippet is silently discarded. If the
#  proof body relies on the dropped name, Lean reports a misleading
#  `unknown identifier` / `unknown tactic` and the model debugs the
#  wrong thing. `set_option`, `namespace`, and `universe` are not
#  flagged: options don't change name resolution, and the latter two
#  re-anchor cleanly under the canonical header.
# ---------------------------------------------------------------------------

def _normalise_header_line(line: str) -> str:
    return " ".join(line.split())


def _canonical_header_set(rebuild_dependency_block: str) -> set[str]:
    """Normalised `import` and `open` lines from the canonical
    skeleton header at the top of `rebuild_dependency_block`. Used as
    the whitelist for `scan_extra_imports_opens`."""
    sanitised = strip_lean_comments_and_strings(rebuild_dependency_block)
    canon: set[str] = set()
    for line in sanitised.split("\n"):
        s = line.strip()
        if s.startswith(("import ", "open ")):
            canon.add(_normalise_header_line(s))
    return canon


def scan_extra_imports_opens(
    code: str, rebuild_dependency_block: str,
) -> list[dict] | None:
    """Detect `import` / `open` lines in `code` that are not present
    in the canonical skeleton header (top of `rebuild_dependency_block`)
    and would be silently dropped by `build_safe_code`. Returns
    violations or None."""
    canonical = _canonical_header_set(rebuild_dependency_block)
    sanitised = strip_lean_comments_and_strings(code)

    violations: list[dict] = []
    offset = 0
    for raw_line in sanitised.split("\n"):
        s = raw_line.strip()
        # Stop at the first non-header, non-blank line. Mirrors the
        # header walk in `scan_dropped_top_level`.
        if s and not s.startswith((
            "import ", "set_option ", "open ",
            "namespace ", "universe ",
        )):
            break
        if s.startswith(("import ", "open ")):
            normalised = _normalise_header_line(s)
            if normalised not in canonical:
                line_indent = len(raw_line) - len(raw_line.lstrip())
                line_text_len = len(raw_line.rstrip())
                start = offset + line_indent
                end = offset + line_text_len
                if s.startswith("import "):
                    kind = "extra_import"
                    msg = (
                        f"Extra `{normalised}` is not allowed in submit "
                        f"snippets — imports come from the canonical "
                        f"skeleton header. Remove it; anything not "
                        f"transitively available from the canonical "
                        f"imports is unavailable here."
                    )
                else:
                    kind = "extra_open"
                    msg = (
                        f"Extra `{normalised}` is not allowed in submit "
                        f"snippets — opens come from the canonical "
                        f"skeleton header. Remove it, or use "
                        f"fully-qualified names."
                    )
                v = _violation_at_offsets(code, start, end, msg)
                v["type"] = kind
                violations.append(v)
        offset += len(raw_line) + 1
    return violations or None


# ---------------------------------------------------------------------------
#  User prompt (one Lean code block followed by a task marker +
#  signature stub).
# ---------------------------------------------------------------------------

def build_user_prompt(
    node: BlueprintNode,
    dependency_block: str,
    skip_proof_sketch: bool = False,
    negation_signature: str | None = None,
) -> str:
    """Render the per-node user prompt.

    `skip_proof_sketch=True` drops the `-- Proof sketch:` block from the
    rendered task header. The `-- Statement:` block (when present) is
    kept so the model still sees an NL paraphrase of the claim.

    `negation_signature` (when given) is the canonical negation theorem
    block built by `_build_node_negation_signature`. When present, an
    explicit alternate-path clause is appended naming the parallel
    `<node>_negation` theorem the model may instead prove if it has
    compiler-confirmed evidence the original is false.
    """
    code_parts = [dependency_block.rstrip()]

    task_lines = [
        f"-- Prove `{node.name}` below.",
    ]
    if node.informal_statement:
        task_lines.append("-- Statement:")
        for line in node.informal_statement.strip().split('\n'):
            task_lines.append(f"-- {line}")
    if node.informal_proof and not skip_proof_sketch:
        task_lines.append("--")
        task_lines.append("-- Proof sketch:")
        for line in node.informal_proof.strip().split('\n'):
            task_lines.append(f"-- {line}")

    code_parts.append("\n".join(task_lines))

    target_stub = f"{node.signature.rstrip()} := by sorry"
    code_parts.append(target_stub)

    code_block = "\n\n".join(code_parts)
    rendered = f"```lean4\n{code_block}\n```"

    if negation_signature:
        negation_stub = f"{negation_signature.rstrip()} := by sorry"
        rendered += (
            "\n\nIf, after compiler-grounded investigation, you become "
            "confident that the targeted theorem above is false (you "
            "have a concrete counterexample that `lean_compile` "
            "confirms), you may register that conclusion instead by "
            "submitting a proof of the parallel negation theorem "
            "below. A passing compile of either the original theorem "
            "or this negation theorem registers a solve. Do not "
            "preemptively switch to the negation path — only when the "
            "compiler has corroborated a counterexample.\n\n"
            f"```lean4\n{negation_stub}\n```"
        )
    return rendered


# ---------------------------------------------------------------------------
#  Per-node negation signature builder (when --allow-negation is on).
#
#  Wraps `negation.build_negation_formal_statement` for the per-node
#  setting. The helper expects a full formal-statement block; for nodes we
#  synthesise a minimal block from `node.signature := by sorry`, run the
#  helper, and strip back to the bare negation theorem line (no header).
#
#  Returns None when the helper can't parse the signature (term-mode
#  body, unbalanced parens, no separating `:` colon, etc.). The caller
#  then falls back to the original-only path for that node.
# ---------------------------------------------------------------------------

def _build_node_negation_signature(node: "BlueprintNode") -> str | None:
    """Return the bare negation signature
    `theorem <node>_negation : ¬ (∀ binders, body)` for a blueprint
    node, or None if the original signature can't be parsed.

    Only used when `--allow-negation` is on. The return value matches
    the `BlueprintNode.signature` convention (see graph.BlueprintNode
    — everything before `:=`) so it can be dropped into a
    `dataclasses.replace(node, signature=...)` and grafted by
    `graft_proof_body` without producing a doubled `:= by sorry := by`
    body. Callers that display it as a stub append `:= by sorry`
    themselves, mirroring the `target_stub` construction.
    """
    fake_stmt = f"{node.signature.rstrip()} := by sorry\n"
    full = build_negation_formal_statement(fake_stmt, node.name)
    if full is None:
        return None
    full = full.strip()
    # `build_negation_formal_statement` always emits its block ending
    # in `:= by sorry`; strip it back off to honour the bare-signature
    # convention.
    return re.sub(r"\s*:=\s*by\s+sorry\s*$", "", full).rstrip()


# ---------------------------------------------------------------------------
#  --allow-forfeit: structured wrap-up when a node rollout terminates
#  without a proof (goedel_architect/prompts/theorem_proving_forfeit.md),
#  and the mirror-image reflection after a registered disproof
#  (theorem_proving_disproof_reflection.md). The disproof reflection fixes
#  the diagnosis to STATEMENT_WRONG and asks for the same Analysis /
#  Suggested Fix sections, so blueprint refinement can use a model-authored
#  explanation instead of boilerplate. It always runs when --allow-negation
#  registers a solve.
# ---------------------------------------------------------------------------

FORFEIT_PROMPT = prompts.load("theorem_proving_forfeit")
DISPROOF_REFLECTION_PROMPT = prompts.load("theorem_proving_disproof_reflection")


def _parse_forfeit(text: str) -> dict | None:
    """Store the raw forfeit output. No structured field extraction —
    downstream consumers are LLM agents that can read the markdown
    directly. The whole content is stored in `forfeit_reason` as-is.
    """
    if not text:
        return None
    return {"forfeit_reason": text.strip()}


def _messages_added_by_turn(turn: dict) -> int:
    """Number of items appended to `messages` while processing one loop
    iteration of `_run_node_inner`.

    Mirrors the harness control flow:
      - context_full / model_error: break before any message is appended
        → 0.
      - all other statuses: assistant message + one entry per tool call
        (native or text-fallback). The fallback path appends a single
        user-role <tool_response>; tool_calls is still set with one
        entry.
    """
    status = turn.get("status")
    if status in ("context_full", "model_error"):
        return 0
    return 1 + len(turn.get("tool_calls") or [])


def _message_index_to_turn_index(
    cut_msg_idx: int, turns: list, head_msgs: int = 2,
) -> int:
    """Map a message-list cut index to the largest turn index whose
    contributions fit entirely within messages[: cut_msg_idx + 1].

    `head_msgs` is the number of messages preceding turn 0 (system +
    initial user_prompt = 2). Returns -1 if the cut lands inside the
    head section.
    """
    if cut_msg_idx < head_msgs - 1:
        return -1
    cum = head_msgs
    last_fully_included = -1
    for i, turn in enumerate(turns):
        if turn.get("status") in ("forfeit_declaration", "disproof_reflection"):
            break
        cum += _messages_added_by_turn(turn)
        if cum - 1 <= cut_msg_idx:
            last_fully_included = i
        else:
            break
    return last_fully_included


def _find_forfeit_injection_point(
    messages: list,
    tokenizer,
    tools_overhead: int,
    max_model_len: int,
    forfeit_budget: int,
) -> tuple[int, int]:
    """Find the largest k such that messages[:k+1] is a valid prefix
    leaving at least `forfeit_budget` tokens of headroom in
    `max_model_len`.

    Validity: the cut must land at a turn boundary. Since each non-final
    turn ends either with the assistant's bare reply (no tool calls,
    only possible at the very end) or with the last tool/user response
    in its cluster, "messages[k+1] is an assistant message" — or k is
    the very last index — flags a valid boundary.

    Returns (k, prompt_tokens). (-1, 0) if no valid prefix fits.
    """
    target = max_model_len - forfeit_budget
    n = len(messages)
    for k in range(n - 1, -1, -1):
        if k < n - 1 and messages[k + 1].get("role") != "assistant":
            continue
        prompt = count_prompt_tokens(tokenizer, messages[: k + 1], tools_overhead)
        if prompt <= target:
            return k, prompt
    return -1, 0


async def _run_wrapup_turn(
    *,
    client,
    model_name: str,
    messages: list,
    tools: list,
    extra_body: dict | None,
    tag: str,
    trace: dict,
    trigger: str,
    prompt: str,
    result_key: str,
    turn_status: str,
    tokenizer,
    tools_overhead: int,
    max_model_len: int,
    forfeit_budget: int = 8192,
) -> None:
    """Inject a structured wrap-up turn at a valid turn boundary.

    Shared machinery for two callers:
      * `_run_forfeit_turn` — fires after a failed rollout, prompts the
        model with FORFEIT_PROMPT and stores the parsed
        STATEMENT_WRONG / PROOF_TOO_HARD diagnosis on
        `trace["forfeit_declaration"]`.
      * `_run_disproof_reflection_turn` — fires after the parallel
        negation pathway registers a solve, prompts the model with
        DISPROOF_REFLECTION_PROMPT (Diagnosis pinned to STATEMENT_WRONG)
        and stores the parsed reflection on
        `trace["disproof_reflection"]`.

    Walk back from the end of the conversation to the latest valid turn
    boundary that still has at least `forfeit_budget` tokens of headroom.
    Truncate to that prefix, replace the final tool/user message with
    `prompt` (or append it as a fresh user message if the last turn
    ended with a bare assistant), and call the model with the full
    remaining context as output budget — uncapped. Tool calls are
    forbidden in plain English; we deliberately do NOT pass
    `tool_choice="none"` so that the prefix cache stays warm.

    Errors are swallowed onto `trace[result_key]["error"]` — the
    original failure (or success) status on the last real turn remains
    authoritative.
    """
    cut_idx, prefix_tokens = _find_forfeit_injection_point(
        messages, tokenizer, tools_overhead, max_model_len, forfeit_budget,
    )
    if cut_idx < 0:
        log.error(
            "[%s] %s: no valid injection point fits "
            "forfeit_budget=%d in max_model_len=%d (head too large)",
            tag, result_key, forfeit_budget, max_model_len,
        )
        trace[result_key] = {
            "trigger": trigger,
            "error": (
                f"no valid injection point "
                f"(forfeit_budget={forfeit_budget})"
            ),
        }
        return

    n_dropped = len(messages) - (cut_idx + 1)
    truncated = list(messages[: cut_idx + 1])
    last = truncated[-1]
    last_role = last.get("role")
    if last_role == "tool":
        truncated[-1] = dict(last)
        truncated[-1]["content"] = prompt
        injection_kind = "replaced_tool_result"
    elif last_role == "user":
        truncated[-1] = dict(last)
        truncated[-1]["content"] = prompt
        injection_kind = "replaced_user"
    else:
        truncated.append({"role": "user", "content": prompt})
        injection_kind = "appended_as_user"

    prompt_tokens_after = count_prompt_tokens(
        tokenizer, truncated, tools_overhead,
    )
    out_budget = compute_max_tokens(
        prompt_tokens_after, max_model_len, None,
    )

    cut_turn_index = _message_index_to_turn_index(
        cut_idx, trace.get("turns", []),
    )

    if out_budget <= 0:
        log.error(
            "[%s] %s: out_budget=%d after injection (prefix=%d, "
            "max_model_len=%d) — wrap-up prompt pushed prefix past the "
            "context window; aborting wrap-up",
            tag, result_key, out_budget, prompt_tokens_after, max_model_len,
        )
        trace[result_key] = {
            "trigger": trigger,
            "error": (
                f"out_budget={out_budget} after injection "
                f"(prefix={prompt_tokens_after}, "
                f"max_model_len={max_model_len})"
            ),
            "injection": {
                "cut_message_index": cut_idx,
                "cut_turn_index": cut_turn_index,
                "messages_dropped": n_dropped,
                "prefix_tokens_before_inject": prefix_tokens,
                "prefix_tokens_after_inject": prompt_tokens_after,
                "out_budget": out_budget,
                "forfeit_budget": forfeit_budget,
            },
        }
        return

    log.info(
        "[%s] %s: trigger=%s cut_msg=%d/%d (dropped=%d, kept turns=%d) "
        "prefix_tokens=%d after_inject=%d out_budget=%d",
        tag, result_key, trigger, cut_idx, len(messages) - 1, n_dropped,
        cut_turn_index + 1, prefix_tokens, prompt_tokens_after, out_budget,
    )

    injection_meta = {
        "kind": injection_kind,
        "cut_message_index": cut_idx,
        "cut_turn_index": cut_turn_index,
        "messages_dropped": n_dropped,
        "prefix_tokens_before_inject": prefix_tokens,
        "prefix_tokens_after_inject": prompt_tokens_after,
        "out_budget": out_budget,
        "forfeit_budget": forfeit_budget,
    }

    MAX_FORFEIT_ATTEMPTS = 5

    def _is_clean(text: str, tool_calls: list) -> bool:
        if tool_calls:
            return False
        if not text:
            return False
        if "## Diagnosis" not in text:
            return False
        return True

    attempts: list[dict] = []
    cumulative_pt = 0
    cumulative_ct = 0
    cumulative_cached = 0
    cumulative_cost = 0.0
    accepted_idx: int | None = None

    for attempt_idx in range(MAX_FORFEIT_ATTEMPTS):
        try:
            resp = await chat_with_retry(
                client,
                tag=tag,
                hard_timeout=STREAM_HARD_TIMEOUT_S,
                model=model_name,
                messages=truncated,
                tools=tools,
                max_tokens=out_budget,
                extra_body=extra_body or None,
            )
        except Exception as e:
            log.error(
                "[%s] %s attempt %d failed: %s",
                tag, result_key, attempt_idx, e,
            )
            if attempt_idx == 0 and not attempts:
                trace[result_key] = {
                    "trigger": trigger,
                    "error": str(e),
                    "injection": injection_meta,
                }
                return
            break

        msg = resp.choices[0].message
        text = msg.content or ""
        reasoning_text = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or ""
        )
        finish = resp.choices[0].finish_reason

        raw_tcs = getattr(msg, "tool_calls", None) or []
        tool_calls_record = []
        for tc in raw_tcs:
            fn = getattr(tc, "function", None)
            nm = getattr(fn, "name", "") if fn else ""
            args_raw = getattr(fn, "arguments", "") if fn else ""
            try:
                args = (
                    json.loads(args_raw)
                    if isinstance(args_raw, str) else args_raw
                )
            except Exception:
                args = args_raw
            tool_calls_record.append({
                "id": getattr(tc, "id", None),
                "name": nm,
                "arguments": args,
            })

        pt = (resp.usage.prompt_tokens or 0) if resp.usage else 0
        ct = (resp.usage.completion_tokens or 0) if resp.usage else 0
        cached = (
            (getattr(resp.usage, "cached_tokens", 0) or 0)
            if resp.usage else 0
        )
        cost = (
            (getattr(resp.usage, "cost_usd", 0.0) or 0.0)
            if resp.usage else 0.0
        )

        cumulative_pt += pt
        cumulative_ct += ct
        cumulative_cached += cached
        cumulative_cost += cost
        trace["total_model_tokens"]["input"] += pt
        trace["total_model_tokens"]["output"] += ct
        trace["total_cached_tokens"] += cached
        trace["total_cost_usd"] += cost

        attempts.append({
            "attempt": attempt_idx,
            "finish_reason": finish,
            "content": text,
            "reasoning_content": reasoning_text,
            "tool_calls": tool_calls_record,
            "billed_prompt_tokens": pt,
            "completion_tokens": ct,
            "cached_tokens": cached,
            "cost_usd": cost,
        })

        if _is_clean(text, raw_tcs):
            accepted_idx = attempt_idx
            log.info(
                "[%s] %s clean on attempt %d/%d "
                "(finish=%s, content=%dch)",
                tag, result_key, attempt_idx + 1, MAX_FORFEIT_ATTEMPTS,
                finish, len(text),
            )
            break

        log.info(
            "[%s] %s attempt %d/%d not clean (finish=%s, "
            "content=%dch, tool_calls=%d) — retrying",
            tag, result_key, attempt_idx + 1, MAX_FORFEIT_ATTEMPTS,
            finish, len(text), len(tool_calls_record),
        )

    if not attempts:
        return

    chosen = (
        attempts[accepted_idx] if accepted_idx is not None else attempts[-1]
    )
    final_text = chosen["content"]
    final_reasoning = chosen["reasoning_content"]
    final_tool_calls = chosen["tool_calls"]
    final_finish = chosen["finish_reason"]
    parsed = _parse_forfeit(final_text)

    trace[result_key] = {
        "trigger": trigger,
        "raw": final_text,
        "reasoning": final_reasoning,
        "parsed": parsed,
        "tool_calls": final_tool_calls,
        "tokens": {"input": cumulative_pt, "output": cumulative_ct},
        "injection": {**injection_meta, "prompt": prompt},
        "attempts": attempts,
        "accepted_attempt": accepted_idx,
    }
    trace["turns"].append({
        "turn": len(trace["turns"]) + 1,
        "status": turn_status,
        "trigger": trigger,
        "forfeit_prompt": prompt,
        "forfeit_injection_kind": injection_kind,
        "forfeit_truncation": {
            "cut_turn_index": cut_turn_index,
            "cut_message_index": cut_idx,
            "messages_dropped": n_dropped,
        },
        "raw_content": (
            f"{final_reasoning}\n\n{final_text}"
            if final_reasoning else final_text
        ),
        "content": final_text,
        "reasoning_content": final_reasoning,
        "tool_calls": final_tool_calls,
        "finish_reason": final_finish,
        "billed_prompt_tokens": cumulative_pt,
        "max_tokens": out_budget,
        "cached_tokens": cumulative_cached,
        "cost_usd": cumulative_cost,
        "attempts": attempts,
        "accepted_attempt": accepted_idx,
    })


async def _run_forfeit_turn(
    *,
    client,
    model_name: str,
    messages: list,
    tools: list,
    extra_body: dict | None,
    tag: str,
    trace: dict,
    trigger: str,
    tokenizer,
    tools_overhead: int,
    max_model_len: int,
    forfeit_budget: int = 8192,
) -> None:
    """--allow-forfeit wrap-up. Thin wrapper over `_run_wrapup_turn`."""
    await _run_wrapup_turn(
        client=client,
        model_name=model_name,
        messages=messages,
        tools=tools,
        extra_body=extra_body,
        tag=tag,
        trace=trace,
        trigger=trigger,
        prompt=FORFEIT_PROMPT,
        result_key="forfeit_declaration",
        turn_status="forfeit_declaration",
        tokenizer=tokenizer,
        tools_overhead=tools_overhead,
        max_model_len=max_model_len,
        forfeit_budget=forfeit_budget,
    )


async def _run_disproof_reflection_turn(
    *,
    client,
    model_name: str,
    messages: list,
    tools: list,
    extra_body: dict | None,
    tag: str,
    trace: dict,
    tokenizer,
    tools_overhead: int,
    max_model_len: int,
    forfeit_budget: int = 8192,
) -> None:
    """--allow-negation post-success reflection. Thin wrapper over
    `_run_wrapup_turn`. Always fires when negation registers a solve.

    The successful-negation case typically has plenty of context-window
    headroom, so the cut usually lands at the very end of `messages`
    (the SUCCESS tool response is replaced with the reflection prompt
    in place — keeping the entire prefix cache-warm). If the rollout
    happened to be near the context limit, the same backwards-walk used
    by forfeit will pick the last valid boundary that still fits.
    """
    await _run_wrapup_turn(
        client=client,
        model_name=model_name,
        messages=messages,
        tools=tools,
        extra_body=extra_body,
        tag=tag,
        trace=trace,
        trigger="negation_registered",
        prompt=DISPROOF_REFLECTION_PROMPT,
        result_key="disproof_reflection",
        turn_status="disproof_reflection",
        tokenizer=tokenizer,
        tools_overhead=tools_overhead,
        max_model_len=max_model_len,
        forfeit_budget=forfeit_budget,
    )


# ---------------------------------------------------------------------------
#  Per-node lean_compile handler (unified feedback)
# ---------------------------------------------------------------------------

async def _handle_lean_compile_node(
    code: str,
    lean_server_url: str,
    lean_timeout: int,
    node: BlueprintNode,
    rebuild_dependency_block: str,
    tag: str,
    turn_idx: int,
    turn_record: dict,
    trace: dict,
    negation_name: str | None = None,
    negation_signature: str | None = None,
) -> str:
    """Handle one lean_compile tool call for a target node.

    Three paths, auto-routed on what the snippet contains:
        * canonical target -> safeguarded submit path (rebuild under
          the canonical signature; registers a solve).
        * canonical NEGATION target (only when `--allow-negation` is
          on AND the negation builder produced a parseable template)
          -> safeguarded submit_negation path (rebuild under the
          negation signature). A complete proof here registers a solve
          and sets `trace.solved_via_negation = True`.
        * neither -> explore path (compile as-given with no rebuild
          and no safeguard checks; never registers a solve, surfaces
          `#check` / `#print` / `#eval` output).

    All submit / submit_negation pre-compile rejections render through
    `_render_violation_blocks` as `Safeguard rejected with N
    violation(s).`. Compile failures use `format_compile_response`.
    Post-compile rejections (custom axioms, target uses bare `sorry`,
    target uses `sorry_using`) also render through
    `_render_violation_blocks` so all rejection gates read consistently
    to the model.
    """
    code_lines = len(code.split('\n'))

    # Route: original-theorem submission takes precedence over negation
    # submission if (somehow) both names appear in the same snippet —
    # registering the original is the model's primary contract. Negation
    # routing only kicks in when --allow-negation built a parseable
    # template AND the snippet contains the negation theorem AND not the
    # original.
    has_orig = check_theorem_present(code, node.name)
    has_neg = (
        not has_orig
        and negation_name is not None
        and negation_signature is not None
        and check_theorem_present(code, negation_name)
    )

    # Route: theoremless / negation-less snippets are exploration; never
    # register a solve.
    if not has_orig and not has_neg:
        turn_record["mode"] = "explore"
        turn_record["theoremless_explore"] = True
        log.info(
            "[%s] turn %d: lean_compile EXPLORE mode (%d lines, node=%s)",
            tag, turn_idx + 1, code_lines, node.name,
        )
        log.debug(
            "[%s] turn %d: === MODEL CODE (explore) ===\n%s",
            tag, turn_idx + 1, code,
        )
        try:
            lean_result = await check_lean_async(
                code, lean_server_url, lean_timeout,
            )
            analysis = analyse_result(lean_result)
            # Defence-in-depth: if `check_lean_async`'s retry logic didn't
            # catch a server-side transient error, the structured error
            # would otherwise leak through `format_compile_response` as
            # `Lean server error: <detail>` tagged with status
            # `explore_compile_fail`. Report the unified `Lean server
            # error.` body instead, so downstream filters keyed on
            # `status == "lean_error"` see it as transport, not
            # compile-fail.
            if analysis["system_error"]:
                log.error(
                    "[%s] Lean server error turn %d (explore, structured): %s",
                    tag, turn_idx + 1, analysis["system_error"],
                )
                trace["total_lean_time"] += analysis["time"]
                turn_record["lean_result"] = lean_result
                turn_record["status"] = "lean_error"
                turn_record["error"] = analysis["system_error"]
                return "Lean server error."
            base_response = format_compile_response(
                code, lean_result, include_info_messages=True,
            )
            trace["total_lean_time"] += analysis["time"]
            turn_record["lean_result"] = lean_result
            turn_record["compilation_pass"] = analysis["pass_"]
            turn_record["compilation_complete"] = analysis["complete"]
            # Deliberately do not set n_errors on explore turns.
            turn_record["status"] = (
                "explore_compile_pass" if analysis["pass_"]
                else "explore_compile_fail"
            )
            # On a successful explore compile, splice an explore clause
            # into the first (status) line so the model can tell the
            # call did not register a solve. On FAILED compiles the
            # error message itself is the relevant signal.
            if analysis["complete"] or analysis["pass_"]:
                if negation_name:
                    clause = (
                        f" Neither `{node.name}` nor `{negation_name}` "
                        "present; does not register a solve."
                    )
                else:
                    clause = (
                        f" Target `{node.name}` not present; "
                        "does not register a solve."
                    )
                first_nl = base_response.find("\n")
                if first_nl == -1:
                    tool_response_text = base_response + clause
                else:
                    tool_response_text = (
                        base_response[:first_nl]
                        + clause
                        + base_response[first_nl:]
                    )
            else:
                tool_response_text = base_response
            log.debug(
                "[%s] turn %d: === EXPLORE FEEDBACK ===\n%s",
                tag, turn_idx + 1, tool_response_text,
            )
            return tool_response_text
        except Exception as e:
            log.error(
                "[%s] Lean error turn %d (explore): %s",
                tag, turn_idx + 1, e,
            )
            turn_record["status"] = "lean_error"
            turn_record["error"] = str(e)
            return "Lean server error."

    # Decide submission mode: submit (canonical theorem present) vs.
    # submit_negation (negation theorem present, allowed by caller).
    # Original-theorem submission takes precedence — the `has_orig` /
    # `has_neg` route logic above already enforced that.
    if has_neg:
        mode = "submit_negation"
        active_name = negation_name
        active_node = dataclasses.replace(
            node, name=negation_name, signature=negation_signature,
        )
    else:
        mode = "submit"
        active_name = node.name
        active_node = node
    turn_record["mode"] = mode

    dep_block_names = names_in_dep_block(rebuild_dependency_block)
    pre_checks: list[tuple[str, list[dict] | None]] = [
        ("unbalanced_comments",
         check_unbalanced_block_comment(code)),
        ("missing_theorem",
         check_missing_theorem_node(code, active_name)),
        ("safeguard_precheck",
         check_forbidden_constructs(code)),
        ("dropped_top_level",
         scan_dropped_top_level(code, active_name, dep_block_names)),
        ("extra_imports_opens",
         scan_extra_imports_opens(code, rebuild_dependency_block)),
    ]
    for status, violations in pre_checks:
        if not violations:
            continue
        n_v = len(violations)
        log.warning(
            "[%s] %s on turn %d (%d %s)",
            tag, status, turn_idx + 1, n_v,
            "violation" if n_v == 1 else "violations",
        )
        turn_record["status"] = status
        turn_record["compilation_pass"] = False
        turn_record["compilation_complete"] = False
        if status == "safeguard_precheck":
            tags: list[str] = []
            for v in violations:
                disp = v.get("display", "")
                if "axiom" in disp and "native_decide" not in disp:
                    tags.append("axiom")
                elif "native_decide" in disp:
                    tags.append("native_decide")
                else:
                    tags.append("forbidden")
            turn_record["safeguard_violations"] = tags
        if status == "dropped_top_level":
            turn_record["safeguard_violations"] = [
                "top_level_decl"
            ] * n_v
        if status == "extra_imports_opens":
            turn_record["safeguard_violations"] = [
                v.get("type", "extra_header") for v in violations
            ]
        return _render_violation_blocks(violations)

    # Build safe code (graft model's proof onto canonical signature).
    # `active_node` is `node` for the original-theorem path or a clone
    # with the `_negation` name + signature for the negation path; both
    # share the same rebuild dependency block.
    safe_code, proof_body = build_safe_code(
        code, rebuild_dependency_block, active_node,
    )
    if safe_code is None:
        log.warning(
            "[%s] safeguard_build_fail on turn %d (no extractable proof body)",
            tag, turn_idx + 1,
        )
        turn_record["status"] = "safeguard_build_fail"
        turn_record["compilation_pass"] = False
        display = (
            f"<error>theorem {active_name} ... := by <tactics></error>"
        )
        message = (
            f"Could not extract a `:= by …` tactic-mode proof body for "
            f"`{active_name}`. The safeguard only handles tactic-mode "
            f"proofs (`theorem {active_name} … := by <tactics>`); "
            f"term-mode submissions cannot be rebuilt under the "
            f"canonical signature. Rewrite as `… := by exact <term>` "
            f"or `… := by <tactics>`."
        )
        return _render_violation_blocks([
            _violation_synthetic(display, message),
        ])
    turn_record["safe_code"] = safe_code
    turn_record["proof_body"] = proof_body

    log.info(
        "[%s] turn %d: lean_compile (%d lines, node=%s, mode=%s)",
        tag, turn_idx + 1, code_lines, active_name, mode,
    )
    log.debug(
        "[%s] turn %d: === CODE ===\n%s",
        tag, turn_idx + 1, safe_code,
    )

    try:
        lean_result = await check_lean_async(
            safe_code, lean_server_url, lean_timeout,
        )
        analysis = analyse_result(lean_result)
        # Defence-in-depth: see the explore-branch comment above. A server
        # transient that bypassed `check_lean_async`'s retry logic would
        # otherwise reach the model as `Lean server error: <detail>`
        # tagged `compile_fail`.
        if analysis["system_error"]:
            log.error(
                "[%s] Lean server error turn %d (submit, structured): %s",
                tag, turn_idx + 1, analysis["system_error"],
            )
            trace["total_lean_time"] += analysis["time"]
            turn_record["lean_result"] = lean_result
            turn_record["status"] = "lean_error"
            turn_record["error"] = analysis["system_error"]
            return "Lean server error."
        trace["total_lean_time"] += analysis["time"]
        turn_record["lean_result"] = lean_result
        turn_record["compilation_pass"] = analysis["pass_"]
        turn_record["compilation_complete"] = analysis["complete"]
        turn_record["n_errors"] = (
            0 if analysis["pass_"] else len(analysis["errors"])
        )
        lean_time = analysis["time"]

        if not analysis["pass_"]:
            # `format_compile_response` produces the unified
            # `Compilation FAILED with N error(s)[ (M distinct)].` +
            # per-error block rendering. Sorry warnings from parent
            # `by sorry` stubs are not surfaced (the error rendering
            # only walks `analysis["errors"]`); target-body sorry is
            # checked post-compile below.
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
            log.debug("[%s] turn %d: === FEEDBACK ===\n%s",
                      tag, turn_idx + 1, tool_response_text)
            return tool_response_text

        # Compile succeeded. Run post-compile rejection gates — all
        # render through the unified `Safeguard rejected …` shape.

        # Custom axiom check.
        axioms = parse_axioms_from_result(lean_result)
        classification, extra = classify_axioms(axioms)
        turn_record["axiom_classification"] = classification
        if extra:
            turn_record["extra_axioms"] = sorted(extra)

        if classification == "CUSTOM_AXIOM":
            log.warning(
                "[%s] turn %d: CUSTOM AXIOM — %s (%.1fs)",
                tag, turn_idx + 1, extra, lean_time,
            )
            turn_record["status"] = "custom_axiom"
            turn_record["compilation_pass"] = False
            extra_names = ", ".join(f"`{a}`" for a in sorted(extra))
            display = f"<error>#print axioms {active_name}</error>"
            message = (
                f"The soundness check on `{active_name}` reports "
                f"non-foundational axiom(s): {extra_names}. Your proof "
                f"depends on `axiom` declarations or `native_decide` "
                f"(transitively). Replace them with real tactics."
            )
            return _render_violation_blocks([
                _violation_synthetic(display, message),
            ])

        # Target sorry check (bare `sorry` inside the target body).
        has_sorry = target_has_sorry(
            safe_code, active_name, analysis["sorries"],
        )
        turn_record["target_sorry"] = has_sorry
        if has_sorry:
            log.info(
                "[%s] turn %d: TARGET SORRY — compiles but target body "
                "uses `sorry` (sorry_count=%d, axioms=%s, %.1fs)",
                tag, turn_idx + 1, len(analysis["sorries"]),
                classification, lean_time,
            )
            turn_record["status"] = "target_sorry"
            turn_record["compilation_pass"] = False
            display = f"<error>theorem {active_name} … := by … sorry …</error>"
            message = (
                f"Compilation passed, but the body of `{active_name}` "
                f"still contains `sorry`. The target theorem must be "
                f"closed with real tactics — `sorry` in the target body "
                f"is not accepted (sorries in dependency stubs above "
                f"are fine and ignored by the checker)."
            )
            return _render_violation_blocks([
                _violation_synthetic(display, message),
            ])

        # Target sorry_using check — Architect's `sorry_using` macro
        # admits axiomatically without emitting a Lean sorry warning,
        # so `target_has_sorry` doesn't catch it.
        has_sorry_using = target_has_sorry_using(safe_code, active_name)
        turn_record["target_sorry_using"] = has_sorry_using
        if has_sorry_using:
            log.info(
                "[%s] turn %d: TARGET SORRY_USING — proof uses "
                "sorry_using inside target body (%.1fs)",
                tag, turn_idx + 1, lean_time,
            )
            turn_record["status"] = "target_sorry_using"
            turn_record["compilation_pass"] = False
            display = (
                f"<error>theorem {active_name} … := by … sorry_using […] …</error>"
            )
            message = (
                f"`sorry_using` is only allowed as the body of an "
                f"UNSOLVED dependency stub, not as a tactic in the proof "
                f"you are writing. Replace the `sorry_using [...]` calls "
                f"in `{active_name}`'s body with real tactics."
            )
            return _render_violation_blocks([
                _violation_synthetic(display, message),
            ])

        # All gates passed.
        via_neg = (mode == "submit_negation")
        log.info(
            "[%s] turn %d: SUCCESS%s — %s proved "
            "(sorry_count=%d, axioms=%s, %.1fs)",
            tag, turn_idx + 1,
            " (via negation)" if via_neg else "",
            active_name, len(analysis["sorries"]),
            classification, lean_time,
        )
        turn_record["status"] = "success"
        trace["success"] = True
        trace["final_code"] = code
        trace["grafted"] = safe_code
        trace["proof_body"] = proof_body
        if via_neg:
            turn_record["success_kind"] = "negation"
            trace["solved_via_negation"] = True
            return (
                f"Compilation SUCCESSFUL. `{active_name}` proved (via "
                f"negation — registered as a solve of `{node.name}`)."
            )
        return f"Compilation SUCCESSFUL. `{active_name}` proved."

    except Exception as e:
        log.error("[%s] Lean error turn %d: %s", tag, turn_idx + 1, e)
        turn_record["status"] = "lean_error"
        turn_record["error"] = str(e)
        # Don't pretend a transport error has structured content. The
        # client-level retry already covers transient Lean transport
        # drops; if it gets here, the one-line status is what the model
        # can read.
        return "Lean server error."


# ---------------------------------------------------------------------------
#  Per-node agentic conversation
# ---------------------------------------------------------------------------

async def run_node(
    client: AsyncOpenAI,
    model_name: str,
    node: BlueprintNode,
    prompt_dependency_block: str,
    rebuild_dependency_block: str,
    lean_server_url: str,
    search_server_url: str | None,
    semaphore: "CountingSemaphore",
    tokenizer,
    tools_overhead: int,
    max_turns: int,
    temperature: float,
    lean_timeout: int,
    tag: str,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_output_tokens: int | None = None,
    extra_body: dict | None = None,
    skip_proof_sketch: bool = False,
    allow_negation: bool = False,
    allow_forfeit: bool = False,
) -> dict:
    async with semaphore:
        return await _run_node_inner(
            client, model_name, node,
            prompt_dependency_block, rebuild_dependency_block,
            lean_server_url, search_server_url,
            tokenizer, tools_overhead, max_turns, temperature,
            lean_timeout, tag, max_model_len, max_output_tokens,
            extra_body, skip_proof_sketch,
            allow_negation, allow_forfeit,
        )


async def _run_node_inner(
    client: AsyncOpenAI,
    model_name: str,
    node: BlueprintNode,
    prompt_dependency_block: str,
    rebuild_dependency_block: str,
    lean_server_url: str,
    search_server_url: str | None,
    tokenizer,
    tools_overhead: int,
    max_turns: int,
    temperature: float,
    lean_timeout: int,
    tag: str,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_output_tokens: int | None = None,
    extra_body: dict | None = None,
    skip_proof_sketch: bool = False,
    allow_negation: bool = False,
    allow_forfeit: bool = False,
) -> dict:
    # Optional parallel negation pathway (--allow-negation). When on AND
    # the node's signature is parseable, the model may instead register
    # a proof of `{node.name}_negation : ¬ (∀ binders, body)`. Falls back
    # silently to the original-only path when the builder can't parse
    # the signature (term-mode body, unbalanced parens, etc.).
    negation_name: str | None = None
    negation_signature: str | None = None
    if allow_negation:
        negation_signature = _build_node_negation_signature(node)
        if negation_signature is not None:
            negation_name = f"{node.name}_negation"
        else:
            log.warning(
                "[%s] negation builder failed to parse the signature; "
                "running this node with the original-only path",
                tag,
            )

    user_prompt = build_user_prompt(
        node, prompt_dependency_block,
        skip_proof_sketch=skip_proof_sketch,
        negation_signature=negation_signature,
    )

    active_tools = [LEAN_COMPILE_TOOL]
    if search_server_url:
        active_tools.append(MATHLIB_SEARCH_TOOL)

    # System prompt: when on, splice in the negation bullet. When off,
    # byte-identical to the base prompt so cached prefixes remain
    # compatible across runs.
    system_prompt = _blueprint_system_prompt(
        with_negation=bool(negation_name),
        with_search=bool(search_server_url),
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    trace: dict = {
        "node_name": node.name,
        "kind": node.kind,
        "depends_on": node.depends_on,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "negation_name": negation_name,
        "negation_signature": negation_signature,
        "max_turns": max_turns,
        "turns": [],
        "success": False,
        "solved_via_negation": False,
        "total_model_tokens": {"input": 0, "output": 0},
        "total_cached_tokens": 0,
        "total_cost_usd": 0.0,
        "total_lean_time": 0.0,
        "total_search_calls": 0,
    }

    t0 = time.time()

    for turn_idx in range(max_turns):
        turn_record: dict = {"turn": turn_idx + 1}

        prompt_tokens = count_prompt_tokens(
            tokenizer, messages, tools_overhead,
        )
        max_tokens = compute_max_tokens(
            prompt_tokens, max_model_len, max_output_tokens,
        )
        turn_record["prompt_tokens"] = prompt_tokens
        turn_record["max_tokens"] = max_tokens

        if max_tokens == 0:
            log.warning(
                "[%s] Context full on turn %d (prompt=%d tokens), stopping",
                tag, turn_idx + 1, prompt_tokens,
            )
            turn_record["status"] = "context_full"
            trace["turns"].append(turn_record)
            break

        try:
            response = await chat_with_retry(
                client,
                tag=tag,
                hard_timeout=STREAM_HARD_TIMEOUT_S,
                model=model_name,
                messages=messages,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        except Exception as e:
            log.error(
                "[%s] Model error turn %d: %s", tag, turn_idx + 1, e,
            )
            turn_record["status"] = "model_error"
            turn_record["error"] = str(e)
            trace["turns"].append(turn_record)
            break

        msg = response.choices[0].message
        content = msg.content or ""
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
                tool_response_text = await _handle_lean_compile_node(
                    code, lean_server_url, lean_timeout,
                    node, rebuild_dependency_block,
                    tag, turn_idx, turn_record, trace,
                    negation_name=negation_name,
                    negation_signature=negation_signature,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        f"<tool_response>\n{tool_response_text}\n"
                        f"</tool_response>"
                    ),
                })
                turn_record["tool_response"] = tool_response_text
                turn_record["tool_calls"] = [{
                    "name": "lean_compile",
                    "arguments": {"code": code},
                    "response": tool_response_text,
                }]
                trace["turns"].append(turn_record)
                if turn_record.get("status") == "success":
                    break
                continue

            # End the rollout cleanly; don't insert a recovery message.
            # Empirically such a nudge fires often (~42% of stuck
            # rollouts) but rescues almost none (~2%).
            log.warning(
                "[%s] No tool call on turn %d — ending rollout",
                tag, turn_idx + 1,
            )
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
                        "content": (
                            "mathlib_search is not enabled on this run."
                        ),
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
                        ceiling=60 * LEAN_TOOL_CEILING_MULTIPLIER,
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
                tool_response_text = await _handle_lean_compile_node(
                    code, lean_server_url, lean_timeout,
                    node, rebuild_dependency_block,
                    tag, turn_idx, turn_record, trace,
                    negation_name=negation_name,
                    negation_signature=negation_signature,
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
            turn_record["status"] = "search_only"
        trace["turns"].append(turn_record)
        if solved_this_turn:
            break

    # --allow-forfeit tail handler. Fires only after the rollout has
    # actually failed — `context_full` (the model exhausted the context
    # window), `no_tool_call` (the model stopped emitting tool calls),
    # or `max_turn` (loop fell through max_turns iterations). Skipped
    # for `model_error` and any successful trace.
    # `_run_forfeit_turn` then backtracks to a turn boundary that
    # leaves `forfeit_budget` tokens of headroom for thinking + output.
    if allow_forfeit and not trace["success"]:
        last_status = (
            trace["turns"][-1].get("status") if trace["turns"] else None
        )
        fell_through = len(trace["turns"]) >= max_turns
        if last_status in ("context_full", "no_tool_call"):
            forfeit_trigger = last_status
        elif fell_through:
            forfeit_trigger = "max_turn"
        else:
            forfeit_trigger = None
        if forfeit_trigger is not None:
            await _run_forfeit_turn(
                client=client,
                model_name=model_name,
                messages=messages,
                tools=active_tools,
                extra_body=extra_body,
                tag=tag,
                trace=trace,
                trigger=forfeit_trigger,
                tokenizer=tokenizer,
                tools_overhead=tools_overhead,
                max_model_len=max_model_len,
            )

    # --allow-negation post-success reflection. Always on whenever the
    # parallel negation pathway registered a solve: rewinds to the
    # natural turn boundary (typically the just-emitted SUCCESS tool
    # response, so the prefix cache stays warm) and asks the model to
    # write a STATEMENT_WRONG diagnosis (Analysis + Suggested Fix). The
    # parsed text is grafted into the revised blueprint by the
    # refinement stage (`refine.py`) in place of the synthetic
    # disproof block.
    if allow_negation and trace.get("solved_via_negation"):
        await _run_disproof_reflection_turn(
            client=client,
            model_name=model_name,
            messages=messages,
            tools=active_tools,
            extra_body=extra_body,
            tag=tag,
            trace=trace,
            tokenizer=tokenizer,
            tools_overhead=tools_overhead,
            max_model_len=max_model_len,
        )

    trace["elapsed_seconds"] = round(time.time() - t0, 2)
    trace["total_turns"] = len(trace["turns"])

    status = "SOLVED" if trace["success"] else "FAIL"
    if trace.get("solved_via_negation"):
        status += " (via negation)"
    log.info(
        "[%s] %s — turns=%d elapsed=%.1fs cost=$%.4f",
        tag, status, trace["total_turns"], trace["elapsed_seconds"],
        trace["total_cost_usd"],
    )
    return trace


# ---------------------------------------------------------------------------
#  Per-problem parallel dispatch
# ---------------------------------------------------------------------------

async def run_problem(
    client: AsyncOpenAI,
    model_name: str,
    problem: dict,
    lean_server_url: str,
    search_server_url: str | None,
    semaphore: "CountingSemaphore",
    tokenizer,
    tools_overhead: int = 0,
    max_turns: int = 64,
    temperature: float = 1.0,
    lean_timeout: int = 300,
    sample_idx: int = 0,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_output_tokens: int | None = None,
    node_retries: int = 1,
    main_retries: int | None = None,
    prune_dead: bool = True,
    extra_body: dict | None = None,
    skip_proof_sketch: bool = False,
    allow_negation: bool = False,
    allow_forfeit: bool = False,
) -> dict:
    """Prove every target node of a skeleton blueprint in parallel."""
    if main_retries is None:
        main_retries = node_retries

    problem_id = problem["problem_id"]
    formal_statement = problem["formal_statement"]
    skeleton_code = problem["skeleton_code"]
    upstream_skeleton = problem["upstream_skeleton"]
    tag_base = f"{problem_id}/s{sample_idx}"

    log.info("[%s] Start blueprint proving", tag_base)

    header, nodes = parse_blueprint_file(skeleton_code)
    # Drop `import Architect` from the header before it reaches the
    # model or the rebuild block. Architect is only required to parse
    # the skeleton's `@[blueprint]` macro and `sorry_using` body, neither
    # of which appears in the dep block we feed downstream (defs use
    # `decl_source` without the attr; parents are rebuilt with plain
    # `:= by sorry`). Leaving the import in seeds an `Architect` token
    # that the model has been observed to chase via `IO.FS.writeFile`
    # recovery loops, corrupting the on-disk Lean toolchain.
    header = "\n".join(
        line for line in header.split("\n")
        if line.strip() != "import Architect"
    )
    if not nodes:
        log.error("[%s] No blueprint nodes parsed", tag_base)
        return {
            "problem_id": problem_id,
            "sample": sample_idx,
            "success": False,
            "error": "no blueprint nodes parsed from skeleton",
            "total_turns": 0,
            "elapsed_seconds": 0,
            "total_model_tokens": {"input": 0, "output": 0},
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "total_lean_time": 0,
            "total_search_calls": 0,
            "node_results": {},
            "solved_nodes_derived": [],
            "unsolved_nodes_derived": [],
            "upstream_skeleton": upstream_skeleton,
        }

    target_nodes = [n for n in nodes if n.is_target]
    def_nodes = [n for n in nodes if not n.is_target]
    main_theorem_name = get_theorem_name(formal_statement)
    log.info(
        "[%s] Parsed skeleton: %d total nodes (%d defs, %d targets); main=%s",
        tag_base, len(nodes), len(def_nodes), len(target_nodes),
        main_theorem_name,
    )

    skeleton_header = header

    node_dep_blocks: dict[str, tuple[str, str]] = {}
    for node in target_nodes:
        prompt_block, rebuild_block = build_dependency_block(
            node, nodes, skeleton_header,
        )
        node_dep_blocks[node.name] = (prompt_block, rebuild_block)

    t0 = time.time()
    total_tokens = {"input": 0, "output": 0}
    total_cached_tokens = 0
    total_cost_usd = 0.0
    total_lean_time = 0.0
    total_search_calls = 0

    async def prove_node(node: BlueprintNode) -> dict:
        # Per-node checkpoint resume: short-circuit if we already
        # finalized this node in a previous (potentially killed) run.
        cache_key = (problem_id, sample_idx, node.name)
        cached = _node_results_cache.get(cache_key)
        if cached is not None:
            log.info(
                "[%s/%s] resuming cached node result "
                "(success=%s, attempts=%d)",
                tag_base, node.name,
                cached.get("success"),
                cached.get("attempts", 0),
            )
            return cached

        prompt_block, rebuild_block = node_dep_blocks[node.name]
        retries = (
            main_retries if node.name == main_theorem_name
            else node_retries
        )

        def tag_for(i: int) -> str:
            if retries <= 1:
                return f"{tag_base}/{node.name}"
            return f"{tag_base}/{node.name}/r{i}"

        async def run_attempt(i: int) -> dict:
            return await run_node(
                client=client,
                model_name=model_name,
                node=node,
                prompt_dependency_block=prompt_block,
                rebuild_dependency_block=rebuild_block,
                lean_server_url=lean_server_url,
                search_server_url=search_server_url,
                semaphore=semaphore,
                tokenizer=tokenizer,
                tools_overhead=tools_overhead,
                max_turns=max_turns,
                temperature=temperature,
                lean_timeout=lean_timeout,
                tag=tag_for(i),
                max_model_len=max_model_len,
                max_output_tokens=max_output_tokens,
                extra_body=extra_body,
                skip_proof_sketch=skip_proof_sketch,
                allow_negation=allow_negation,
                allow_forfeit=allow_forfeit,
            )

        # Sequential retries: r0, then r1 if r0 didn't solve, ...
        # Stops at the first success. No speculation, no cancel churn —
        # one streaming chat per node at a time.
        completed_results: list[dict] = []
        winner: dict | None = None
        winner_idx: int | None = None

        for i in range(retries):
            try:
                r = await run_attempt(i)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                r = {
                    "node_name": node.name,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            completed_results.append(r)
            if r.get("success"):
                winner = r
                winner_idx = i
                log.info(
                    "[%s] Node solved on attempt r%d (%d/%d attempts)",
                    tag_for(i), i, i + 1, retries,
                )
                break
            log.info(
                "[%s] Node failed attempt r%d", tag_for(i), i,
            )

        if winner is None:
            if completed_results:
                winner_idx, winner = max(
                    enumerate(completed_results),
                    key=lambda ir: ir[1].get("total_turns", 0),
                )
            else:
                winner_idx = None
                winner = {
                    "node_name": node.name,
                    "success": False,
                    "error": "no attempts completed",
                }

        winner["attempts"] = len(completed_results)
        if winner_idx is not None:
            winner["winner_attempt"] = winner_idx

        prior = [
            r for j, r in enumerate(completed_results) if j != winner_idx
        ]
        if prior:
            winner["prior_attempts"] = prior

        # Checkpoint this node's final result before returning so a
        # killed run can resume node-by-node.
        await _save_node_result(
            problem_id, sample_idx, node.name, winner,
        )

        return winner

    gathered = await asyncio.gather(
        *(prove_node(n) for n in target_nodes),
        return_exceptions=True,
    )

    node_results: dict[str, dict] = {}
    for node, result in zip(target_nodes, gathered):
        if isinstance(result, Exception):
            log.error(
                "[%s/%s] Exception: %s", tag_base, node.name, result,
            )
            node_results[node.name] = {
                "node_name": node.name,
                "success": False,
                "error": str(result),
            }
        else:
            node_results[node.name] = result
            total_tokens["input"] += result.get(
                "total_model_tokens", {},
            ).get("input", 0)
            total_tokens["output"] += result.get(
                "total_model_tokens", {},
            ).get("output", 0)
            total_cached_tokens += result.get("total_cached_tokens", 0)
            total_cost_usd += result.get("total_cost_usd", 0.0)
            total_lean_time += result.get("total_lean_time", 0)
            total_search_calls += result.get("total_search_calls", 0)

    all_nodes_succeeded = all(
        node_results.get(n.name, {}).get("success", False)
        for n in target_nodes
    )

    final_code = assemble_final_file(
        skeleton_code, target_nodes, node_results,
    )
    final_check_passed = False
    final_check_reason = ""

    async def _run_assembly_check(
        code: str, label: str,
    ) -> tuple[bool, str, float]:
        check_code = code
        if main_theorem_name:
            check_code = (
                code.rstrip()
                + f"\n\n#print axioms {main_theorem_name}\n"
            )
        log.info(
            "[%s] Running %s assembly check (%d nodes, %d chars)",
            tag_base, label, len(target_nodes), len(check_code),
        )
        try:
            result = await check_lean_async(
                check_code, lean_server_url, lean_timeout,
            )
            analysis = analyse_result(result)
            t_lean = analysis["time"]
            if not analysis["pass_"]:
                reason = (
                    f"final assembly compile failed with "
                    f"{len(analysis['errors'])} error(s)"
                )
                log.warning(
                    "[%s] %s assembly FAILED — %d error(s), %.1fs",
                    tag_base, label, len(analysis["errors"]), t_lean,
                )
                for i, err in enumerate(analysis["errors"][:10]):
                    pos = err.get("pos", {})
                    line = pos.get("line", "?")
                    msg = err.get("data", "")[:300]
                    log.warning(
                        "[%s] %s assembly error %d: L%s: %s",
                        tag_base, label, i + 1, line, msg,
                    )
                return False, reason, t_lean
            main_has_sorry = False
            if main_theorem_name:
                main_has_sorry = target_has_sorry(
                    check_code, main_theorem_name, analysis["sorries"],
                )
            if main_has_sorry:
                log.warning(
                    "[%s] %s assembly: main has sorry", tag_base, label,
                )
                return (
                    False,
                    "main theorem still contains sorry after assembly",
                    t_lean,
                )
            axioms = parse_axioms_from_result(result)
            classification, extra = classify_axioms(axioms)
            # PARTIAL = uses a strict subset of the foundational axioms
            # (propext / Classical.choice / Quot.sound). That's a legitimate
            # proof — reject only the buckets that actually signal unsoundness.
            if classification not in ("USES_SORRY", "CUSTOM_AXIOM"):
                log.info(
                    "[%s] %s assembly PASSED — axioms=%s (%.1fs)",
                    tag_base, label, classification, t_lean,
                )
                return True, "", t_lean
            log.warning(
                "[%s] %s assembly: axioms=%s extra=%s",
                tag_base, label, classification, extra,
            )
            return (
                False,
                f"non-legitimate axioms: {classification} "
                f"{sorted(extra) if extra else ''}",
                t_lean,
            )
        except Exception as e:
            log.error(
                "[%s] %s assembly exception: %s", tag_base, label, e,
            )
            return False, f"final assembly exception: {e}", 0.0

    if all_nodes_succeeded:
        final_check_passed, final_check_reason, _t = (
            await _run_assembly_check(final_code, "final")
        )
        total_lean_time += _t

    elapsed = round(time.time() - t0, 2)

    theorem_nodes = [n for n in nodes if n.kind in graph._TARGET_KINDS]
    solved_nodes_derived: list[str] = []
    unsolved_nodes_derived: list[str] = []
    for n in theorem_nodes:
        if not n.is_target:
            solved_nodes_derived.append(n.name)
            if n.name not in node_results:
                _, pb = graft_proof_body(n.decl_source, n)
                if pb and pb.strip() != "sorry" and \
                        not pb.strip().startswith("sorry_using"):
                    node_results[n.name] = {
                        "node_name": n.name,
                        "success": True,
                        "proof_body": pb,
                        "source": "carry_forward",
                    }
            continue
        r = node_results.get(n.name)
        if isinstance(r, dict) and r.get("success", False):
            solved_nodes_derived.append(n.name)
        else:
            unsolved_nodes_derived.append(n.name)

    # --allow-negation bookkeeping. A problem counts as solved-via-negation
    # at the problem level only when the MAIN theorem node was solved via
    # the negation pathway. Sub-lemma negations are still recorded on the
    # individual node trace but don't flip the problem-level flag, since
    # a non-main lemma's negation doesn't help the assembly compile.
    nodes_solved_via_negation = sorted(
        n for n, r in node_results.items()
        if isinstance(r, dict) and r.get("solved_via_negation")
    )
    main_node_result = node_results.get(main_theorem_name) if main_theorem_name else None
    problem_solved_via_negation = bool(
        isinstance(main_node_result, dict)
        and main_node_result.get("solved_via_negation")
    )

    # --allow-forfeit bookkeeping. Each per-node trace may carry a
    # `forfeit_declaration` dict; collect the names of nodes that
    # forfeited for telemetry, plus the parsed/raw payloads keyed by
    # node so downstream consumers can read failure analyses without
    # walking every node-result manually.
    node_forfeits: dict[str, dict] = {}
    for n_name, r in node_results.items():
        if not isinstance(r, dict):
            continue
        fd = r.get("forfeit_declaration")
        if fd is not None:
            node_forfeits[n_name] = fd

    # --allow-negation post-success reflections, mirroring node_forfeits.
    # When the reflection turn produced a STATEMENT_WRONG diagnosis, the
    # refinement stage grafts the parsed text into the revised blueprint
    # in place of the synthetic disproof block.
    node_disproofs: dict[str, dict] = {}
    for n_name, r in node_results.items():
        if not isinstance(r, dict):
            continue
        dr = r.get("disproof_reflection")
        if dr is not None:
            node_disproofs[n_name] = dr

    trace = {
        "problem_id": problem_id,
        "sample": sample_idx,
        "system_prompt": SYSTEM_PROMPT,
        "tools": TOOLS,
        "formal_statement": formal_statement,
        "skeleton_code": skeleton_code,
        "upstream_skeleton": upstream_skeleton,
        "nodes_total": len(nodes),
        "nodes_defs": len(def_nodes),
        "nodes_targets": len(target_nodes),
        "solved_nodes_derived": solved_nodes_derived,
        "unsolved_nodes_derived": unsolved_nodes_derived,
        "nodes_solved_via_negation": nodes_solved_via_negation,
        "problem_solved_via_negation": problem_solved_via_negation,
        "node_forfeits": node_forfeits,
        "node_disproofs": node_disproofs,
        "main_theorem_name": main_theorem_name,
        "node_results": node_results,
        "success": all_nodes_succeeded,
        "assembly_passed": final_check_passed,
        "final_check_reason": final_check_reason,
        "final_code": final_code,
        "total_model_tokens": total_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
        "total_lean_time": round(total_lean_time, 2),
        "total_search_calls": total_search_calls,
        "elapsed_seconds": elapsed,
        "total_turns": sum(
            r.get("total_turns", 0)
            for r in node_results.values()
            if isinstance(r, dict)
        ),
    }

    if prune_dead and final_code and main_theorem_name:
        try:
            # Trace-based reachability: avoids Lean's `simp [X]` blind
            # spot that the Lean-backed `prune_dead_nodes` hits via
            # `Architect.collectUsed`. See graph.analyze_blueprint_from_trace.
            (pruned_code, pruned_skel, pruned_solved,
             pruned_unsolved, ana) = prune_dead_nodes_from_trace(
                trace=trace,
                code=final_code,
                main_theorem_name=main_theorem_name,
                skeleton=upstream_skeleton,
                solved_nodes=solved_nodes_derived,
                unsolved_nodes=unsolved_nodes_derived,
            )
            if ana is not None and ana.dead:
                log.info(
                    "[%s] post-prove pruned %d dead nodes: %s "
                    "(reachable=%d unsolved=%d)",
                    tag_base, len(ana.dead), sorted(ana.dead),
                    len(ana.reachable), len(ana.unsolved_targets),
                )

                pruned_passed, pruned_reason, _t = (
                    await _run_assembly_check(pruned_code, "post-prune")
                )
                total_lean_time += _t

                # Revert-on-regress: if the pre-prune assembly was
                # passing and the pruned one isn't, throw the prune
                # away rather than let a dep-graph blind spot break
                # the assembly.
                regressed = final_check_passed and not pruned_passed
                if regressed:
                    log.warning(
                        "[%s] post-prune REGRESSED (pre=PASSED, "
                        "post=%s) — reverting to pre-prune assembly",
                        tag_base, pruned_reason,
                    )
                    trace["total_lean_time"] = round(total_lean_time, 2)
                else:
                    trace["final_code"] = pruned_code
                    if pruned_skel is not None:
                        trace["upstream_skeleton"] = pruned_skel
                    if pruned_solved is not None:
                        trace["solved_nodes_derived"] = pruned_solved
                    if pruned_unsolved is not None:
                        trace["unsolved_nodes_derived"] = pruned_unsolved
                    if pruned_passed and not final_check_passed:
                        log.info(
                            "[%s] assembly_passed flipped False -> True "
                            "after prune (dropped %d unreachable node(s))",
                            tag_base, len(ana.dead),
                        )
                    trace["assembly_passed"] = pruned_passed
                    trace["final_check_reason"] = pruned_reason
                    trace["total_lean_time"] = round(total_lean_time, 2)
                    final_check_passed = pruned_passed
                    final_check_reason = pruned_reason
        except Exception as e:
            log.warning("[%s] post-prove prune skipped: %s", tag_base, e)

    if final_check_passed:
        status = "SOLVED (assembly)"
    elif all_nodes_succeeded:
        status = f"NODES SOLVED (assembly: {final_check_reason})"
    else:
        status = "FAIL"
    nodes_solved = sum(
        1 for r in node_results.values()
        if isinstance(r, dict) and r.get("success")
    )
    log.info(
        "[%s] %s — %d/%d nodes solved, elapsed=%.1fs, searches=%d, "
        "cost=$%.4f",
        tag_base, status, nodes_solved, len(target_nodes),
        elapsed, total_search_calls, total_cost_usd,
    )
    return trace


# ---------------------------------------------------------------------------
#  Main async loop
# ---------------------------------------------------------------------------

async def async_main():
    parser = argparse.ArgumentParser(
        description=(
            "Theorem proving: replace the `sorry_using` bodies of each "
            "blueprint with real tactic proofs, one agentic conversation "
            "per lemma (parallel across lemmas, sequential retries per "
            "lemma)."
        ),
    )
    parser.add_argument(
        "--skeleton-results", required=True,
        help=(
            "Path to a blueprint-generation or refinement output directory "
            "containing traces.jsonl (and summary.json)."
        ),
    )
    parser.add_argument("--output", required=True, help="Output directory")
    add_model_args(parser, x_title="goedel-architect-prover")
    parser.add_argument(
        "--lean-server", default=DEFAULT_LEAN_SERVER,
        help="Lean server check endpoint (kimina-lean-server API) with "
             "Mathlib + LeanArchitect",
    )
    parser.add_argument(
        "--search-server", default=DEFAULT_SEARCH_SERVER,
        help="Mathlib search endpoint for the `mathlib_search` tool; "
             "'none' disables search",
    )
    parser.add_argument(
        "--max-turns", type=int, default=64,
        help="Max turns per node conversation",
    )
    parser.add_argument(
        "--num-samples", type=int, default=2,
        help="Independent blueprint-proof samples per problem",
    )
    parser.add_argument(
        "--concurrency", type=int, default=128,
        help="Max concurrent per-node conversations across all problems",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lean-timeout", type=int, default=300)
    parser.add_argument(
        "--tokenizer-path", default=None,
        help=(
            "Optional HF tokenizer path/name for prompt-token counting; "
            "falls back to char estimator"
        ),
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=(
            "Max context length the remote model accepts. DeepSeek-V4 "
            "supports up to 1M; default 256K is the agentic sweet spot."
        ),
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=None,
        help=(
            "Optional cap on output tokens per response. If unset, the "
            "per-turn max is whatever budget remains in --max-model-len "
            "after the prompt."
        ),
    )
    parser.add_argument(
        "--node-retries", type=int, default=1,
        help="Retry attempts per node before giving up (default 1)",
    )
    parser.add_argument(
        "--main-retries", type=int, default=None,
        help=(
            "Retry attempts for the main theorem node. Defaults to "
            "--node-retries."
        ),
    )
    parser.add_argument(
        "--problem-max-attempts", type=int, default=3,
        help=(
            "Retry a (problem, sample) this many times if run_problem "
            "raises an unexpected exception. Default: 3."
        ),
    )
    parser.add_argument(
        "--limit", type=int, help="Process only first N problems",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (problem, sample) already in traces.jsonl",
    )
    parser.add_argument(
        "--early-stop", action="store_true", default=True,
        help=(
            "Stop sampling a problem after first success (default: on)"
        ),
    )
    parser.add_argument(
        "--no-early-stop", dest="early_stop", action="store_false",
        help="Disable early-stop sampling",
    )
    parser.add_argument(
        "--input-jsonl", default=None,
        help=(
            "Optional: the original NL dataset JSONL so we can pick up "
            "the canonical formal_statement. If omitted, we use the "
            "formal_statement already stored in the skeleton trace."
        ),
    )
    parser.add_argument(
        "--prune-dead", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "After proving, prune nodes unreachable from the main "
            "theorem in the final partial assembly. ON by default; "
            "use --no-prune-dead to disable."
        ),
    )
    parser.add_argument(
        "--skip-proof-sketch", action="store_true",
        help=(
            "Drop the `-- Proof sketch:` block from the per-node user "
            "prompt. The `-- Statement:` block (when present) is kept "
            "so the model still sees an NL paraphrase of the claim. "
            "Useful when the blueprint's embedded sketch is unreliable, "
            "or when measuring how much the proof sketch helps the "
            "prover."
        ),
    )
    parser.add_argument(
        "--allow-negation", action="store_true",
        help=(
            "Permit the model to register a falsity proof of "
            "`{node_name}_negation` as an alternative to proving the "
            "original target node. Off by default; when on, the "
            "per-node user prompt names the parallel negation theorem "
            "and `lean_compile` accepts a complete proof of either as "
            "a node solve. The problem-level `solved_via_negation` "
            "flag flips True only when the MAIN theorem node solved "
            "via negation; sub-lemma negations stay confined to the "
            "per-node trace."
        ),
    )
    parser.add_argument(
        "--allow-forfeit", action="store_true",
        help=(
            "Inject a final structured forfeit declaration "
            "(FORFEIT_REASON / ANALYSIS / SUGGESTED_FIX) after a node "
            "rollout fails. Triggers on context_full, no_tool_call, or "
            "max_turn fall-through. The wrap-up backtracks to a turn "
            "boundary that leaves ~8k tokens of headroom for thinking "
            "+ output. Off by default — current behavior is unchanged."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    log_path = setup_logger(out_dir)
    log.info("Output directory: %s", out_dir)
    log.info("Session log: %s", log_path)
    log.info(
        "Timeouts: stream_hard_timeout=%.0fs lean_tool_ceiling=%.1fx "
        "timeout_budget=%d",
        STREAM_HARD_TIMEOUT_S, LEAN_TOOL_CEILING_MULTIPLIER, TIMEOUT_BUDGET,
    )
    log.info("Skeleton results: %s", args.skeleton_results)
    log.info("Model URL: %s", args.model_url)
    log.info("Model name: %s", args.model_name)
    log.info("Max model len: %d", args.max_model_len)
    log.info(
        "Max output tokens: %s",
        args.max_output_tokens if args.max_output_tokens is not None
        else "<unset — context-window remainder>",
    )
    log.info("Node retries: %d", args.node_retries)
    log.info(
        "Main retries: %s",
        args.main_retries if args.main_retries is not None
        else f"{args.node_retries} (same as node)",
    )
    log.info(
        "Num samples: %d (early_stop=%s)",
        args.num_samples, args.early_stop,
    )

    api_key = read_api_key(args)

    tokenizer = load_tokenizer(args.tokenizer_path)
    tools_overhead = estimate_tools_overhead(tokenizer, TOOLS)

    # Load skeletons
    skeleton_results_dir = Path(args.skeleton_results)
    skeleton_traces = load_skeleton_traces(skeleton_results_dir)
    if not skeleton_traces:
        log.error("No successful skeletons to process")
        sys.exit(1)

    dataset_stmts: dict[str, str] = {}
    if args.input_jsonl:
        with open(args.input_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r.get("problem_id") or r.get("uuid")
                if pid and r.get("formal_statement"):
                    dataset_stmts[pid] = r["formal_statement"]
        log.info(
            "Loaded %d formal statements from %s",
            len(dataset_stmts), args.input_jsonl,
        )

    problems: list[dict] = []
    for pid, t in skeleton_traces.items():
        skeleton_code = t.get("final_code")
        if not skeleton_code:
            continue
        upstream_skeleton = t.get("final_skeleton")
        if not upstream_skeleton:
            log.error(
                "[%s] upstream trace missing `final_skeleton` (blueprint "
                "generation and refinement always emit it). Skipping.",
                pid,
            )
            continue
        formal_statement = (
            dataset_stmts.get(pid) or t.get("formal_statement")
        )
        if not formal_statement:
            log.warning("No formal_statement for %s — skipping", pid)
            continue
        problems.append({
            "problem_id": pid,
            "formal_statement": formal_statement,
            "skeleton_code": skeleton_code,
            "upstream_skeleton": upstream_skeleton,
            "skeleton_sample": t.get("sample"),
        })
    problems.sort(key=lambda p: p["problem_id"])
    if args.limit:
        problems = problems[:args.limit]
    log.info(
        "Prepared %d problems for blueprint proving", len(problems),
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.model_url,
        default_headers=default_headers(args),
        timeout=httpx.Timeout(120.0, connect=15.0),
    )
    extra_body = build_extra_body(args)
    search_server_url = resolve_search_server(args.search_server)

    num_samples = args.num_samples
    semaphore = CountingSemaphore(args.concurrency)

    _effective_main_retries = (
        args.main_retries if args.main_retries is not None
        else args.node_retries
    )
    log.info(
        "Retry policy: sequential (concurrency=%d node_retries=%d "
        "main_retries=%d); speculation disabled",
        args.concurrency, args.node_retries, _effective_main_retries,
    )

    traces_path = out_dir / "traces.jsonl"
    write_lock = asyncio.Lock()

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
                done_keys.add(
                    (entry["problem_id"], entry.get("sample", 0)),
                )
                if entry.get("success"):
                    solved_pids.add(entry["problem_id"])
        log.info(
            "Resumed %d completed traces from %s",
            len(done_keys), traces_path,
        )
        if args.early_stop:
            log.info(
                "Early-stop resume: %d already solved",
                len(solved_pids),
            )

    if (not args.resume and traces_path.exists()
            and traces_path.stat().st_size > 0):
        log.error(
            "Output %s already contains results. Pass --resume to "
            "continue or choose a new --output dir.",
            traces_path,
        )
        sys.exit(1)
    if not args.resume:
        traces_path.write_text("")

    # Per-node checkpoint sidecar: lets a killed run resume
    # node-by-node within an in-progress problem instead of redoing
    # every node from scratch. Loaded into a module-level cache; the
    # `prove_node` short-circuit consults it on entry.
    global _node_results_path
    _node_results_path = out_dir / "node_results.jsonl"
    if args.resume:
        n_cached = _load_node_results(_node_results_path)
        log.info(
            "Resumed %d cached node results from %s",
            n_cached, _node_results_path,
        )
    else:
        _node_results_path.write_text("")

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
                    search_server_url=search_server_url,
                    semaphore=semaphore,
                    tokenizer=tokenizer,
                    tools_overhead=tools_overhead,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    lean_timeout=args.lean_timeout,
                    sample_idx=sample_idx,
                    max_model_len=args.max_model_len,
                    max_output_tokens=args.max_output_tokens,
                    node_retries=args.node_retries,
                    main_retries=args.main_retries,
                    prune_dead=args.prune_dead,
                    extra_body=extra_body,
                    skip_proof_sketch=args.skip_proof_sketch,
                    allow_negation=args.allow_negation,
                    allow_forfeit=args.allow_forfeit,
                )
                if attempt > 1:
                    log.info(
                        "[%s] succeeded on retry attempt %d/%d",
                        tag, attempt, max_attempts,
                    )
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                if attempt >= max_attempts:
                    log.error(
                        "[%s] gave up after %d attempts: %s: %s",
                        tag, max_attempts, type(e).__name__, e,
                    )
                    break
                backoff = min(2 ** (attempt - 1) * 5, 60)
                log.warning(
                    "[%s] attempt %d/%d failed (%s: %s); "
                    "retrying in %ds",
                    tag, attempt, max_attempts,
                    type(e).__name__, e, backoff,
                )
                await asyncio.sleep(backoff)

        if trace is None:
            trace = {
                "problem_id": pid,
                "sample": sample_idx,
                "success": False,
                "total_turns": 0,
                "elapsed_seconds": 0.0,
                "total_model_tokens": {"input": 0, "output": 0},
                "total_cached_tokens": 0,
                "total_cost_usd": 0.0,
                "total_lean_time": 0.0,
                "total_search_calls": 0,
                "node_results": {},
                "error": (
                    f"{type(last_exc).__name__}: {last_exc}"
                    if last_exc is not None else "unknown error"
                ),
                "attempts": max_attempts,
            }

        async with write_lock:
            with open(traces_path, "a") as f:
                f.write(
                    json.dumps(
                        trace, ensure_ascii=False, default=str,
                    ) + "\n"
                )
            completed += 1
            if completed % 10 == 0 or completed == total_convos:
                log.info(
                    "Progress: %d/%d blueprints saved",
                    completed, total_convos,
                )

        fc = trace.get("final_code")
        if fc:
            # A "proved" blueprint must be a real proof of the main
            # theorem. If any node was discharged via the negation
            # pathway (model proved the sub-lemma's negation, i.e. that
            # the lemma is false), the file is a disproof of a node, not
            # a proof of the theorem — bucket it as unproved so the
            # reviser still sees it as needing restructuring.
            is_real_proof = (
                trace.get("success")
                and not trace.get("nodes_solved_via_negation")
            )
            bucket = (
                "proved_blueprints"
                if is_real_proof
                else "unproved_blueprints"
            )
            out_lean = (
                out_dir / bucket
                / f"{pid}_s{sample_idx}.lean"
            )
            out_lean.parent.mkdir(parents=True, exist_ok=True)
            out_lean.write_text(fc, encoding="utf-8")

        return trace

    t_start = time.time()

    if args.early_stop:
        problems_to_run = [
            p for p in problems
            if p["problem_id"] not in solved_pids
        ]
        total_convos = len(problems_to_run) * num_samples
        log.info(
            "Early-stop: scheduling %d problems (up to %d samples each, "
            "%d already solved), concurrency=%d",
            len(problems_to_run), num_samples, len(solved_pids),
            args.concurrency,
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
                    log.info(
                        "[%s] Solved on sample %d (attempt %d/%d)",
                        pid, si, si + 1, num_samples,
                    )
                    break
            return traces_for_problem

        results_nested = await asyncio.gather(
            *(run_problem_early_stop(p) for p in problems_to_run),
            return_exceptions=True,
        )
        all_traces: list[dict] = list(existing_traces)
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
            "Scheduling %d traversals (%d problems x %d samples, "
            "%d resumed), concurrency=%d",
            total_convos, len(problems), num_samples, len(done_keys),
            args.concurrency,
        )
        results = await asyncio.gather(
            *(run_and_save(p, si) for p, si in tasks_to_run),
            return_exceptions=True,
        )
        for (problem, sample_idx), result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                pid = problem["problem_id"]
                log.error(
                    "[%s/s%d] Exception: %s", pid, sample_idx, result,
                )
                error_trace = {
                    "problem_id": pid,
                    "sample": sample_idx,
                    "success": False,
                    "error": str(result),
                    "total_turns": 0,
                    "elapsed_seconds": 0,
                    "total_model_tokens": {"input": 0, "output": 0},
                    "total_cached_tokens": 0,
                    "total_cost_usd": 0.0,
                    "total_lean_time": 0,
                    "total_search_calls": 0,
                    "node_results": {},
                }
                with open(traces_path, "a") as f:
                    f.write(
                        json.dumps(
                            error_trace, ensure_ascii=False,
                            default=str,
                        ) + "\n"
                    )
        all_traces = list(existing_traces)
        for result in results:
            if not isinstance(result, Exception):
                all_traces.append(result)

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
            nr = t.get("node_results", {})
            nodes_solved = sum(
                1 for r in nr.values()
                if isinstance(r, dict) and r.get("success")
            )
            nodes_total = len(nr)
            # Per-trace forfeit summary: how many nodes forfeited and a
            # compact {node: {trigger, parsed, error}} index. Matches
            # the agentic prover's per-sample `forfeit` field shape.
            node_forfeits = t.get("node_forfeits") or {}
            forfeit_field = None
            if node_forfeits:
                forfeit_field = {
                    "triggered": True,
                    "count": len(node_forfeits),
                    "nodes": {
                        n_name: {
                            "trigger": fd.get("trigger"),
                            "parsed": fd.get("parsed"),
                            "error": fd.get("error"),
                        }
                        for n_name, fd in node_forfeits.items()
                    },
                }
            # Per-trace disproof-reflection summary, same shape as
            # forfeit_field above. None when --allow-negation didn't
            # register any negation solves on this sample.
            node_disproofs = t.get("node_disproofs") or {}
            disproof_reflection_field = None
            if node_disproofs:
                disproof_reflection_field = {
                    "triggered": True,
                    "count": len(node_disproofs),
                    "nodes": {
                        n_name: {
                            "trigger": dr.get("trigger"),
                            "parsed": dr.get("parsed"),
                            "error": dr.get("error"),
                        }
                        for n_name, dr in node_disproofs.items()
                    },
                }
            sums.append({
                "problem_id": pid,
                "sample": t.get("sample", 0),
                "success": t.get("success", False),
                "assembly_passed": t.get("assembly_passed", False),
                "final_check_reason": t.get("final_check_reason", ""),
                "problem_solved_via_negation": t.get(
                    "problem_solved_via_negation", False,
                ),
                "nodes_solved_via_negation": t.get(
                    "nodes_solved_via_negation", [],
                ),
                "nodes_solved": nodes_solved,
                "nodes_total": nodes_total,
                "turns": t.get("total_turns", 0),
                "elapsed": t.get("elapsed_seconds", 0),
                "tokens_in": t.get(
                    "total_model_tokens", {},
                ).get("input", 0),
                "tokens_out": t.get(
                    "total_model_tokens", {},
                ).get("output", 0),
                "cached_tokens": t.get("total_cached_tokens", 0),
                "cost_usd": t.get("total_cost_usd", 0.0),
                "lean_time": t.get("total_lean_time", 0),
                "search_calls": t.get("total_search_calls", 0),
                "forfeit": forfeit_field,
                "disproof_reflection": disproof_reflection_field,
            })
        all_summaries[pid] = sums
        problem_success_counts[pid] = sum(
            1 for s in sums if s["success"]
        )

    total_conversations = sum(len(s) for s in all_summaries.values())
    total_search_calls = sum(
        t.get("total_search_calls", 0) for t in all_traces
    )
    total_input_tokens = sum(
        t.get("total_model_tokens", {}).get("input", 0)
        for t in all_traces
    )
    total_output_tokens = sum(
        t.get("total_model_tokens", {}).get("output", 0)
        for t in all_traces
    )
    total_cached_tokens = sum(
        t.get("total_cached_tokens", 0) for t in all_traces
    )
    total_cost_usd = sum(t.get("total_cost_usd", 0.0) for t in all_traces)
    cache_hit_rate = (
        total_cached_tokens / total_input_tokens
        if total_input_tokens else 0.0
    )

    pass_at_k_results = {}
    for k in sorted(
        {1, 2, 4, num_samples} & set(range(1, num_samples + 1))
    ):
        eligible = [
            (pid, len(all_summaries[pid]), problem_success_counts[pid])
            for pid in per_problem_traces
            if len(all_summaries[pid]) >= k
        ]
        if not eligible:
            continue
        total = sum(
            pass_at_k(n_actual, c, k)
            for _, n_actual, c in eligible
        )
        rate = total / n_problems if n_problems else 0
        pass_at_k_results[k] = {
            "rate": round(rate, 6),
            "solved_estimate": round(rate * n_problems, 1),
            "total": n_problems,
            "eligible": len(eligible),
        }

    solved_any = sum(
        1 for c in problem_success_counts.values() if c > 0
    )

    # --allow-negation aggregates. A problem counts as
    # solved-via-negation only when ANY of its samples solved the MAIN
    # theorem node via the negation pathway. Sub-lemma negations
    # populate `nodes_solved_via_negation_total` for telemetry.
    solved_negation = sum(
        1 for traces in per_problem_traces.values()
        if any(t.get("problem_solved_via_negation") for t in traces)
    )
    solved_positive = solved_any - solved_negation
    nodes_solved_via_negation_total = sum(
        len(t.get("nodes_solved_via_negation") or [])
        for t in all_traces
    )

    # --allow-forfeit aggregates. Each per-node `forfeit_declaration`
    # contributes one call to the totals; per-trigger and per-status
    # breakdowns mirror the agentic prover. All zero on runs without
    # the flag.
    forfeit_declarations: list[dict] = []
    for t in all_traces:
        for fd in (t.get("node_forfeits") or {}).values():
            forfeit_declarations.append(fd)
    total_forfeit_calls = len(forfeit_declarations)
    total_forfeit_tokens = {
        "input": sum(
            (fd.get("tokens") or {}).get("input", 0)
            for fd in forfeit_declarations
        ),
        "output": sum(
            (fd.get("tokens") or {}).get("output", 0)
            for fd in forfeit_declarations
        ),
    }
    forfeit_triggers: dict[str, int] = {}
    for fd in forfeit_declarations:
        key = fd.get("trigger") or "unknown"
        forfeit_triggers[key] = forfeit_triggers.get(key, 0) + 1
    forfeit_parse_ok = sum(
        1 for fd in forfeit_declarations if fd.get("parsed") is not None
    )
    forfeit_errors = sum(
        1 for fd in forfeit_declarations if fd.get("error") is not None
    )

    # --allow-negation post-success reflection aggregates. One entry per
    # node whose `<node>_negation` was Lean-checked and whose follow-up
    # reflection turn ran. Mirrors the forfeit aggregates so the two
    # wrap-up pathways are reported uniformly.
    disproof_reflections: list[dict] = []
    for t in all_traces:
        for dr in (t.get("node_disproofs") or {}).values():
            disproof_reflections.append(dr)
    total_disproof_reflections = len(disproof_reflections)
    total_disproof_reflection_tokens = {
        "input": sum(
            (dr.get("tokens") or {}).get("input", 0)
            for dr in disproof_reflections
        ),
        "output": sum(
            (dr.get("tokens") or {}).get("output", 0)
            for dr in disproof_reflections
        ),
    }
    disproof_reflection_parse_ok = sum(
        1 for dr in disproof_reflections if dr.get("parsed") is not None
    )
    disproof_reflection_errors = sum(
        1 for dr in disproof_reflections if dr.get("error") is not None
    )

    problem_assembly_counts: dict[str, int] = {}
    for pid, sums in all_summaries.items():
        problem_assembly_counts[pid] = sum(
            1 for s in sums if s["assembly_passed"]
        )
    assembled_any = sum(
        1 for c in problem_assembly_counts.values() if c > 0
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "stage": "blueprint_prover",
        "skeleton_results": str(skeleton_results_dir),
        "input_jsonl": args.input_jsonl,
        "model_url": args.model_url,
        "model_name": args.model_name,
        "lean_server": args.lean_server,
        "search_server": search_server_url,
        "max_turns": args.max_turns,
        "num_samples": num_samples,
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "tool_calling": (
            "openrouter (streaming) + unified_safeguard "
            "+ mathlib_search + parallel_nodes"
        ),
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "provider": args.provider,
        "node_retries": args.node_retries,
        "main_retries": (
            args.main_retries if args.main_retries is not None
            else args.node_retries
        ),
        "retry_policy": "sequential",
        "early_stop": args.early_stop,
        "skip_proof_sketch": args.skip_proof_sketch,
        "allow_negation": bool(args.allow_negation),
        "allow_forfeit": bool(args.allow_forfeit),
        "total_problems": n_problems,
        "total_traversals": total_conversations,
        "total_search_calls": total_search_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_tokens": total_cached_tokens,
        "cache_hit_rate": round(cache_hit_rate, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "solved_any_sample": solved_any,
        "solved_positive": solved_positive,
        "solved_negation": solved_negation,
        "nodes_solved_via_negation_total": nodes_solved_via_negation_total,
        "total_forfeit_calls": total_forfeit_calls,
        "total_forfeit_tokens": total_forfeit_tokens,
        "forfeit_triggers": forfeit_triggers,
        "forfeit_parse_ok": forfeit_parse_ok,
        "forfeit_errors": forfeit_errors,
        "total_disproof_reflections": total_disproof_reflections,
        "total_disproof_reflection_tokens": total_disproof_reflection_tokens,
        "disproof_reflection_parse_ok": disproof_reflection_parse_ok,
        "disproof_reflection_errors": disproof_reflection_errors,
        "solve_rate_any": (
            solved_any / n_problems if n_problems else 0
        ),
        "assembled_any_sample": assembled_any,
        "assembly_rate_any": (
            assembled_any / n_problems if n_problems else 0
        ),
        "pass@k": pass_at_k_results,
        "elapsed_seconds": round(elapsed_total, 2),
        "per_problem_results": all_summaries,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("=" * 60)
    log.info(
        "DONE: %d/%d nodes solved, %d/%d assembled (%.1f%%/%.1f%%) "
        "— %d samples, concurrency=%d",
        solved_any, n_problems,
        assembled_any, n_problems,
        100 * solved_any / n_problems if n_problems else 0,
        100 * assembled_any / n_problems if n_problems else 0,
        num_samples, args.concurrency,
    )
    for k, info in pass_at_k_results.items():
        log.info(
            "  pass@%d = %.1f%% (~%.0f/%d)",
            k, info["rate"] * 100,
            info["solved_estimate"], n_problems,
        )
    log.info("Total mathlib_search calls: %d", total_search_calls)
    log.info(
        "Tokens: input=%d (cached=%d, %.1f%% hit) output=%d",
        total_input_tokens, total_cached_tokens,
        100 * cache_hit_rate, total_output_tokens,
    )
    log.info("Total cost (OpenRouter billed): $%.4f", total_cost_usd)
    if args.allow_negation:
        log.info(
            "Negation pathway: solved_negation=%d (positive=%d) — "
            "%d sub-lemma negation solves across all samples",
            solved_negation, solved_positive,
            nodes_solved_via_negation_total,
        )
        log.info(
            "Disproof reflections: %d declarations "
            "(parse_ok=%d, errors=%d, tokens in=%d/out=%d)",
            total_disproof_reflections,
            disproof_reflection_parse_ok, disproof_reflection_errors,
            total_disproof_reflection_tokens["input"],
            total_disproof_reflection_tokens["output"],
        )
    if args.allow_forfeit:
        log.info(
            "Forfeit pathway: %d declarations across %d nodes "
            "(triggers=%s, parse_ok=%d, errors=%d, tokens in=%d/out=%d)",
            total_forfeit_calls, total_forfeit_calls,
            forfeit_triggers, forfeit_parse_ok, forfeit_errors,
            total_forfeit_tokens["input"],
            total_forfeit_tokens["output"],
        )
    log.info("Total wall time: %.1fs", elapsed_total)
    log.info("Results: %s", summary_path)


def main():
    # Line-buffer output so progress shows up promptly when piped.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
