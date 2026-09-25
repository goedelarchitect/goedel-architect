"""Blueprint refinement (paper §3.3).

After a proving pass leaves lemmas unproved, the refinement model rewrites
the dependency graph around the failures. Its input is the blueprint
annotated with the prover's per-lemma verdicts:

  * `-- PROVED` after each lemma the prover closed;
  * `-- UNPROVED` after each lemma it failed on (or formally disproved via
    the negation path), followed by one `/- ... -/` block with the prover's
    structured diagnosis (`## Diagnosis` STATEMENT_WRONG | PROOF_TOO_HARD,
    `## Analysis`, `## Suggested Fix`).

The model emits a revised blueprint — every lemma body again
`:= by sorry_using [deps]` — typically decomposing a hard lemma into
helpers, rewiring dependencies, or repairing / dropping a false statement.
Its `lean_compile` tool enforces the same safeguards and graph validity as
blueprint generation, forbids real proofs ("skeleton-out"), and requires
every sliced per-node context to compile. On success, proof bodies from
the previous pass that still fit the revised statements are carried forward
(each is test-grafted in isolation), so solved lemmas are not re-proved.

Input:  the prover's `traces.jsonl` (`--prover-traces`).
Output: `traces.jsonl` (revised blueprints: `final_skeleton` is the pure
        `sorry_using` graph, `final_code` also carries the reused proofs),
        `summary.json`, `session.log` under `--output`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from . import blueprint
from . import graph
from . import prompts
from .graph import pass_at_k, prune_dead_nodes, extract_lean_code_from_text
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

# Pre-compile safeguards, violation rendering, and the graph-validity check
# are shared with blueprint generation so feedback reads identically.
from .blueprint import (
    _render_violation_blocks,
    _render_validation_blocks,
    _violation_synthetic,
    check_unbalanced_block_comment,
    check_missing_theorem,
    check_forbidden_constructs,
    check_imports,
    check_main_signature_matches,
    check_no_bare_sorry,
    check_all_theorems_annotated,
    check_blueprint_graph_validity,
    graft_main_signature,
)

# Fixed name (not __name__): when run with `python -m`, __name__ is "__main__".
log = logging.getLogger("goedel_architect.refine")


# ---------------------------------------------------------------------------
#  Tool definitions
# ---------------------------------------------------------------------------

LEAN_COMPILE_TOOL = {
    "type": "function",
    "function": {
        "name": "lean_compile",
        "description": (
            "Compile a Lean 4 LeanArchitect blueprint skeleton against "
            "Mathlib + LeanArchitect and return compilation feedback. "
            "Pre-compile safeguards (forbidden constructs, missing "
            "imports, signature mismatch, bare sorry, unannotated "
            "theorems) reject malformed input before it reaches the "
            "Lean gateway. The skeleton-out invariant is also enforced: "
            "every theorem/lemma body must be `:= by sorry_using "
            "[deps]` — the reviser does NOT fill in real tactics. After "
            "a clean compile, a graph-validity check runs: every node "
            "must carry a non-empty `(statement := /-- ... -/)` field, "
            "every Lemma/Theorem must carry a non-empty `(proof := "
            "/-- ... -/)` field, the `sorry_using` graph must be acyclic "
            "with no self-loops and only reference declared names, "
            "exactly one main Theorem must exist under the canonical "
            "name, and every node must be reachable from the main "
            "Theorem. On success the response also reports a per-"
            "declaration proof-reuse check — which prior proof bodies "
            "from the previous prover pass still fit the revised parent "
            "statements — so you know which nodes carry forward and "
            "which will need fresh work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Lean 4 LeanArchitect blueprint skeleton. "
                        "Must include `import Mathlib` and `import "
                        "Architect`. Every theorem/lemma must be "
                        "preceded by `@[blueprint (statement := /-- ... "
                        "-/) (proof := /-- ... -/)]` and end in `:= by "
                        "sorry_using [d1, d2, ...]`. Definitions need "
                        "only `(statement := /-- ... -/)` and carry a "
                        "real Lean body. The main theorem must use the "
                        "exact original Lean signature provided in the "
                        "user prompt."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

TOOLS = [LEAN_COMPILE_TOOL, MATHLIB_SEARCH_TOOL]


# ---------------------------------------------------------------------------
#  System / user prompts (goedel_architect/prompts/blueprint_refinement.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = prompts.load("blueprint_refinement")

# Extra edit (beyond dropping the `mathlib_search` paragraph) applied when
# the run has no Mathlib search; see `prompts.without_search`.
_NO_SEARCH_EDITS = (
    (" If `lean_compile` reports `Unknown identifier` / `Unknown constant`, "
     "**always call `mathlib_search` first to discover the correct name** "
     "before re-emitting the skeleton.", ""),
)


def system_prompt_for(with_search: bool) -> str:
    """System prompt with or without the search guidance."""
    if with_search:
        return SYSTEM_PROMPT
    return prompts.without_search(SYSTEM_PROMPT, _NO_SEARCH_EDITS)


def build_user_prompt(annotated_skeleton: str, formal_statement: str) -> str:
    """Per-sample user prompt: the annotated input skeleton + the \
canonical Lean signature for reference. The system prompt holds all \
behavioral rules and is cached across samples."""
    _, body = blueprint.split_header_body(formal_statement)
    body = body.strip("\n")
    return (
        "Revise the following Lean 4 dependency graph. "
        "Address each `-- UNPROVED` node per the review block that "
        "follows it, then emit a revised dependency graph.\n\n"
        "## Original theorem\n"
        "Use this signature exactly for the main theorem. At "
        "`lean_compile`, your code is rebuilt with this canonical "
        "signature.\n\n"
        f"```lean4\n{body}\n```\n\n"
        "## Annotated input dependency graph\n"
        f"```lean4\n{annotated_skeleton}\n```\n"
    )


# ---------------------------------------------------------------------------
#  Skeleton annotation: PROVED / UNPROVED + Diagnosis block
# ---------------------------------------------------------------------------
#
#  Annotates each theorem/lemma in the input skeleton with the prover's
#  per-node verdict. UNPROVED nodes additionally carry a `/- Diagnosis
#  ... -/` comment block — the prover's `--allow-forfeit` declaration
#  when one exists, or a synthetic STATEMENT_WRONG block when the
#  parallel `--allow-negation` pathway succeeded for that node.
#  Definitions are left untouched.
#
#  Strip helpers tolerate any markers and `/- ... -/` block the model
#  preserved in its revised output; `final_code` / `final_skeleton` are
#  always scrubbed before being saved.
# ---------------------------------------------------------------------------

_ANNOTATED_REPORT_RE = re.compile(
    r"\n-- UNPROVED[^\n]*\n/-\s*\n"
    r"## Diagnosis"
    r".*?-/",
    re.DOTALL,
)
_ANNOTATED_MARKER_RE = re.compile(r"\n-- (?:PROVED|UNPROVED)[^\n]*")


def strip_annotations(code: str) -> str:
    """Remove `-- PROVED` / `-- UNPROVED` markers and any attached \
`/- Diagnosis ... -/` block that `annotate_skeleton` may have \
injected. The model tends to preserve them in its revised output; \
we don't want them in `final_code` or `final_skeleton`."""
    code = _ANNOTATED_REPORT_RE.sub("", code)
    code = _ANNOTATED_MARKER_RE.sub("", code)
    return code


def strip_annotations_in_tool_args(arguments_json: str) -> str:
    """Strip input annotations from the `code` field of a `lean_compile` \
tool-call's `arguments` JSON.

    The compile itself still runs on the original (un-stripped) code; \
this only rewrites what gets persisted into the chat history so the \
model doesn't burn input tokens on echoed `-- UNPROVED` markers and \
attached review blocks on every subsequent turn.

    Returns the JSON string unchanged if parsing fails or there is no \
`code` field.
    """
    try:
        d = json.loads(arguments_json)
    except Exception:
        return arguments_json
    if isinstance(d, dict) and isinstance(d.get("code"), str):
        stripped = strip_annotations(d["code"])
        if stripped == d["code"]:
            return arguments_json
        d["code"] = stripped
        return json.dumps(d, ensure_ascii=False)
    return arguments_json


# ---------------------------------------------------------------------------
#  Diagnosis-priority forfeit picker.
#
#  When a node fails every retry, each failed attempt may have produced
#  its own `forfeit_declaration` with a Diagnosis label
#  (STATEMENT_WRONG / PROOF_TOO_HARD). The blueprint prover serializes
#  only the WINNER attempt's forfeit (winner = max-turns failed
#  attempt), losing the others. Here we re-derive the per-node forfeit
#  from `node_results[n]` (winner + prior_attempts) using a tier order:
#
#      Tier 1 (formally disproved)  → handled separately via
#                                     `node_disproofs` and the
#                                     `_format_disproof_block` path;
#                                     never enters this function.
#      Tier 2 (STATEMENT_WRONG)     → the lemma is FALSE under its
#                                     hypotheses. Stronger claim about
#                                     wrongness — promote it.
#      Tier 3 (PROOF_TOO_HARD)      → the lemma is provable but the
#                                     prover couldn't chain to it.
#      Tier 4 (unparseable Diagnosis) → wrap-up errored or model
#                                       returned malformed text.
#
#  Within a tier, ties are broken by attempt `total_turns` — more turns
#  ≈ more thoroughly explored ≈ better-grounded analysis.
# ---------------------------------------------------------------------------

_DIAGNOSIS_RE = re.compile(
    r"##\s*Diagnosis[^\n]*\n\s*(STATEMENT_WRONG|PROOF_TOO_HARD)\b",
    re.IGNORECASE,
)


def _diagnosis_label(forfeit: dict) -> str | None:
    """Extract `STATEMENT_WRONG` / `PROOF_TOO_HARD` from a \
`forfeit_declaration`'s markdown. Returns `None` if no diagnosis line \
is present (wrap-up turn errored or model produced malformed text)."""
    parsed = forfeit.get("parsed") or {}
    text = (parsed.get("forfeit_reason") or "").strip()
    if not text:
        text = (forfeit.get("raw") or "").strip()
    if not text:
        return None
    m = _DIAGNOSIS_RE.search(text)
    return m.group(1).upper() if m else None


def _pick_forfeit_by_priority(node_result: dict) -> dict | None:
    """Pick the most-informative `forfeit_declaration` across every \
retry attempt for one node. Walks `node_result["forfeit_declaration"]` \
(the winner attempt) and every `prior_attempts[i]["forfeit_declaration"]`, \
buckets by Diagnosis label, then returns the dict from the highest- \
priority non-empty bucket — tie-broken by attempt `total_turns`. \
Returns `None` when no attempt produced a forfeit."""
    candidates: list[tuple[dict, int]] = []
    fd = node_result.get("forfeit_declaration")
    if fd is not None:
        candidates.append((fd, node_result.get("total_turns") or 0))
    for prior in node_result.get("prior_attempts") or []:
        if not isinstance(prior, dict):
            continue
        pfd = prior.get("forfeit_declaration")
        if pfd is not None:
            candidates.append((pfd, prior.get("total_turns") or 0))
    if not candidates:
        return None
    buckets: dict[str | None, list[tuple[dict, int]]] = {
        "STATEMENT_WRONG": [], "PROOF_TOO_HARD": [], None: [],
    }
    for fd, turns in candidates:
        buckets[_diagnosis_label(fd)].append((fd, turns))
    for label in ("STATEMENT_WRONG", "PROOF_TOO_HARD", None):
        if buckets[label]:
            return max(buckets[label], key=lambda ft: ft[1])[0]
    return None


def _format_forfeit_block(forfeit: dict) -> str:
    """Render a `forfeit_declaration` dict (from the prover's per-node \
trace) as the body of a `/- Diagnosis ... -/` \
comment. Prefer the parsed `forfeit_reason` text; fall back to \
`raw`. The text is the prover's own three-section markdown \
(`## Diagnosis: ...`, `## Analysis`, `## Suggested Fix`); a synthetic \
PROOF_TOO_HARD block is emitted when no usable text is recorded."""
    parsed = forfeit.get("parsed") or {}
    text = (parsed.get("forfeit_reason") or "").strip()
    if not text:
        text = (forfeit.get("raw") or "").strip()
    if not text:
        text = (
            "## Diagnosis: PROOF_TOO_HARD\n"
            "\n"
            "## Analysis\n"
            "No structured diagnosis was recorded: the rollout ended "
            "without producing a wrap-up.\n"
            "\n"
            "## Suggested Fix\n"
            "Without further information, default to a helper-lemma "
            "decomposition; if other context suggests the statement "
            "itself is suspect, treat as STATEMENT_WRONG and rewrite "
            "it instead."
        )
    return text


def _format_disproof_block(reflection: dict | None = None) -> str:
    """Render the body of a `/- Diagnosis ... -/` block for a node \
whose `--allow-negation` pathway succeeded.

    When the prover's post-success reflection turn captured a \
STATEMENT_WRONG diagnosis (always on with --allow-negation), the \
`reflection` dict carries that model-authored markdown under \
`parsed.forfeit_reason` (raw fallback under `raw`); we feed the text \
through verbatim, mirroring `_format_forfeit_block`. Falls back to a \
synthetic STATEMENT_WRONG block when no reflection was recorded \
(legacy traces, parse failure, or reflection-turn error)."""
    parsed = (reflection or {}).get("parsed") or {}
    text = (parsed.get("forfeit_reason") or "").strip()
    if not text and reflection is not None:
        text = (reflection.get("raw") or "").strip()
    if text:
        return text
    return (
        "## Diagnosis: STATEMENT_WRONG\n"
        "\n"
        "## Analysis\n"
        "The parallel negation pathway constructed a Lean-checked "
        "proof of the negation under this lemma's hypotheses. The "
        "lemma is therefore FALSE under those hypotheses.\n"
        "\n"
        "## Suggested Fix\n"
        "Rewrite the formal statement (strengthen hypotheses, weaken "
        "the conclusion, fix a quantifier or coercion, etc.) so it "
        "becomes provable, or drop the node entirely and re-route "
        "every node that depended on it. Do NOT keep the lemma as-is "
        "and rely on `sorry_using` to mask the falsity — the prover "
        "will not be able to close it on the next pass either."
    )


def annotate_skeleton(
    skeleton: str,
    proved: set[str],
    unproved: set[str],
    forfeits: dict[str, dict] | None = None,
    disproofs: dict[str, dict] | None = None,
) -> str:
    """Insert `-- PROVED` / `-- UNPROVED` markers after each \
theorem/lemma declaration. UNPROVED nodes additionally get a \
`/- Diagnosis ... -/` block: when the negation pathway succeeded the \
block carries the prover's post-success STATEMENT_WRONG reflection \
(falling back to synthetic boilerplate if no reflection was recorded); \
when the rollout merely forfeited, the block carries the prover's \
forfeit declaration. Plain UNPROVED markers without an attached block \
are reserved for nodes the prover failed silently on (no forfeit, \
no negation).

    `disproofs` maps `node_name → reflection dict` (same shape as a \
`forfeit_declaration` blob: `parsed.forfeit_reason`, `raw`, `trigger`, \
`tokens`, ...). Membership in `disproofs` is what flags a formally- \
disproved node; the value supplies the model-authored body."""
    forfeits = forfeits or {}
    disproofs = disproofs or {}
    try:
        _, nodes = graph.parse_blueprint_file(skeleton)
    except Exception as e:
        log.warning("parse_blueprint_file failed during annotation: %s", e)
        return skeleton

    targets = [n for n in nodes if n.kind in {"theorem", "lemma"}]
    targets.sort(key=lambda n: n.decl_end, reverse=True)
    out = skeleton
    for n in targets:
        if n.name in proved:
            extra = ""
            block = "\n-- PROVED" + extra
        elif n.name in unproved:
            extra = ""
            if n.name in disproofs:
                body = _format_disproof_block(disproofs[n.name])
                extra = (
                    "\n/-\n"
                    f"{body}\n"
                    "-/"
                )
            elif n.name in forfeits:
                body = _format_forfeit_block(forfeits[n.name])
                extra = (
                    "\n/-\n"
                    f"{body}\n"
                    "-/"
                )
            block = "\n-- UNPROVED" + extra
        else:
            # Every theorem/lemma in the input skeleton should appear
            # in exactly one of `proved` / `unproved`. Falling through
            # here means the upstream derivation lost a node — surface
            # loudly rather than silently default.
            log.warning(
                "annotate_skeleton: node %s appears in the skeleton "
                "but is absent from both proved and unproved sets; "
                "skipping annotation so the model is not misled.",
                n.name,
            )
            continue
        insert_at = n.decl_end
        out = out[:insert_at] + block + out[insert_at:]
    return out


# ---------------------------------------------------------------------------
#  Stripping pre-blueprint doc comments (keeps the input skeleton clean
#  before annotation).
# ---------------------------------------------------------------------------

def strip_pre_blueprint_doc_comments(code: str) -> str:
    """Drop any `/-- ... -/` doc comment whose closing `-/` is followed \
only by whitespace before the next `@[blueprint ...]` attribute.

    Uses `graph.parse_blueprint_file` to locate each node's `attr_start` \
(the `@` of `@[blueprint`), then scans backward past whitespace and a \
single matching `/-- ... -/` block. Inner doc comments inside \
`(statement := /-- ... -/)` / `(proof := /-- ... -/)` are untouched \
because they sit AFTER `attr_start`, not before. Idempotent.
    """
    try:
        _, nodes = graph.parse_blueprint_file(code)
    except Exception as e:
        log.warning("strip_pre_blueprint_doc_comments: parse failed: %s", e)
        return code
    cuts: list[tuple[int, int]] = []
    for n in nodes:
        end = n.attr_start
        i = end
        while i > 0 and code[i - 1] in " \t\r\n":
            i -= 1
        if i >= 2 and code[i - 2:i] == "-/":
            j = code.rfind("/--", 0, i - 2)
            if j >= 0 and "-/" not in code[j + 3:i - 2]:
                cuts.append((j, end))
    for start, end in sorted(cuts, reverse=True):
        code = code[:start] + code[end:]
    return code


# ---------------------------------------------------------------------------
#  Skeleton-out check: every theorem/lemma body must be sorry_using.
# ---------------------------------------------------------------------------

def check_skeleton_out(code: str) -> list[dict] | None:
    """Reject any theorem/lemma whose body is NOT `:= by sorry_using \
[...]`. The reviser is forbidden from filling in real tactics — that's \
the next prover stage's job.

    Renders matched non-skeleton nodes as a single safeguard violation \
listing every offending name; this keeps the response concise even \
when the model makes the same mistake on every node.
    """
    try:
        _, nodes = graph.parse_blueprint_file(code)
    except Exception:
        # Unparseable code is caught by other gates (compile-fail);
        # don't double-report here.
        return None
    real_proofs = [
        n.name for n in nodes
        if n.kind in {"theorem", "lemma"} and not n.is_target
    ]
    if not real_proofs:
        return None
    name_list = ", ".join(f"`{n}`" for n in real_proofs)
    message = (
        "The reviser stage requires every theorem/lemma body to be "
        "exactly `:= by sorry_using [deps]` (no real tactic proofs — "
        "the next pipeline stage fills them in). The following "
        f"declarations have non-skeleton bodies and must be replaced "
        f"with `:= by sorry_using [...]`: {name_list}."
    )
    return [_violation_synthetic("", message)]


# ---------------------------------------------------------------------------
#  Per-node assembly check + partial assembly of fitting bodies.
#
#  On a clean skeleton compile we reuse prior proof bodies on a per-
#  node basis. For each theorem/lemma in the revised skeleton whose
#  name matches a successful node from the previous prover stage, we
#  build a minimal Lean file containing:
#      - the skeleton's imports / set_option / open header
#      - every definition from the revised skeleton
#      - this node's direct-parent lemmas as `sorry` stubs
#      - this node with its prior proof body grafted in
#  and compile it in isolation. If it compiles, the body "fits". We
#  then build the final partial assembly (using the full revised
#  skeleton) with ONLY the fitting bodies grafted; unfit candidates
#  and no-match nodes stay as `sorry_using`. Cascaded errors from one
#  broken graft do not taint the others.
#
#  Solved-via-negation nodes are excluded from candidacy: their
#  recorded proof body proves `<node>_negation`, not `<node>` itself,
#  so grafting it onto the original target would fail compilation.
# ---------------------------------------------------------------------------


def _build_per_node_check_file(
    node: "graph.BlueprintNode",
    proof_body: str,
    all_nodes: list,
    formal_header: str,
) -> str | None:
    """Minimal Lean file testing whether `proof_body` fits `node` in \
the revised context. Reuses `graph.build_dependency_block` and \
`graph.graft_proof_body` to mirror the per-node scope the blueprint \
prover applies.
    """
    _, rebuild_block = graph.build_dependency_block(node, all_nodes, formal_header)
    stub_model_code = (
        f"{node.kind} {node.name} := by\n{proof_body.rstrip()}\n"
    )
    grafted, _ = graph.graft_proof_body(stub_model_code, node)
    if grafted is None:
        return None
    return f"{rebuild_block.strip()}\n\n{grafted}\n"


def _extract_proof_body(node_result: dict) -> str | None:
    """Prefer an explicit `proof_body` field; fall back to re-extracting \
from `grafted` (the full declaration the prover wrote). Mirrors the \
fallback in `graph.assemble_final_file`."""
    pb = node_result.get("proof_body")
    if pb:
        return pb
    g = node_result.get("grafted")
    if g:
        m = re.search(r":=\s*by\b", g)
        if m:
            out = g[m.end():].rstrip()
            out = re.sub(r"\n*#print\s+axioms\s+\S+\s*$", "", out).rstrip()
            return out.lstrip("\n")
    return None


def _empty_assembly_result(safe_code: str) -> dict:
    return {
        "attempted": False,
        "assembled_code": safe_code,
        "n_fit": 0,
        "n_unfit": 0,
        "n_pending": 0,
        "fit_node_names": [],
        "unfit_node_names": [],
        "pending_node_names": [],
        "compile_pass": None,
        "compile_time": 0.0,
        "lean_result": None,
    }


async def _assemble_and_check(
    safe_code: str,
    node_results: dict,
    lean_server_url: str,
    lean_timeout: int,
) -> dict:
    """Per-node isolated test-graft → partial assembly of fitting bodies."""
    try:
        header, nodes = graph.parse_blueprint_file(safe_code)
    except Exception as e:
        log.warning("assembly: parse_blueprint_file failed: %s", e)
        return _empty_assembly_result(safe_code)

    target_nodes = [n for n in nodes if n.kind in {"theorem", "lemma"}]
    candidates: list = []
    candidate_bodies: dict[str, str] = {}
    pending: list[str] = []
    for n in target_nodes:
        r = node_results.get(n.name) or {}
        # Solved-via-negation nodes record a proof body for
        # `<node>_negation`, not for `<node>` itself. They cannot be
        # grafted onto the original target — the original is FALSE
        # under its hypotheses. Drop them straight into `pending`.
        if not r.get("success") or r.get("solved_via_negation"):
            pending.append(n.name)
            continue
        pb = _extract_proof_body(r)
        if not pb:
            pending.append(n.name)
            continue
        candidates.append(n)
        candidate_bodies[n.name] = pb

    if not candidates:
        out = _empty_assembly_result(safe_code)
        out["pending_node_names"] = pending
        out["n_pending"] = len(pending)
        return out

    async def _check_one(node) -> tuple:
        check_file = _build_per_node_check_file(
            node, candidate_bodies[node.name], nodes, header,
        )
        if check_file is None:
            return (node.name, False, 0.0, "graft_proof_body returned None")
        try:
            lr = await check_lean_async(check_file, lean_server_url, lean_timeout)
            ana = analyse_result(lr)
            if ana["pass_"]:
                return (node.name, True, ana["time"], None)
            first_err = ana["errors"][0] if ana["errors"] else {}
            return (
                node.name, False, ana["time"],
                str(first_err.get("data", ""))[:120],
            )
        except Exception as e:
            return (node.name, False, 0.0, f"lean check raised: {e}")

    verdicts = await asyncio.gather(*[_check_one(n) for n in candidates])
    fit_names: list[str] = []
    unfit_names: list[str] = []
    compile_time_total = 0.0
    for name, fits, t, err in verdicts:
        compile_time_total += t
        (fit_names if fits else unfit_names).append(name)
        if not fits:
            log.debug("assembly: node %s unfit: %s", name, err)

    fit_nodes = [n for n in candidates if n.name in set(fit_names)]
    assembled = graph.assemble_final_file(safe_code, fit_nodes, node_results)

    try:
        final_lr = await check_lean_async(
            assembled, lean_server_url, lean_timeout,
        )
        final_ana = analyse_result(final_lr)
        compile_pass = final_ana["pass_"]
        compile_time_total += final_ana["time"]
    except Exception as e:
        log.warning("assembly: final sanity compile raised: %s", e)
        final_lr = {"error": str(e)}
        compile_pass = False

    return {
        "attempted": True,
        "assembled_code": assembled,
        "n_fit": len(fit_names),
        "n_unfit": len(unfit_names),
        "n_pending": len(pending),
        "fit_node_names": fit_names,
        "unfit_node_names": unfit_names,
        "pending_node_names": pending,
        "compile_pass": compile_pass,
        "compile_time": compile_time_total,
        "lean_result": final_lr,
    }


def _markdown_assembly_block(asm: dict) -> str:
    """Render the proof-reuse section of a successful tool response."""
    parts = ["# Proof-reuse check", ""]
    if not asm.get("attempted"):
        parts.append(
            "No declaration in the revised skeleton matches any "
            "previously-proved node — nothing to reuse."
        )
        return "\n".join(parts)
    fit = asm.get("fit_node_names") or []
    unfit = asm.get("unfit_node_names") or []
    pending = asm.get("pending_node_names") or []
    parts.append(
        "For each declaration in the revised skeleton we tested any "
        "prior proof body against the new parent statements in "
        "isolation."
    )
    parts.append("")
    if fit:
        parts.append(
            "The following declarations carry forward from the "
            "previous prover pass — their prior proof bodies still "
            "fit the revised parent statements and have been grafted "
            "into the assembly: "
            + ", ".join(f"`{n}`" for n in fit)
            + "."
        )
        parts.append("")
    if unfit:
        parts.append(
            "The following declarations had a prior proof body that "
            "no longer fits the revised parent statements, so they "
            "are left as `sorry_using` and need fresh work: "
            + ", ".join(f"`{n}`" for n in unfit)
            + "."
        )
        parts.append("")
    if pending:
        parts.append(
            "The following declarations have no prior proof to "
            "reuse, so they are left as `sorry_using` and need fresh "
            "work: "
            + ", ".join(f"`{n}`" for n in pending)
            + "."
        )
        parts.append("")
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Lean compile handler — pre-checks → compile → skeleton-out →
#  assembly check → graph-validity check.
# ---------------------------------------------------------------------------

async def _handle_lean_compile_revise(
    code: str,
    lean_server_url: str,
    lean_timeout: int,
    formal_statement: str,
    main_theorem_name: str,
    tag: str,
    turn_idx: int,
    turn_record: dict,
    trace: dict,
    node_results: dict,
) -> str:
    """Compile a revised skeleton.

    Pre-compile safeguards (each rejected via the unified `Safeguard \
rejected with N violation.` status line so the model can distinguish \
pre-compile rejection from a real Lean error):
      0. `/- ... -/` block-comment balance
      1. main theorem present
      2. forbidden constructs (axiom, native_decide)
      3. imports (`Mathlib` + `Architect`)
      4. main theorem signature byte-matches `formal_statement`
      5. no bare `sorry`
      6. every theorem/lemma is `@[blueprint]`-annotated
      7. SKELETON-OUT: every theorem/lemma body is `:= by sorry_using \
[deps]` (no real tactics — the next pipeline stage fills them in)

    On all pre-checks pass: graft the canonical main signature, send to \
gateway. On compile pass: run the per-node assembly check (test-graft \
each prior proof body in isolation; build a partial assembly with \
fitting bodies) and then the graph-validity check (statement / proof \
fields, cycles / self-loops / undeclared deps, single main theorem, \
reachability from main theorem). All three (compile + assembly + \
validity) must pass for the revision to be accepted.
    """
    pre_checks: list[tuple[str, list[dict] | None]] = [
        ("unbalanced_block_comment", check_unbalanced_block_comment(code)),
        (
            "missing_theorem",
            check_missing_theorem(code, main_theorem_name, formal_statement),
        ),
        ("safeguard_precheck", check_forbidden_constructs(code)),
        ("missing_imports", check_imports(code)),
        (
            "signature_mismatch",
            check_main_signature_matches(
                code, formal_statement, main_theorem_name,
            ),
        ),
        ("bare_sorry", check_no_bare_sorry(code)),
        ("unannotated_theorem", check_all_theorems_annotated(code)),
        ("non_skeleton_body", check_skeleton_out(code)),
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
                if "axiom" in disp:
                    tags.append("axiom")
                elif "native_decide" in disp:
                    tags.append("native_decide")
                else:
                    tags.append("forbidden")
            turn_record["safeguard_violations"] = tags
        return _render_violation_blocks(violations)

    # Graft canonical main signature
    if main_theorem_name:
        safe_code = graft_main_signature(code, formal_statement, main_theorem_name)
    else:
        safe_code = code
    turn_record["safe_code"] = safe_code

    code_lines = len(safe_code.split("\n"))
    log.info(
        "[%s] turn %d: lean_compile (%d lines)",
        tag, turn_idx + 1, code_lines,
    )

    try:
        lean_result = await check_lean_async(
            safe_code, lean_server_url, lean_timeout,
        )
        analysis = analyse_result(lean_result)
    except Exception as e:
        log.error("[%s] Lean error turn %d: %s", tag, turn_idx + 1, e)
        turn_record["status"] = "lean_error"
        turn_record["error"] = str(e)
        return f"Lean 4 server error: {e}"

    trace["total_lean_time"] += analysis["time"]
    turn_record["lean_result"] = lean_result
    turn_record["compilation_pass"] = analysis["pass_"]
    turn_record["compilation_complete"] = analysis["complete"]
    turn_record["n_errors"] = (
        0 if analysis["pass_"] else len(analysis["errors"])
    )
    lean_time = analysis["time"]

    if not analysis["pass_"]:
        turn_record["status"] = "compile_fail"
        first_err = analysis["errors"][0] if analysis["errors"] else {}
        err_line = first_err.get("pos", {}).get("line", "?")
        err_msg = str(first_err.get("data", ""))[:120]
        n_err = len(analysis["errors"])
        log.info(
            "[%s] turn %d: COMPILE FAIL — %d %s, %.1fs, first@L%s: %s",
            tag, turn_idx + 1, n_err,
            "error" if n_err == 1 else "errors",
            lean_time, err_line, err_msg,
        )
        return format_compile_response(safe_code, lean_result)

    # Compile clean — run graph validity (same gate as blueprint
    # generation).
    issues = check_blueprint_graph_validity(safe_code, main_theorem_name)
    turn_record["validation_issues"] = len(issues) if issues else 0
    if issues:
        n_i = len(issues)
        log.info(
            "[%s] turn %d: COMPILE OK but graph validation FAILED — "
            "%d %s, %.1fs (sorry_count=%d)",
            tag, turn_idx + 1, n_i,
            "issue" if n_i == 1 else "issues",
            lean_time, len(analysis["sorries"]),
        )
        turn_record["status"] = "validation_fail"
        return _render_validation_blocks(issues)

    # Final acceptance gate: per-target dep block must compile (mirrors
    # what the downstream prover extracts). On failure the model gets up
    # to blueprint._DEP_BLOCK_FEEDBACK_TURNS repair turns (see
    # blueprint._dep_block_feedback), then the sample is abandoned.
    dep_fail = await graph.check_dep_blocks_compile(
        safe_code, lean_server_url, lean_timeout,
    )
    if dep_fail is not None:
        fail_name, fail_errors = dep_fail
        err_data = [e.get("data", "")[:120] for e in fail_errors[:3]]
        turn_record["status"] = "dep_block_fail"
        turn_record["dep_block_fail_target"] = fail_name
        turn_record["dep_block_fail_errors"] = fail_errors
        n_fails = trace.get("dep_block_fail_count", 0) + 1
        trace["dep_block_fail_count"] = n_fails
        if n_fails <= blueprint._DEP_BLOCK_FEEDBACK_TURNS:
            log.warning(
                "[%s] turn %d: revised skeleton compiles but per-node "
                "dep block for `%s` does not (%d %s) — feedback %d/%d. "
                "First errors: %s",
                tag, turn_idx + 1, fail_name, len(fail_errors),
                "error" if len(fail_errors) == 1 else "errors",
                n_fails, blueprint._DEP_BLOCK_FEEDBACK_TURNS,
                "; ".join(err_data),
            )
            return blueprint._dep_block_feedback(
                safe_code, fail_name, fail_errors,
                n_fails, blueprint._DEP_BLOCK_FEEDBACK_TURNS,
            )
        log.warning(
            "[%s] turn %d: revised skeleton compiles but per-node dep "
            "block for `%s` does not (%d %s) — feedback exhausted, "
            "abandoning sample. First errors: %s",
            tag, turn_idx + 1, fail_name, len(fail_errors),
            "error" if len(fail_errors) == 1 else "errors",
            "; ".join(err_data),
        )
        trace["dep_block_fail_abort"] = True
        trace["dep_block_fail_target"] = fail_name
        return (
            f"Revised skeleton compiles but the per-node dependency "
            f"block for `{fail_name}` fails to compile. Abandoning "
            f"this revise sample (the next num_sample retry will start "
            f"fresh)."
        )

    log.info(
        "[%s] turn %d: SUCCESS — revised skeleton compiles "
        "(sorry_count=%d, %.1fs)",
        tag, turn_idx + 1, len(analysis["sorries"]), lean_time,
    )
    turn_record["status"] = "success"
    trace["success"] = True
    trace["complete"] = analysis["complete"]

    # Per-node isolated assembly check.
    asm = await _assemble_and_check(
        safe_code, node_results, lean_server_url, lean_timeout,
    )
    trace["total_lean_time"] += asm["compile_time"]
    trace["final_code"] = strip_annotations(
        asm["assembled_code"] if asm["attempted"] else safe_code
    )
    trace["final_skeleton"] = strip_annotations(safe_code)
    turn_record["assembly"] = {
        "attempted": asm["attempted"],
        "n_fit": asm["n_fit"],
        "n_unfit": asm["n_unfit"],
        "n_pending": asm["n_pending"],
        "fit_node_names": asm["fit_node_names"],
        "unfit_node_names": asm["unfit_node_names"],
        "pending_node_names": asm["pending_node_names"],
        "compile_pass": asm["compile_pass"],
        "compile_time": asm["compile_time"],
    }
    trace["assembly_attempted"] = asm["attempted"]
    trace["assembly_compile_pass"] = asm["compile_pass"]
    trace["assembly_n_fit"] = asm["n_fit"]
    trace["assembly_n_unfit"] = asm["n_unfit"]
    trace["assembly_n_pending"] = asm["n_pending"]
    trace["assembly_fit_names"] = asm["fit_node_names"]
    trace["assembly_unfit_names"] = asm["unfit_node_names"]
    if asm["attempted"]:
        log.info(
            "[%s] turn %d: assembly fit=%d unfit=%d pending=%d "
            "final_compile=%s unfit_names=%s (%.1fs)",
            tag, turn_idx + 1,
            asm["n_fit"], asm["n_unfit"], asm["n_pending"],
            "PASS" if asm["compile_pass"] else "FAIL",
            asm["unfit_node_names"] or "[]",
            asm["compile_time"],
        )
        if not asm["compile_pass"]:
            log.warning(
                "[%s] partial assembly compile unexpectedly FAILED "
                "despite all per-node checks passing "
                "(fit=%s, unfit=%s, pending=%s, time=%.2fs)",
                tag,
                asm["fit_node_names"], asm["unfit_node_names"],
                asm["pending_node_names"], asm["compile_time"],
            )
    else:
        log.info(
            "[%s] turn %d: assembly skipped (no prior proofs reusable)",
            tag, turn_idx + 1,
        )

    head = (
        "# Compilation\n\n"
        "Compilation SUCCESSFUL. Validation SUCCESSFUL. Revised "
        "skeleton compiles and the blueprint graph is valid.\n\n"
    )
    return head + _markdown_assembly_block(asm)


# ---------------------------------------------------------------------------
#  Per-sample agentic loop — single conversation, OpenRouter streaming.
# ---------------------------------------------------------------------------

async def _run_one_sample_inner(
    client: AsyncOpenAI,
    model_name: str,
    *,
    pid: str,
    sample_idx: int,
    skeleton: str,
    annotated_skeleton: str,
    input_partial_assembly: str,
    formal_statement: str,
    main_theorem_name: str,
    node_results: dict,
    lean_server_url: str,
    tokenizer,
    tools_overhead: int,
    max_turns: int,
    temperature: float,
    lean_timeout: int,
    max_model_len: int,
    max_output_tokens: int | None,
    extra_body: dict | None,
    search_server_url: str | None = None,
) -> dict:
    tag = f"{pid}/s{sample_idx}"
    log.info(
        "[%s] revise start (max_turns=%d)", tag, max_turns,
    )

    user_prompt = build_user_prompt(annotated_skeleton, formal_statement)
    system_prompt = system_prompt_for(bool(search_server_url))
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    active_tools = [LEAN_COMPILE_TOOL]
    if search_server_url:
        active_tools.append(MATHLIB_SEARCH_TOOL)

    trace = {
        "problem_id": pid,
        "sample": sample_idx,
        "stage": "refine",
        "input_code": input_partial_assembly,
        "input_skeleton": skeleton,
        "annotated_skeleton": annotated_skeleton,
        "formal_statement": formal_statement,
        "main_theorem_name": main_theorem_name,
        "system_prompt": system_prompt,
        "tools": active_tools,
        "user_prompt": user_prompt,
        "max_turns": max_turns,
        "turns": [],
        "success": False,
        "complete": False,
        "total_model_tokens": {"input": 0, "output": 0},
        "total_cached_tokens": 0,
        "total_cost_usd": 0.0,
        "total_lean_time": 0.0,
        "total_search_calls": 0,
    }

    handler_kwargs = dict(
        lean_server_url=lean_server_url,
        lean_timeout=lean_timeout,
        formal_statement=formal_statement,
        main_theorem_name=main_theorem_name,
        node_results=node_results,
    )

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
                "[%s] context full on turn %d (prompt=%d tokens)",
                tag, turn_idx + 1, prompt_tokens,
            )
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
            log.error("[%s] model error turn %d: %s", tag, turn_idx + 1, e)
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
                        "arguments": (
                            strip_annotations_in_tool_args(tc.function.arguments)
                            if tc.function.name == "lean_compile"
                            else tc.function.arguments
                        ),
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
                tool_response_text = await _handle_lean_compile_revise(
                    code,
                    tag=tag, turn_idx=turn_idx,
                    turn_record=turn_record, trace=trace,
                    **handler_kwargs,
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
                if trace.get("dep_block_fail_abort"):
                    break
                continue

            log.warning(
                "[%s] no tool call on turn %d — ending rollout",
                tag, turn_idx + 1,
            )
            turn_record["status"] = "no_tool_call"
            trace["turns"].append(turn_record)
            break

        solved_this_turn = False
        tool_calls_record: list[dict] = []

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
                tool_response_text = await _handle_lean_compile_revise(
                    code,
                    tag=tag, turn_idx=turn_idx,
                    turn_record=turn_record, trace=trace,
                    **handler_kwargs,
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
                trace["final_code"] = strip_annotations(tc["code"])
                trace["final_skeleton"] = strip_annotations(tc["code"])
                break

    status = "REVISED" if trace["success"] else "FAIL"
    log.info(
        "[%s] %s — turns=%d elapsed=%.1fs cost=$%.4f",
        tag, status, trace["total_turns"], trace["elapsed_seconds"],
        trace["total_cost_usd"],
    )
    return trace


async def run_one_sample(
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    **kwargs,
) -> dict:
    async with semaphore:
        return await _run_one_sample_inner(client, model_name, **kwargs)


# ---------------------------------------------------------------------------
#  Per-problem orchestration
# ---------------------------------------------------------------------------

async def run_one_problem(
    client: AsyncOpenAI,
    model_name: str,
    problem: dict,
    semaphore: asyncio.Semaphore,
    *,
    num_samples: int,
    early_stop: bool,
    lean_server_url: str,
    tokenizer,
    tools_overhead: int,
    max_turns: int,
    temperature: float,
    lean_timeout: int,
    max_model_len: int,
    max_output_tokens: int | None,
    extra_body: dict | None,
    write_lock: asyncio.Lock,
    traces_path: Path,
    done_keys: set,
    prev_solved_pids: frozenset[str] = frozenset(),
    prune_dead: bool = True,
    search_server_url: str | None = None,
) -> list[dict]:
    pid = problem["problem_id"]
    # Resume + early-stop: if a previous run already wrote a successful
    # sample for this pid, the existing trace is the win — don't burn
    # more samples filling out the remaining (pid, sample) slots.
    if early_stop and pid in prev_solved_pids:
        return []
    skeleton = problem["skeleton"]
    partial_assembly = problem.get("partial_assembly", "") or ""
    proved = set(problem["proved_nodes"])
    unproved = set(problem["unproved_nodes"])
    forfeits = problem.get("forfeits") or {}
    disproofs = problem.get("disproofs") or {}
    formal_statement = problem["formal_statement"]
    main_theorem_name = problem["main_theorem_name"]
    node_results = (problem.get("prover_trace") or {}).get("node_results") or {}

    annotated = annotate_skeleton(
        skeleton, proved, unproved,
        forfeits=forfeits, disproofs=disproofs,
    )

    out: list[dict] = []
    for si in range(num_samples):
        if (pid, si) in done_keys:
            continue
        trace = await run_one_sample(
            client, model_name, semaphore,
            pid=pid, sample_idx=si,
            skeleton=skeleton,
            annotated_skeleton=annotated,
            input_partial_assembly=partial_assembly,
            formal_statement=formal_statement,
            main_theorem_name=main_theorem_name,
            node_results=node_results,
            lean_server_url=lean_server_url,
            tokenizer=tokenizer,
            tools_overhead=tools_overhead,
            max_turns=max_turns,
            temperature=temperature,
            lean_timeout=lean_timeout,
            max_model_len=max_model_len,
            max_output_tokens=max_output_tokens,
            extra_body=extra_body,
            search_server_url=search_server_url,
        )

        # Post-revise prune. The reviser frequently emits new helper
        # nodes that aren't actually wired into the main theorem's dep
        # tree (`\uses{p_prod}` outbound, but nothing refers back to
        # them). Without this step, dead helpers ride into the next
        # iteration's prover, the dep graph stays cluttered, and the
        # reviser sees them again on its next turn and may keep
        # extending them. Reachability runs on `final_code` because
        # solved nodes there have real bodies (more accurate scrape);
        # the same dead set then prunes `final_skeleton` so the next
        # prover's `upstream_skeleton` carry-forward stays in lockstep.
        if (
            prune_dead and trace.get("final_code") and main_theorem_name
        ):
            try:
                pruned_code, pruned_skel, _, _, ana = prune_dead_nodes(
                    code=trace["final_code"],
                    main_theorem_name=main_theorem_name,
                    skeleton=trace.get("final_skeleton"),
                    lean_server_url=lean_server_url,
                    timeout=lean_timeout,
                )
                if ana is not None and ana.dead:
                    log.info(
                        "[%s/s%d] post-revise pruned %d dead nodes: "
                        "%s (reachable=%d unsolved=%d)",
                        pid, si, len(ana.dead), sorted(ana.dead),
                        len(ana.reachable), len(ana.unsolved_targets),
                    )
                    trace["final_code"] = pruned_code
                    if pruned_skel is not None:
                        trace["final_skeleton"] = pruned_skel
            except Exception as e:
                log.warning(
                    "[%s/s%d] post-revise prune skipped: %s", pid, si, e,
                )

        out.append(trace)
        async with write_lock:
            with open(traces_path, "a") as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
        if early_stop and trace.get("success"):
            log.info("[%s] revised on sample %d (early stop)", pid, si)
            break
    return out


# ---------------------------------------------------------------------------
#  Loaders
# ---------------------------------------------------------------------------

def load_prover_traces(prover_traces_path: Path) -> dict[str, dict]:
    """Load prover traces from `traces.jsonl` \
and return `{pid: row}` keeping the EARLIEST-sample row per pid.

    Each chosen row supplies everything downstream needs:
      - `upstream_skeleton`        → pure sorry_using skeleton carried
                                     forward from the upstream sketch
                                     or the prior reviser
      - `final_code`               → partial assembly with grafted
                                     proof bodies (used by the assembly
                                     check to read prior parents)
      - `solved_nodes_derived`     → solved node names (carry-forward
                                     solved included)
      - `unsolved_nodes_derived`   → unsolved node names
      - `nodes_solved_via_negation`→ nodes whose negation pathway hit
                                     instead of the original target
      - `node_forfeits`            → per-node forfeit declarations
      - `node_disproofs`           → per-node post-success negation
                                     reflections (model-authored
                                     STATEMENT_WRONG diagnosis); only
                                     populated when --allow-negation
                                     was on for the upstream prover run
      - `formal_statement`         → byte-exact main signature
      - `main_theorem_name`        → main theorem name
      - `node_results`             → per-node prover trace dicts
    """
    if not prover_traces_path.exists():
        raise FileNotFoundError(f"Expected {prover_traces_path}")
    chosen: dict[str, dict] = {}
    with open(prover_traces_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("problem_id")
            if not pid:
                continue
            cur = chosen.get(pid)
            if cur is None or r.get("sample", 0) < cur.get("sample", 0):
                chosen[pid] = r
    return chosen


# ---------------------------------------------------------------------------
#  Per-pid revised-skeleton summary entry
# ---------------------------------------------------------------------------

def build_revised_skeleton_entry(
    pid: str,
    final_code: str,
    final_skeleton: str,
    *,
    input_meta: dict,
    revise_success: bool,
    no_op_fallback: bool,
    chosen_trace: dict | None = None,
) -> dict:
    """Build the per-pid entry for `summary.json:revised_skeletons`.

    `final_code` is the chosen sample's output blueprint: a partial \
assembly containing every per-node-fitting prior proof body grafted in, \
with `sorry_using` on the rest. Always compiles. `final_skeleton` is \
the same revision before grafting — a pure `sorry_using` skeleton, \
consumed by the next prover through the trace's `upstream_skeleton` \
carry-forward. On no-op (no sample succeeded) both fall back to the \
input skeleton.
    """
    try:
        _, final_nodes = graph.parse_blueprint_file(final_code)
    except Exception as e:
        log.warning("[%s] could not parse final code: %s", pid, e)
        final_nodes = []

    nodes_total = len(final_nodes)
    solved_nodes = [
        n.name for n in final_nodes
        if n.kind in {"theorem", "lemma"} and not n.is_target
    ]
    unsolved_nodes = [n.name for n in final_nodes if n.is_target]

    chosen_trace = chosen_trace or {}

    return {
        "problem_id": pid,
        "stage": "refine",
        "nodes_total": nodes_total,
        "nodes_solved": len(solved_nodes),
        "solved_nodes": solved_nodes,
        "unsolved_nodes": unsolved_nodes,
        "formal_statement": input_meta.get("formal_statement"),
        "main_theorem_name": input_meta.get("main_theorem_name"),
        "revise_success": revise_success,
        "no_op_fallback": no_op_fallback,
        "input_proved_nodes": list(input_meta.get("proved_nodes") or []),
        "input_unproved_nodes": list(input_meta.get("unproved_nodes") or []),
        "input_disproved_nodes": list(input_meta.get("disproved_nodes") or []),
        "input_forfeit_nodes": list(input_meta.get("forfeit_nodes") or []),
        "final_code": final_code,
        "final_skeleton": final_skeleton,
        "assembly_attempted": bool(chosen_trace.get("assembly_attempted")),
        "assembly_compile_pass": chosen_trace.get("assembly_compile_pass"),
        "assembly_n_fit": chosen_trace.get("assembly_n_fit", 0),
        "assembly_n_unfit": chosen_trace.get("assembly_n_unfit", 0),
        "assembly_n_pending": chosen_trace.get("assembly_n_pending", 0),
        "assembly_fit_names": chosen_trace.get("assembly_fit_names") or [],
        "assembly_unfit_names": chosen_trace.get("assembly_unfit_names") or [],
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

async def async_main():
    parser = argparse.ArgumentParser(
        description=(
            "Blueprint refinement: read a proving pass's traces via "
            "--prover-traces and write a revised blueprint per unsolved "
            "problem to traces.jsonl / summary.json / session.log under "
            "--output."
        )
    )
    parser.add_argument(
        "--prover-traces", required=True,
        help=(
            "Path to a prover traces.jsonl. Each line is one (problem, "
            "sample) prover trace; the earliest sample per problem is "
            "chosen."
        ),
    )
    parser.add_argument("--output", required=True, help="Output directory")
    add_model_args(parser, x_title="goedel-architect-refine")
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
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="Independent revision samples per problem",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help=(
            "Max concurrent revision conversations. If unset, auto-"
            "picks min(len(problems), 128)."
        ),
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
        "--problem-max-attempts", type=int, default=3,
        help=(
            "Retry a (problem, sample) this many times if run_one_sample "
            "raises an unexpected exception. Default: 3."
        ),
    )
    parser.add_argument(
        "--limit", type=int,
        help="Process only first N problems",
    )
    parser.add_argument(
        "--only", nargs="+",
        help="Restrict to a specific list of problem ids",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (problem, sample) already in traces.jsonl",
    )
    parser.add_argument(
        "--early-stop", action="store_true",
        help=(
            "Stop sampling a problem after the first revision succeeds "
            "(sequential attempts within a problem)"
        ),
    )
    parser.add_argument(
        "--prune-dead", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "After revision, prune nodes unreachable from the main "
            "theorem in the reviser's own output. ON by default; "
            "use --no-prune-dead to disable."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    log_path = setup_logger(out_dir)
    log.info("Output directory: %s", out_dir)
    log.info("Session log: %s", log_path)
    log.info("Prover traces: %s", args.prover_traces)
    log.info("Model URL: %s", args.model_url)
    log.info("Model name: %s", args.model_name)
    log.info("Max model len: %d", args.max_model_len)
    log.info(
        "Max output tokens: %s",
        args.max_output_tokens if args.max_output_tokens is not None
        else "<unset — context-window remainder>",
    )

    api_key = read_api_key(args)

    tokenizer = load_tokenizer(args.tokenizer_path)
    # The token-overhead estimate uses blueprint generation's tool list, as
    # in the runs reported in the paper.
    tools_overhead = estimate_tools_overhead(tokenizer, blueprint.TOOLS)

    prover_rows = load_prover_traces(Path(args.prover_traces))
    log.info("Loaded %d prover-trace rows", len(prover_rows))

    failed_pids = {
        pid for pid, row in prover_rows.items()
        if row.get("unsolved_nodes_derived")
        or row.get("nodes_solved_via_negation")
    }
    skipped_fully_solved = sorted(set(prover_rows) - failed_pids)
    log.info(
        "Filtered to %d failed problems (skipped %d fully-solved)",
        len(failed_pids), len(skipped_fully_solved),
    )

    pids = sorted(failed_pids)
    only_set = set(args.only) if args.only else None
    if only_set:
        pids = [p for p in pids if p in only_set]
    if args.limit:
        pids = pids[: args.limit]
    log.info("Will revise %d problems", len(pids))

    if only_set:
        missing = [p for p in only_set if p not in prover_rows]
        if missing:
            log.warning(
                "--only ids missing from --prover-traces: %s", missing,
            )
        already = [p for p in only_set if p in skipped_fully_solved]
        if already:
            log.warning(
                "--only ids already fully solved upstream: %s", already,
            )

    problems: list[dict] = []
    for pid in pids:
        row = prover_rows[pid]
        solved_derived = list(row.get("solved_nodes_derived") or [])
        unsolved_derived = list(row.get("unsolved_nodes_derived") or [])
        nodes_via_neg = set(row.get("nodes_solved_via_negation") or [])
        node_disproofs = row.get("node_disproofs") or {}
        node_results = row.get("node_results") or {}

        # A solved-via-negation node IS in `solved_nodes_derived`
        # (because success=True), but the original target is not
        # actually proved — it's the NEGATION that's proved. From the
        # reviser's perspective the original lemma is UNPROVED (and in
        # fact, formally disproved). Move those names from proved to
        # unproved.
        proved_nodes = [n for n in solved_derived if n not in nodes_via_neg]
        unproved_nodes = list(unsolved_derived) + sorted(
            n for n in nodes_via_neg if n not in unsolved_derived
        )

        # `disproofs` carries the model-authored reflection blob per
        # disproved node (same shape as a forfeit_declaration). Nodes
        # that registered a negation solve but lack a reflection (the
        # turn errored, parse failed, or this trace predates the
        # feature) still appear here with an empty dict so they get the
        # synthetic-boilerplate fallback in `_format_disproof_block`.
        disproofs: dict[str, dict] = {
            n: node_disproofs.get(n, {}) for n in nodes_via_neg
        }

        # `forfeits` is re-derived from `node_results` via the
        # diagnosis-priority picker. For each truly-unsolved node we
        # walk every retry attempt (winner + prior_attempts) and
        # select the forfeit_declaration whose Diagnosis label is most
        # informative (STATEMENT_WRONG > PROOF_TOO_HARD > unparseable),
        # ties broken by attempt total_turns. The prover's pre-baked
        # `row["node_forfeits"]` is winner-only and gets discarded
        # here; this rebuild rescues higher-tier diagnoses that the
        # winner-only path would have lost. Nodes already in
        # `disproofs` are skipped — `annotate_skeleton` prefers the
        # disproof block when both are present.
        forfeits: dict[str, dict] = {}
        promoted = 0
        for n_name in unproved_nodes:
            if n_name in nodes_via_neg:
                continue
            nr = node_results.get(n_name)
            if not isinstance(nr, dict):
                continue
            picked = _pick_forfeit_by_priority(nr)
            if picked is None:
                continue
            forfeits[n_name] = picked
            winner_fd = nr.get("forfeit_declaration")
            if winner_fd is not None and picked is not winner_fd:
                promoted += 1
                log.debug(
                    "[%s/%s] forfeit picker chose prior-attempt "
                    "forfeit (winner_label=%s, picked_label=%s)",
                    pid, n_name,
                    _diagnosis_label(winner_fd),
                    _diagnosis_label(picked),
                )
        if promoted:
            log.info(
                "[%s] forfeit picker promoted %d/%d node forfeit(s) "
                "from prior retry attempts",
                pid, promoted, len(forfeits),
            )

        raw_skeleton = row.get("upstream_skeleton") or ""
        if not raw_skeleton:
            log.error(
                "[%s] prover trace missing `upstream_skeleton`; skipping.",
                pid,
            )
            continue
        skeleton = strip_pre_blueprint_doc_comments(raw_skeleton)
        partial_assembly = row.get("final_code") or ""
        main_name = row.get("main_theorem_name") or ""
        formal_statement = row.get("formal_statement") or ""

        problems.append({
            "problem_id": pid,
            "skeleton": skeleton,
            "partial_assembly": partial_assembly,
            "prover_trace": row,
            "proved_nodes": proved_nodes,
            "unproved_nodes": unproved_nodes,
            "forfeits": forfeits,
            "disproofs": disproofs,
            "formal_statement": formal_statement,
            "main_theorem_name": main_name,
            "input_meta": {
                "proved_nodes": proved_nodes,
                "unproved_nodes": unproved_nodes,
                "disproved_nodes": sorted(disproofs.keys()),
                "forfeit_nodes": sorted(forfeits.keys()),
                "formal_statement": formal_statement,
                "main_theorem_name": main_name,
                "nodes_total": row.get("nodes_total"),
            },
        })

    bad = [(p["problem_id"], "formal_statement")
           for p in problems if not p["formal_statement"]]
    bad += [(p["problem_id"], "main_theorem_name")
            for p in problems if not p["main_theorem_name"]]
    if bad:
        for pid, field in bad[:10]:
            log.error("Problem %s missing %r", pid, field)
        log.error(
            "Aborting: %d problems missing required fields", len(bad),
        )
        sys.exit(1)

    if args.concurrency is None:
        concurrency = min(max(1, len(problems)), 128)
        log.info(
            "Concurrency: auto -> %d (= min(len(problems)=%d, 128))",
            concurrency, len(problems),
        )
    else:
        concurrency = args.concurrency
        log.info("Concurrency: %d (explicit)", concurrency)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.model_url,
        default_headers=default_headers(args),
        timeout=httpx.Timeout(120.0, connect=15.0),
    )
    extra_body = build_extra_body(args)
    search_server_url = resolve_search_server(args.search_server)

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    traces_path = out_dir / "traces.jsonl"

    done_keys: set[tuple[str, int]] = set()
    existing: list[dict] = []
    solved_pids: set[str] = set()
    if args.resume and traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing.append(r)
                done_keys.add((r["problem_id"], r.get("sample", 0)))
                if r.get("success"):
                    solved_pids.add(r["problem_id"])
        log.info("Resume: %d completed samples loaded", len(done_keys))
        if args.early_stop:
            log.info(
                "Early-stop resume: %d already solved", len(solved_pids),
            )

    if (
        not args.resume
        and traces_path.exists()
        and traces_path.stat().st_size > 0
    ):
        log.error(
            "Output %s already contains results. Pass --resume to "
            "continue or choose a new --output dir.",
            traces_path,
        )
        sys.exit(1)
    if not args.resume:
        traces_path.write_text("")

    t_start = time.time()

    log.info(
        "Scheduling %d problems × up to %d samples "
        "(concurrency=%d, early_stop=%s)",
        len(problems), args.num_samples, concurrency, args.early_stop,
    )

    results = await asyncio.gather(*(
        run_one_problem(
            client, args.model_name, p, semaphore,
            num_samples=args.num_samples,
            early_stop=args.early_stop,
            lean_server_url=args.lean_server,
            tokenizer=tokenizer,
            tools_overhead=tools_overhead,
            max_turns=args.max_turns,
            temperature=args.temperature,
            lean_timeout=args.lean_timeout,
            max_model_len=args.max_model_len,
            max_output_tokens=args.max_output_tokens,
            extra_body=extra_body,
            write_lock=write_lock,
            traces_path=traces_path,
            done_keys=done_keys,
            prev_solved_pids=frozenset(solved_pids),
            prune_dead=args.prune_dead,
            search_server_url=search_server_url,
        )
        for p in problems
    ), return_exceptions=True)

    per_pid: dict[str, list[dict]] = {}
    for e in existing:
        per_pid.setdefault(e["problem_id"], []).append(e)
    for p, r in zip(problems, results):
        pid = p["problem_id"]
        if isinstance(r, Exception):
            log.error("[%s] problem-level exception: %s", pid, r)
            per_pid.setdefault(pid, []).append({
                "problem_id": pid, "sample": -1, "success": False,
                "status": "exception", "error": str(r),
            })
        else:
            per_pid.setdefault(pid, []).extend(r)

    revised_skeletons: dict[str, dict] = {}
    for p in problems:
        pid = p["problem_id"]
        traces = per_pid.get(pid, [])
        successes = [
            t for t in traces if t.get("success") and t.get("final_code")
        ]
        if successes:
            successes.sort(key=lambda t: t.get("sample", 0))
            chosen = successes[0]
            final_code = chosen["final_code"]
            final_skeleton = chosen["final_skeleton"]
            no_op = False
            success = True
        else:
            chosen = None
            final_code = p["skeleton"]
            final_skeleton = p["skeleton"]
            no_op = True
            success = False
            log.info(
                "[%s] no successful sample; falling back to input skeleton",
                pid,
            )
        revised_skeletons[pid] = build_revised_skeleton_entry(
            pid, final_code, final_skeleton,
            input_meta=p["input_meta"],
            revise_success=success,
            no_op_fallback=no_op,
            chosen_trace=chosen,
        )

    elapsed = time.time() - t_start
    n_problems = len(per_pid)
    succ_counts = {
        pid: sum(1 for t in ts if t.get("success"))
        for pid, ts in per_pid.items()
    }
    solved_any = sum(1 for c in succ_counts.values() if c > 0)
    total_input_tokens = sum(
        t.get("total_model_tokens", {}).get("input", 0)
        for ts in per_pid.values() for t in ts
    )
    total_output_tokens = sum(
        t.get("total_model_tokens", {}).get("output", 0)
        for ts in per_pid.values() for t in ts
    )
    total_cached_tokens = sum(
        t.get("total_cached_tokens", 0)
        for ts in per_pid.values() for t in ts
    )
    total_cost_usd = sum(
        t.get("total_cost_usd", 0.0)
        for ts in per_pid.values() for t in ts
    )
    cache_hit_rate = (
        total_cached_tokens / total_input_tokens
        if total_input_tokens else 0.0
    )

    pak: dict = {}
    for k in sorted(
        {1, 2, 4, args.num_samples} & set(range(1, args.num_samples + 1))
    ):
        eligible = [
            (pid, len(per_pid[pid]), succ_counts[pid])
            for pid in per_pid if len(per_pid[pid]) >= k
        ]
        if not eligible:
            continue
        total = sum(pass_at_k(n_actual, c, k) for _, n_actual, c in eligible)
        rate = total / n_problems if n_problems else 0
        pak[k] = {
            "rate": round(rate, 6),
            "solved_estimate": round(rate * n_problems, 1),
            "total": n_problems,
            "eligible": len(eligible),
        }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "stage": "refine",
        "prover_traces": args.prover_traces,
        "model_url": args.model_url,
        "model_name": args.model_name,
        "lean_server": args.lean_server,
        "search_server": search_server_url,
        "max_turns": args.max_turns,
        "num_samples": args.num_samples,
        "concurrency": concurrency,
        "temperature": args.temperature,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "provider": args.provider,
        "lean_timeout": args.lean_timeout,
        "early_stop": args.early_stop,
        "total_problems": n_problems,
        "solved_any_sample": solved_any,
        "solve_rate_any": (
            solved_any / n_problems if n_problems else 0
        ),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_tokens": total_cached_tokens,
        "cache_hit_rate": round(cache_hit_rate, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "pass@k": pak,
        "elapsed_seconds": round(elapsed, 2),
        "per_problem_results": {
            pid: [
                {
                    "sample": t.get("sample"),
                    "success": t.get("success"),
                    "complete": t.get("complete"),
                    "turns": t.get("total_turns"),
                    "elapsed": t.get("elapsed_seconds"),
                    "tokens_in": (
                        t.get("total_model_tokens") or {}
                    ).get("input", 0),
                    "tokens_out": (
                        t.get("total_model_tokens") or {}
                    ).get("output", 0),
                    "cached_tokens": t.get("total_cached_tokens", 0),
                    "cost_usd": t.get("total_cost_usd", 0.0),
                    "lean_time": t.get("total_lean_time", 0),
                    "assembly_attempted": t.get("assembly_attempted"),
                    "assembly_compile_pass": t.get("assembly_compile_pass"),
                    "assembly_n_fit": t.get("assembly_n_fit", 0),
                    "assembly_n_unfit": t.get("assembly_n_unfit", 0),
                    "assembly_n_pending": t.get("assembly_n_pending", 0),
                }
                for t in ts
            ]
            for pid, ts in per_pid.items()
        },
        "revised_skeletons": revised_skeletons,
        "skipped_fully_solved": skipped_fully_solved,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info("=" * 60)
    log.info(
        "DONE: %d/%d successful revisions (%.1f%%) — "
        "%d samples × %d turns",
        solved_any, n_problems,
        100 * solved_any / n_problems if n_problems else 0,
        args.num_samples, args.max_turns,
    )
    for k, info in pak.items():
        log.info(
            "  pass@%d = %.1f%% (~%.0f/%d)",
            k, info["rate"] * 100, info["solved_estimate"], n_problems,
        )
    log.info(
        "Tokens: input=%d (cached=%d, %.1f%% hit) output=%d",
        total_input_tokens, total_cached_tokens,
        100 * cache_hit_rate, total_output_tokens,
    )
    log.info("Total cost (OpenRouter billed): $%.4f", total_cost_usd)
    log.info("Total wall time: %.1fs", elapsed)
    log.info("Results: %s", summary_path)


def main():
    # Line-buffer output so progress shows up promptly when piped.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
