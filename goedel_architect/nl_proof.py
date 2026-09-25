"""Natural-language proof guidance (optional; paper §3.1, NL-proof guidance).

Produces a natural-language proof for each problem with a
generation / grading / refinement loop, to guide blueprint generation
(`blueprint.py --nl-proof-decompose`). Per problem:

    Round 0:  K_SOLVERS parallel solves                       (K calls)
    Round k=1..MAX_ROUNDS:
        - parallel grade each candidate                       (K calls)
        - if max score >= EARLY_STOP_SCORE (confirmed by CONSENSUS_PASSES
          independent grades): return best
        - take top K_REFINE candidates with grader feedback
        - parallel refine                                     (K_REFINE calls)
    Final grade pass, return best-scoring proof

The solver prompt anchors the proof to the formal Lean statement and asks
for a decomposable structure (small self-contained lemmas); the grader
scores rigor and decomposability on a 0-10 scale.

Output:
    <out>/<pid>/result.json   {uuid, problem_id, status, proof, score, ...}
                              status == "solved" iff best score >= --solve-threshold
    <out>/<pid>/proof.txt     the proof
    <out>/<pid>/multipass.json per-round trace (all candidates, scores)
    <out>/traces.jsonl, <out>/summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import openai

from . import prompts

# Fixed name (not __name__): when run with `python -m`, __name__ is "__main__".
log = logging.getLogger("goedel_architect.nl_proof")


# ---------------------------------------------------------------------------
# Prompts (goedel_architect/prompts/nl_proof_*.md)
# ---------------------------------------------------------------------------

# Solver = base Lean-aware prompt + decomposability + library-availability
# + rigor addenda (rigor last, so it is the dominant instruction).
SOLVER_SYSTEM_PROMPT_DECOMP = prompts.load("nl_proof_solver")

SOLVER_USER_TEMPLATE = """\
Problem id: {problem_id}

== Problem statement (natural language, for context) ==
{problem}

== Formal statement (the source of truth — the claim you must prove) ==
{formal_statement}

Write a rigorous, fully detailed natural-language proof of the formal statement. Type every variable to match the formal statement, invoke every formal hypothesis at its point of use, and show every key equation. Output the proof body only — no preamble, no closing remark, no markers, no commentary."""


def extract_proof(text: str) -> str | None:
    """Use the model output as the proof body verbatim.

    The system prompt forbids preambles, closing remarks, and markers, so
    the response *is* the proof. Whitespace-only output is treated as
    empty.
    """
    if not text:
        return None
    body = text.strip()
    return body or None


GRADER_SYSTEM_PROMPT = prompts.load("nl_proof_grader")


GRADER_USER_TEMPLATE = """\
Problem id: {problem_id}

== Problem statement (natural language) ==
{problem}

== Formal statement (the source of truth — what the proof must establish) ==
{formal_statement}

== Candidate proof to grade ==
{proof}

Grade the candidate proof on the 0-10 scale defined in the system prompt. Output a single JSON object as specified — no preamble, no markdown fences, no closing remark."""


# Refiner uses the same solver system prompt as round-0 solving (so the
# model's *target* doesn't change), plus a refinement addendum; the user
# prompt adds the prior attempt and the grader's structured feedback.
REFINER_SYSTEM_PROMPT = SOLVER_SYSTEM_PROMPT_DECOMP + prompts.load("nl_proof_refiner_addendum")


REFINER_USER_TEMPLATE = """\
Problem id: {problem_id}

== Problem statement (natural language) ==
{problem}

== Formal statement (the source of truth — what the proof must establish) ==
{formal_statement}

== Previous attempt (grader score: {prev_score}/10, verdict: {prev_verdict}) ==
{prev_proof}

== Grader's structured feedback ==
{grader_feedback_block}

Produce a refined proof that addresses every issue identified above. Output the proof body only — no preamble, no markers, no closing remark."""


# ---------------------------------------------------------------------------
# OpenAI-compatible client (OpenRouter by default)
# ---------------------------------------------------------------------------

def _get_api_key(env_var: str = "OPENROUTER_API_KEY") -> str:
    key = os.environ.get(env_var)
    if key:
        return key.strip()
    raise ValueError(f"{env_var} is not set")


def _build_extra_body(provider: str, reasoning_effort: str,
                      model_url: str = "") -> dict:
    """Request fields beyond the OpenAI schema (see `llm.build_extra_body`)."""
    if model_url and "openrouter.ai" not in model_url:
        # Other OpenAI-compatible endpoints (e.g. Google AI Studio) reject
        # unknown body fields (usage/provider/reasoning); use the standard
        # reasoning_effort parameter (xhigh -> high, Gemini's max).
        eb: dict = {}
        if reasoning_effort != "off":
            eb["reasoning_effort"] = {"xhigh": "high"}.get(
                reasoning_effort, reasoning_effort)
        return eb
    eb = {"usage": {"include": True}}
    if provider:
        eb["provider"] = {"order": [provider], "allow_fallbacks": False}
    if reasoning_effort != "off":
        eb["reasoning"] = {"effort": reasoning_effort}
    return eb


PERMANENT_ERRORS = (
    openai.NotFoundError, openai.AuthenticationError,
    openai.PermissionDeniedError, openai.BadRequestError,
)


async def _stream_chat(
    client: openai.AsyncOpenAI, *, per_chunk_timeout: float = 240.0, **kwargs
) -> dict:
    """Call chat.completions.create with stream=True and assemble the
    result. Streaming avoids the non-streaming failure modes that bite
    large/slow responses (V4 Pro especially): a giant buffered HTTP body
    getting truncated mid-JSON, and idle read-timeouts while the whole
    response generates.

    Timeout model: there is NO total-request cap — only a *per-chunk*
    read timeout. A long but healthy
    stream (V4 Pro reasoning for 10+ min, chunks every ~0.1s) is fine; a
    genuinely stalled stream (no chunk for `per_chunk_timeout` s) raises
    TimeoutError and the caller retries. This is the key fix: a total
    asyncio.wait_for cap would kill a healthy long V4 Pro reasoning stream.

    Returns {content, finish_reason, prompt_tokens, completion_tokens,
             cached_tokens, cost_usd}. Raises on a stream that ends with
             no finish_reason (connection died mid-response) so the
             caller's retry loop kicks in.
    """
    kwargs = dict(kwargs)
    kwargs["stream"] = True
    kwargs.setdefault("stream_options", {"include_usage": True})

    content_parts: list[str] = []
    finish_reason = None
    prompt_tokens = completion_tokens = cached_tokens = 0
    cost_usd = 0.0

    stream = await client.chat.completions.create(**kwargs)
    try:
        stream_iter = stream.__aiter__()
        while True:
            # Per-chunk timeout: a stalled stream trips this; a healthy
            # long reasoning stream never does (chunks keep arriving).
            try:
                chunk = await asyncio.wait_for(
                    stream_iter.__anext__(), timeout=per_chunk_timeout
                )
            except StopAsyncIteration:
                break
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
                u = chunk.usage.model_dump() if hasattr(chunk.usage, "model_dump") else {}
                ptd = u.get("prompt_tokens_details") or {}
                ct = ptd.get("cached_tokens")
                if ct is not None:
                    cached_tokens = ct
                c = u.get("cost")
                if c is not None:
                    cost_usd = float(c)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            # reasoning deltas are not part of the proof and are not kept
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    if finish_reason is None:
        # Provider streamed (or didn't) but never sent a finish_reason —
        # the upstream connection died mid-response. Raise so the retry
        # loop treats it as transient.
        raise RuntimeError(
            "stream ended with no finish_reason "
            f"(completion_tokens={completion_tokens}, "
            f"content_chars={sum(len(c) for c in content_parts)})"
        )
    return {
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cost_usd": cost_usd,
    }


async def call_model(
    client: openai.AsyncOpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    extra_body: dict,
    temperature: float,
    top_p: float,
    max_output_tokens: int | None,
    request_timeout: float,
    tag: str,
) -> dict:
    """Single streaming chat completion, 4-retry exp-backoff on transient
    errors. Streams (via _stream_chat) so large/slow V4-class responses
    don't get truncated or time out.

    Returns dict {ok, content, prompt_tokens, completion_tokens,
                  cached_tokens, cost_usd, elapsed_seconds, attempts,
                  error?}
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "extra_body": extra_body,
    }
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens

    t0 = time.monotonic()
    last_err: str | None = None
    # 8 attempts / backoff capped at 120s (~4 min coverage): a multi-minute
    # upstream outage should degrade the run, not kill every in-flight
    # problem. The tool-loop stages use max_retries=12.
    for attempt in range(8):
        try:
            # No total cap — _stream_chat enforces a per-chunk timeout
            # (request_timeout here is reused as the per-chunk read
            # timeout). A long-but-healthy V4 Pro reasoning stream must
            # not be killed by a total deadline.
            r = await _stream_chat(
                client, per_chunk_timeout=request_timeout, **kwargs
            )
            return {
                "ok": True,
                "content": r["content"],
                "finish_reason": r["finish_reason"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "cached_tokens": r["cached_tokens"],
                "cost_usd": r["cost_usd"],
                "elapsed_seconds": time.monotonic() - t0,
                "attempts": attempt + 1,
            }
        except asyncio.CancelledError:
            raise
        except PERMANENT_ERRORS as e:
            last_err = f"{type(e).__name__}: {e}"
            log.error("[%s] permanent: %s", tag, last_err)
            return {
                "ok": False, "error": last_err, "permanent": True,
                "elapsed_seconds": time.monotonic() - t0,
                "attempts": attempt + 1,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            wait = min(2 ** attempt, 120)
            log.warning("[%s] transient (attempt %d/8): %s; sleeping %ds",
                        tag, attempt + 1, last_err, wait)
            await asyncio.sleep(wait)
    return {
        "ok": False, "error": last_err or "unknown",
        "elapsed_seconds": time.monotonic() - t0, "attempts": 8,
    }


# ---------------------------------------------------------------------------
# Round logic
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One proof attempt with its grade + grader feedback."""
    proof: str
    raw_response: str
    role: str            # 'solver' | 'refiner'
    round_idx: int       # 0 for initial solve, 1+ for refinement
    sample_idx: int      # index among siblings produced this round
    score: int | None = None
    grader_payload: dict | None = None  # full JSON parsed from grader
    solver_call: dict = field(default_factory=dict)
    grader_call: dict = field(default_factory=dict)
    parent_proof: str | None = None     # only set for refiner candidates


def _parse_grader_json(text: str) -> dict | None:
    """Pull the JSON object out of the grader response.

    Models sometimes wrap JSON in markdown fences despite our instructions,
    or add a preamble. Try (a) direct parse, (b) first balanced `{...}`,
    (c) inside a ```json``` block.
    """
    text = (text or "").strip()
    if not text:
        return None
    # Direct parse.
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences.
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: first {...} run that parses.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_grader_payload(parsed: dict | None) -> dict:
    """Normalize parsed grader output to a stable schema with defaults.

    Score is clamped to [0, 10]. Lists are coerced from str/None to list.
    Missing keys get sensible defaults so downstream code never crashes.
    """
    if not isinstance(parsed, dict):
        return {
            "score": 0,
            "rigor_issues": ["grader output was unparseable"],
            "missing_steps": [],
            "type_errors": [],
            "forbidden_phrases": [],
            "decomposability_issues": [],
            "final_verdict": "INCORRECT",
            "_parse_ok": False,
        }
    def lst(k: str) -> list[str]:
        v = parsed.get(k, [])
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str) and v.strip():
            return [v]
        return []
    try:
        score = int(parsed.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(10, score))
    verdict = str(parsed.get("final_verdict", "MAJOR_GAPS")).upper()
    if verdict not in {"COMPLETE", "MINOR_GAPS", "MAJOR_GAPS",
                       "NOT_DECOMPOSABLE", "WRONG_STRATEGY", "INCORRECT"}:
        verdict = "MAJOR_GAPS"
    decomp = lst("decomposability_issues")
    # Defensive score cap: if the grader flagged decomposability defects but
    # still scored high, cap at 7 (the rubric says a monolithic crux can't
    # exceed 7). Cheap insurance against a lenient grade slipping through.
    if decomp and score > 7:
        score = 7
    return {
        "score": score,
        "rigor_issues": lst("rigor_issues"),
        "missing_steps": lst("missing_steps"),
        "type_errors": lst("type_errors"),
        "forbidden_phrases": lst("forbidden_phrases"),
        "decomposability_issues": decomp,
        "final_verdict": verdict,
        "_parse_ok": True,
    }


def _format_feedback_block(payload: dict) -> str:
    """Render the grader payload as a readable block for the refiner prompt."""
    out: list[str] = []
    out.append(f"score: {payload.get('score', 0)}/10  "
               f"verdict: {payload.get('final_verdict', 'MAJOR_GAPS')}")
    for k, label in [
        ("rigor_issues", "Rigor issues"),
        ("missing_steps", "Missing steps"),
        ("type_errors", "Type-anchoring errors vs the formal statement"),
        ("forbidden_phrases", "Forbidden phrases used"),
        ("decomposability_issues", "Decomposability defects (split monolithic lemmas, cite cross-references)"),
    ]:
        items = payload.get(k) or []
        if not items:
            continue
        out.append(f"\n{label}:")
        for x in items:
            out.append(f"  - {x}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Per-problem driver
# ---------------------------------------------------------------------------

async def _solve_one_call(
    client: openai.AsyncOpenAI,
    *,
    problem: dict,
    args: argparse.Namespace,
    extra_body: dict,
    tag: str,
    role: str,
    sample_idx: int,
    round_idx: int,
    prev_candidate: Candidate | None = None,
) -> Candidate:
    """One LLM call producing a proof — solver or refiner."""
    pid = _pid_of(problem)
    if role == "solver":
        system_prompt = SOLVER_SYSTEM_PROMPT_DECOMP
        user_prompt = SOLVER_USER_TEMPLATE.format(
            problem_id=pid,
            problem=problem.get("problem", ""),
            formal_statement=problem.get("formal_statement", ""),
        )
    elif role == "refiner":
        assert prev_candidate is not None and prev_candidate.grader_payload is not None
        system_prompt = REFINER_SYSTEM_PROMPT
        user_prompt = REFINER_USER_TEMPLATE.format(
            problem_id=pid,
            problem=problem.get("problem", ""),
            formal_statement=problem.get("formal_statement", ""),
            prev_score=prev_candidate.grader_payload.get("score", 0),
            prev_verdict=prev_candidate.grader_payload.get("final_verdict", "?"),
            prev_proof=prev_candidate.proof,
            grader_feedback_block=_format_feedback_block(prev_candidate.grader_payload),
        )
    else:
        raise ValueError(f"unknown role: {role}")

    call = await call_model(
        client,
        system_prompt=system_prompt, user_prompt=user_prompt,
        model_name=args.model_name, extra_body=extra_body,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
        tag=tag,
    )
    raw = call.get("content") or ""
    proof = (extract_proof(raw) or raw).strip()
    return Candidate(
        proof=proof, raw_response=raw, role=role,
        round_idx=round_idx, sample_idx=sample_idx,
        solver_call=call,
        parent_proof=(prev_candidate.proof if prev_candidate else None),
    )


async def _grade_one_call(
    client: openai.AsyncOpenAI,
    *,
    problem: dict,
    candidate: Candidate,
    args: argparse.Namespace,
    extra_body: dict,
    tag: str,
) -> Candidate:
    """One grader call. Mutates `candidate` with score + grader_payload."""
    pid = _pid_of(problem)
    user_prompt = GRADER_USER_TEMPLATE.format(
        problem_id=pid,
        problem=problem.get("problem", ""),
        formal_statement=problem.get("formal_statement", ""),
        proof=candidate.proof,
    )
    # Grader can run a different (cheaper/faster) model than the
    # solver/refiner — judging proof quality doesn't need the solver's
    # reasoning depth, so when the solver is a slow model (e.g. V4 Pro)
    # the grader stays on Flash. Defaults to the solver model.
    grader_model = getattr(args, "grader_model", None) or args.model_name
    call = await call_model(
        client,
        system_prompt=GRADER_SYSTEM_PROMPT, user_prompt=user_prompt,
        model_name=grader_model, extra_body=extra_body,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
        tag=tag,
    )
    parsed = _parse_grader_json(call.get("content", ""))
    payload = _coerce_grader_payload(parsed)
    candidate.grader_payload = payload
    candidate.score = payload["score"]
    candidate.grader_call = call
    return candidate


async def solve_problem_multipass(
    client: openai.AsyncOpenAI,
    *,
    problem: dict,
    args: argparse.Namespace,
    extra_body: dict,
) -> dict:
    """Run the full multipass for one problem. Returns the `result.json`
    dict (see the module docstring), including a `multipass` field with
    the per-round trace.
    """
    pid = _pid_of(problem)
    t0 = time.monotonic()
    log.info("[%s] multipass start (K=%d, rounds=%d)",
             pid, args.k_solvers, args.max_rounds)

    # Round 0: K parallel solves.
    initial = await asyncio.gather(*[
        _solve_one_call(
            client, problem=problem, args=args, extra_body=extra_body,
            tag=f"{pid}/r0/solve_{i}", role="solver",
            sample_idx=i, round_idx=0,
        )
        for i in range(args.k_solvers)
    ])
    candidates: list[Candidate] = list(initial)
    all_candidates: list[Candidate] = list(initial)

    best_so_far: Candidate | None = None
    early_stopped = False

    for round_idx in range(args.max_rounds + 1):  # +1 = final grade pass
        # Grade every candidate in parallel.
        graded = await asyncio.gather(*[
            _grade_one_call(
                client, problem=problem, candidate=c,
                args=args, extra_body=extra_body,
                tag=f"{pid}/r{round_idx}/grade_{i}",
            )
            for i, c in enumerate(candidates)
        ])

        # Update best.
        for c in graded:
            if best_so_far is None or (c.score or 0) > (best_so_far.score or 0):
                best_so_far = c

        max_score = max((c.score or 0) for c in graded)
        log.info("[%s] round %d: scores=%s  best_so_far=%d",
                 pid, round_idx,
                 sorted([c.score for c in graded], reverse=True),
                 best_so_far.score if best_so_far else 0)

        # Early stop — but require CONSENSUS. A single grade >= the
        # threshold isn't enough to halt: re-grade the top candidate
        # independently (consensus_passes - 1) more times, and only halt
        # if every re-grade also clears the threshold. This guards against
        # a single lenient grade ending the loop prematurely (e.g. a
        # rigorous-but-not-decomposable proof scoring 9).
        if max_score >= args.early_stop_score:
            top_cand = max(graded, key=lambda c: (c.score or 0))
            extra_needed = max(0, args.consensus_passes - 1)
            consensus_scores = [top_cand.score or 0]
            consensus_ok = True
            for j in range(extra_needed):
                regrade = await _grade_one_call(
                    client, problem=problem,
                    candidate=Candidate(
                        proof=top_cand.proof,
                        raw_response=top_cand.raw_response,
                        role=top_cand.role, round_idx=top_cand.round_idx,
                        sample_idx=top_cand.sample_idx,
                    ),
                    args=args, extra_body=extra_body,
                    tag=f"{pid}/r{round_idx}/consensus_{j}",
                )
                all_candidates.append(regrade)  # count its grader cost
                consensus_scores.append(regrade.score or 0)
                if (regrade.score or 0) < args.early_stop_score:
                    consensus_ok = False
                    break
            log.info("[%s] consensus check: scores=%s need %d>=%d -> %s",
                     pid, consensus_scores, args.consensus_passes,
                     args.early_stop_score,
                     "HALT" if consensus_ok else "continue")
            if consensus_ok:
                early_stopped = True
                break
        # No more refinement rounds — we've already done the final grade.
        if round_idx >= args.max_rounds:
            break

        # Refine top-K.
        top = sorted(graded, key=lambda c: (c.score or 0), reverse=True)
        top = top[: args.top_k_refine]
        refined = await asyncio.gather(*[
            _solve_one_call(
                client, problem=problem, args=args, extra_body=extra_body,
                tag=f"{pid}/r{round_idx+1}/refine_{i}", role="refiner",
                sample_idx=i, round_idx=round_idx + 1,
                prev_candidate=top[i],
            )
            for i in range(len(top))
        ])
        candidates = list(refined)
        all_candidates.extend(refined)

    elapsed = time.monotonic() - t0

    # Best proof = whichever candidate scored highest across all rounds.
    best = best_so_far
    # Status: "solved" iff best score >= solve_threshold. The pipeline
    # carries a proof into blueprint generation only when
    # `status == "solved"` (unless --include-failed-nl).
    status = "solved" if (best and (best.score or 0) >= args.solve_threshold) else "failed"

    def _summarize_call(c: Candidate) -> dict:
        return {
            "role": c.role,
            "round_idx": c.round_idx,
            "sample_idx": c.sample_idx,
            "score": c.score,
            "grader_verdict": (c.grader_payload or {}).get("final_verdict"),
            "solver_tokens": {
                "prompt": c.solver_call.get("prompt_tokens", 0),
                "completion": c.solver_call.get("completion_tokens", 0),
                "cached": c.solver_call.get("cached_tokens", 0),
            },
            "solver_cost_usd": c.solver_call.get("cost_usd", 0.0),
            "grader_tokens": {
                "prompt": c.grader_call.get("prompt_tokens", 0),
                "completion": c.grader_call.get("completion_tokens", 0),
                "cached": c.grader_call.get("cached_tokens", 0),
            },
            "grader_cost_usd": c.grader_call.get("cost_usd", 0.0),
        }

    total_cost = sum(
        (c.solver_call.get("cost_usd", 0.0) + c.grader_call.get("cost_usd", 0.0))
        for c in all_candidates
    )
    total_calls = sum(
        (1 if c.solver_call else 0) + (1 if c.grader_call else 0)
        for c in all_candidates
    )

    result = {
        "uuid": pid,
        "problem_id": pid,
        "problem": problem.get("problem", ""),
        "formal_statement": problem.get("formal_statement", ""),
        "status": status,
        "proof": best.proof if best else "",
        "raw_response": best.raw_response if best else "",
        "score": best.score if best else 0,
        "grader_verdict": (best.grader_payload or {}).get("final_verdict") if best else None,
        "best_round": best.round_idx if best else None,
        "early_stopped": early_stopped,
        "rounds_run": (round_idx + 1) if 'round_idx' in dir() else 0,
        "total_calls": total_calls,
        "total_cost_usd": total_cost,
        "elapsed_seconds": elapsed,
        "multipass": {
            "k_solvers": args.k_solvers,
            "max_rounds": args.max_rounds,
            "top_k_refine": args.top_k_refine,
            "early_stop_score": args.early_stop_score,
            "candidates": [_summarize_call(c) for c in all_candidates],
        },
    }
    log.info("[%s] DONE status=%s score=%s calls=%d cost=$%.4f elapsed=%.1fs",
             pid, status, best.score if best else None,
             total_calls, total_cost, elapsed)
    return result


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _pid_of(problem: dict) -> str:
    return problem.get("uuid") or problem.get("problem_id") or problem.get("name") or ""


def _load_problems(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("skipping malformed line: %s", e)
    return out


async def _run_one(
    sem: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    problem: dict,
    args: argparse.Namespace,
    extra_body: dict,
    out_dir: Path,
    trace_fh,
    resume_pids: set[str],
) -> dict | None:
    pid = _pid_of(problem)
    pid_dir = out_dir / pid
    result_path = pid_dir / "result.json"
    if args.resume and pid in resume_pids and result_path.exists():
        log.info("[%s] skip (resume)", pid)
        return None
    async with sem:
        try:
            result = await solve_problem_multipass(
                client, problem=problem, args=args, extra_body=extra_body,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] unrecoverable: %s", pid, e)
            return None
    pid_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(pid_dir / "proof.txt", "w") as f:
        f.write(result.get("proof") or "")
    # Per-round trace also lives in result.json (under "multipass"); a
    # parallel `multipass.json` is convenient for inspection.
    with open(pid_dir / "multipass.json", "w") as f:
        json.dump(result["multipass"], f, ensure_ascii=False, indent=2)
    trace_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    trace_fh.flush()
    return result


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = _get_api_key(args.api_key_env)
    client = openai.AsyncOpenAI(
        base_url=args.model_url, api_key=api_key,
        timeout=httpx.Timeout(args.request_timeout, connect=15.0),
    )
    extra_body = _build_extra_body(args.provider, args.reasoning_effort,
                                   args.model_url)
    log.info("model=%s provider=%s effort=%s K_solvers=%d max_rounds=%d "
             "top_k_refine=%d early_stop_score=%d concurrency=%d",
             args.model_name, args.provider, args.reasoning_effort,
             args.k_solvers, args.max_rounds, args.top_k_refine,
             args.early_stop_score, args.concurrency)

    problems = _load_problems(Path(args.input))
    if args.limit:
        problems = problems[: args.limit]
    log.info("loaded %d problems from %s", len(problems), args.input)

    resume_pids: set[str] = set()
    if args.resume:
        for sub in out_dir.iterdir() if out_dir.exists() else []:
            if sub.is_dir() and (sub / "result.json").exists():
                resume_pids.add(sub.name)
        log.info("resume: skipping %d already-done problems", len(resume_pids))

    traces_path = out_dir / "traces.jsonl"
    trace_fh = open(traces_path, "a")
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()
    tasks = [
        _run_one(sem, client, p, args, extra_body, out_dir, trace_fh, resume_pids)
        for p in problems
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    trace_fh.close()

    # Aggregate summary.
    solved = sum(1 for r in results if r and r.get("status") == "solved")
    completed = sum(1 for r in results if r is not None)
    total_cost = sum((r or {}).get("total_cost_usd", 0.0) for r in results)
    total_calls = sum((r or {}).get("total_calls", 0) for r in results)
    summary = {
        "stage": "nl_proof_multipass",
        "model_name": args.model_name,
        "provider": args.provider,
        "reasoning_effort": args.reasoning_effort,
        "k_solvers": args.k_solvers,
        "max_rounds": args.max_rounds,
        "top_k_refine": args.top_k_refine,
        "early_stop_score": args.early_stop_score,
        "consensus_passes": args.consensus_passes,
        "solve_threshold": args.solve_threshold,
        "total_problems": len(problems),
        "completed": completed,
        "solved": solved,
        "solve_rate": (solved / max(1, completed)),
        "total_calls": total_calls,
        "total_cost_usd": total_cost,
        "elapsed_seconds": time.monotonic() - t0,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("DONE  solved=%d/%d  cost=$%.4f  calls=%d  wall=%.0fs",
             solved, completed, total_cost, total_calls,
             summary["elapsed_seconds"])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # I/O
    p.add_argument("--input", required=True, help="JSONL problems file")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="skip pids that already have result.json")
    # Model
    p.add_argument("--model-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="env var name to read the API key from")
    p.add_argument("--model-name", default="deepseek/deepseek-v4-flash")
    p.add_argument("--grader-model", default=None,
                   help="model for the grader role (default: --model-name). "
                        "Set to a fast model (e.g. deepseek/deepseek-v4-flash) "
                        "when the solver is slow (e.g. v4-pro).")
    p.add_argument("--provider", default="DeepSeek")
    p.add_argument("--reasoning-effort", default="high",
                   choices=["off", "minimal", "low", "medium", "high", "xhigh"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0,
                   help="nucleus sampling top_p (DeepSeek V4 rec: 1.0)")
    p.add_argument("--max-output-tokens", type=int, default=None,
                   help="cap on completion tokens per call; default lets OR pick")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help="per-call HTTP timeout in seconds")
    # Multipass knobs
    p.add_argument("--k-solvers", type=int, default=4,
                   help="initial parallel solver count (round 0)")
    p.add_argument("--max-rounds", type=int, default=3,
                   help="refinement rounds after round 0; final grade pass "
                        "always runs even at max_rounds")
    p.add_argument("--top-k-refine", type=int, default=2,
                   help="top-K candidates by grader score that get refined "
                        "each round")
    p.add_argument("--early-stop-score", type=int, default=9,
                   help="candidate score required to *consider* halting; "
                        "actual halt also requires --consensus-passes "
                        "independent grades at or above this")
    p.add_argument("--consensus-passes", type=int, default=2,
                   help="independent grades >= early-stop-score required "
                        "before halting (guards against a single lenient "
                        "grade)")
    p.add_argument("--solve-threshold", type=int, default=8,
                   help="result.status='solved' iff best score >= this; "
                        "only solved proofs guide blueprint generation")
    # Concurrency
    p.add_argument("--concurrency", type=int, default=8,
                   help="max problems in flight simultaneously")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
