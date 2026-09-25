"""Lean server client: compile a Lean file and turn the result into model feedback.

The pipeline talks to a Lean server that speaks the kimina-lean-server API
(`POST /api/check` with `{"snippets": [{"id", "code"}], "timeout"}`), backed
by a Lean project that provides Mathlib and LeanArchitect. Every call sends
one complete file; the response's `messages` (errors, warnings, infos such as
`#print axioms`) and `sorries` are what the stages read.

Also defines the process-wide client-side timeout budget shared by Lean
checks and LLM streams: a silently hung request is cut after a hard ceiling,
and a process that hits `TIMEOUT_BUDGET` such timeouts exits, since its tools
are evidently unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid as uuid_mod

import httpx

log = logging.getLogger(__name__)

# kimina-lean-server's default local endpoint.
DEFAULT_LEAN_SERVER = "http://localhost:8000/api/check"


# ---------------------------------------------------------------------------
#  Tool-call argument parsing
# ---------------------------------------------------------------------------

def extract_tool_code_arg(raw_arguments) -> str:
    """Robustly extract the ``code`` field from a ``lean_compile`` tool call.

    DeepSeek V4 Flash occasionally double-encodes the tool argument — it
    sends ``{"code": "{\\"code\\": \\"import Mathlib\\\\n...\\"}"}`` instead
    of ``{"code": "import Mathlib\\n..."}``. With the naive
    ``json.loads(...).get("code")`` parse, the still-encoded inner JSON
    blob ends up as the Lean source: Lean then sees a file starting with
    ``{`` (so ``import`` is no longer at the beginning) and containing
    literal ``\\n`` characters instead of real newlines, and emits both
    ``unexpected token '{'`` and ``invalid 'import' command``.

    Accepts the raw ``tool_call.function.arguments`` string (or an already
    parsed dict) and unwraps up to six levels of ``{"code": ...}`` wrapping.
    """
    val = raw_arguments
    for _ in range(6):
        if isinstance(val, str):
            s = val.strip()
            if s.startswith("{") and "\"code\"" in s:
                try:
                    val = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    return val
                continue
            return val
        if isinstance(val, dict):
            val = val.get("code", "")
            continue
        return "" if val is None else str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("code", "") or ""
    return str(val)


# ---------------------------------------------------------------------------
#  Compile feedback formatting (lean_compile tool result)
# ---------------------------------------------------------------------------

def _error_priority(e: dict) -> int:
    """Assign priority to an error (lower = more actionable, show first)."""
    msg = e.get("data", e.get("message", "")).lower()
    if "unknown identifier" in msg or "unknown constant" in msg:
        return 0  # wrong name — very fixable
    if "type mismatch" in msg:
        return 1  # wrong types
    # `unsolved goals` is the open proof state and is the most
    # actionable signal whenever it appears — surface it ahead of
    # cascade-prone errors.
    if "unsolved goals" in msg:
        return 2
    if "failed to synthesize" in msg:
        return 3  # missing instance
    if "rewrite" in msg and "failed" in msg:
        return 4  # wrong rewrite target
    if "tactic" in msg and "failed" in msg:
        return 5  # tactic issue
    if "unexpected token" in msg or "expected" in msg:
        return 6  # syntax — often cascade
    return 4  # default


def _dedup_errors(
    errors: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Deduplicate errors by message text; return (unique, counts)."""
    seen: set[str] = set()
    unique: list[dict] = []
    counts: dict[str, int] = {}
    for e in errors:
        data = e.get("data", e.get("message", str(e)))
        key = data[:200]
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique, counts


def _extract_context_with_span(
    code: str, msg: dict, span_tag: str = "error",
) -> str:
    """Return ±3-line context with <span_tag></span_tag> around the
    msg's pos/endPos. Shared by errors and info messages.
    """
    code_lines = code.split("\n")
    pos = msg.get("pos", {})
    end_pos = msg.get("endPos", pos)
    err_line = pos.get("line", 1)
    err_col = pos.get("column", 0)
    end_line = end_pos.get("line", err_line)
    end_col = end_pos.get("column", err_col + 1)

    el = err_line - 1
    edl = end_line - 1
    ctx_start = max(0, el - 3)
    ctx_end = min(len(code_lines), edl + 3)

    out: list[str] = []
    for ln in range(ctx_start, ctx_end):
        line_text = code_lines[ln] if ln < len(code_lines) else ""
        if el == edl and ln == el:
            before = line_text[:err_col]
            span = line_text[err_col:end_col]
            after = line_text[end_col:]
            out.append(f"{before}<{span_tag}>{span}</{span_tag}>{after}")
        elif ln == el and el != edl:
            out.append(f"{line_text[:err_col]}<{span_tag}>{line_text[err_col:]}")
        elif ln == edl and el != edl:
            out.append(f"{line_text[:end_col]}</{span_tag}>{line_text[end_col:]}")
        else:
            out.append(line_text)
    return "\n".join(out)


def _format_errors_with_context(code: str, errors: list[dict],
                                max_errors: int = 3) -> str:
    """Per-error blocks. Caller supplies the status line; this function
    returns only the body.

    Sort order: source line ascending (root causes first; Lean's
    elaboration cascade is monotone in line number), with `_error_priority`
    as a tiebreaker for errors at the same position. Errors are
    deduplicated by message text and capped at `max_errors` (default 3).
    """
    unique_errors, msg_counts = _dedup_errors(errors)
    unique_errors.sort(key=lambda e: (
        e.get("pos", {}).get("line", 9999),
        _error_priority(e),
    ))
    show_errors = unique_errors[:max_errors]

    parts: list[str] = []
    for i, e in enumerate(show_errors, 1):
        data = e.get("data", e.get("message", str(e)))
        count = msg_counts.get(data[:200], 1)
        count_note = f" (×{count})" if count > 1 else ""
        context_code = _extract_context_with_span(code, e, span_tag="error")
        parts.append(f"Error {i}{count_note}:")
        parts.append("```lean4")
        parts.append(context_code)
        parts.append("```")
        parts.append(f"Error Message: {data}")
        parts.append("")

    omitted = len(unique_errors) - len(show_errors)
    if omitted > 0:
        plural = "s" if omitted != 1 else ""
        parts.append(
            f"({omitted} more distinct error{plural} omitted; the "
            "errors above are the earliest in source order — fix those "
            "first since later errors are usually cascades.)"
        )
        parts.append("")

    return "\n".join(parts)


def _format_infos_with_context(
    code: str, infos: list[dict], per_msg_cap: int = 1500,
) -> str:
    """Render severity=info messages parallel to errors:
    `Info N:` / fenced `<info></info>` snippet / `Info Message: …`.

    Each message's `data` is truncated at `per_msg_cap` chars to avoid a
    single `#print` of a giant term blowing up context.
    """
    parts: list[str] = []
    for i, m in enumerate(infos, 1):
        data = (m.get("data") or "").rstrip()
        if len(data) > per_msg_cap:
            data = data[:per_msg_cap] + "\n…[truncated]"
        context_code = _extract_context_with_span(code, m, span_tag="info")
        parts.append(f"Info {i}:")
        parts.append("```lean4")
        parts.append(context_code)
        parts.append("```")
        parts.append(f"Info Message: {data}")
        parts.append("")
    return "\n".join(parts)


# Diagnostic notes appended to `Lean server error: ...` messages when a
# Lean gateway annotates its per-snippet result with a `kind` field. Kinds
# not listed here render the server error verbatim. The notes describe the
# failure mode so the model can steer off it; they do not prescribe tactics.
_KIND_HINTS = {
    "repl_crash": (
        "The Lean REPL crashed evaluating this snippet (typically OOM "
        "or kernel panic from heavy `decide` / `native_decide` over "
        "large finite search spaces, or `set_option maxHeartbeats 0` / "
        "very large `maxHeartbeats` defeating the cooperative cancel). "
        "Avoid brute-force evaluation of large `Finset.range N` style "
        "goals; prefer structural proofs."
    ),
    "repl_timeout": (
        "The proof exceeded the per-snippet wall-clock timeout. Common "
        "causes: `decide` / `native_decide` over a large search space, "
        "deep `Finset.range N` enumerations, or runaway tactics under "
        "`set_option maxHeartbeats 0`. Try smaller, structural proof "
        "steps instead of brute-force evaluation."
    ),
    "response_lost": (
        "The Lean server lost the response mid-stream (transient infra "
        "hiccup; the snippet did reach a REPL). The previous attempt "
        "may have already succeeded or failed — try a slight reformulation "
        "if this recurs."
    ),
    "worker_error": (
        "The Lean worker reported a non-timeout error during evaluation. "
        "If the message is opaque, simplify the snippet to localize the "
        "failure."
    ),
}


def format_compile_response(
    code: str,
    result: dict,
    include_info_messages: bool = False,
) -> str:
    """Format a lean-server result as the lean_compile tool response string.

    Status-line wording:
        - complete (no errors, no sorry):  `Compilation SUCCESSFUL.`
        - pass with sorry:                 `Compilation SUCCESSFUL with N sorry.`
        - errors:                          `Compilation FAILED with N error[s][ (M distinct)].`
        - server error:                    `Lean server error: ...` (+ kind-keyed note if present)

    No legend, no `Corresponding Code:` label, no HINT footer on
    normal compile errors, no sorry WARNING / INCOMPLETE epilogue (no
    prescriptive scaffolding on real compiler output). Server-error
    paths are annotated with `_KIND_HINTS` because the model has no
    other diagnostic for them.

    When `include_info_messages=True`, append severity=info messages
    (`#check` / `#print` / `#eval` / `logInfo` / traces) parallel to
    errors. Off by default: in submit mode the safeguard strips top-
    level declarations so info commands never reach the compiler.
    """
    analysis = analyse_result(result)

    if analysis["complete"]:
        head = "Compilation SUCCESSFUL."
        if include_info_messages and analysis.get("infos"):
            return head + "\n\n" + _format_infos_with_context(
                code, analysis["infos"])
        return head

    if analysis["system_error"]:
        msg = f"Lean server error: {analysis['system_error']}"
        kind = result.get("kind") if isinstance(result, dict) else None
        hint = _KIND_HINTS.get(kind)
        if hint:
            msg = f"{msg}\n\nNote: {hint}"
        return msg

    parts: list[str] = []

    if analysis["errors"]:
        unique_errors, _ = _dedup_errors(analysis["errors"])
        n_total = len(analysis["errors"])
        n_distinct = len(unique_errors)
        if n_total == 1:
            head = "Compilation FAILED with 1 error."
        elif n_distinct < n_total:
            head = (f"Compilation FAILED with {n_total} errors "
                    f"({n_distinct} distinct).")
        else:
            head = f"Compilation FAILED with {n_total} errors."
        parts.append(head)
        parts.append("")
        parts.append(_format_errors_with_context(code, analysis["errors"]))
    elif analysis["pass_"]:
        # No errors but proof incomplete (contains sorry).
        n_sorry = len(analysis["sorries"])
        parts.append(f"Compilation SUCCESSFUL with {n_sorry} sorry.")

    if include_info_messages and analysis.get("infos"):
        parts.append("")
        parts.append(_format_infos_with_context(code, analysis["infos"]))

    return "\n".join(parts).rstrip()


# ---------------------------------------------------------------------------
#  Lean server HTTP transport
# ---------------------------------------------------------------------------
#
# A per-snippet result may carry `kind` (a precise classification added by
# some multi-worker Lean gateways). When present it is the authoritative
# retry signal: only the kinds below mean the snippet never reached a REPL,
# so retrying cannot run the same proof twice. Without `kind` (e.g. a plain
# kimina-lean-server), the regex below recognises the gateway's transport
# error strings; kimina itself signals overload with HTTP 429/5xx, which are
# retried at the HTTP level.
RETRYABLE_KINDS = {"backend_unreachable", "backend_5xx_pre", "gateway_error"}

# Gateway transport-error strings, used only when a result has no `kind`:
#   1. `Timeout on <ip>:<port>`     -- backend exceeded its timeout
#   2. `Backend HTTP <status>`      -- backend REPL returned non-200
#   3. `Gateway error: <exc>`       -- no backend could be reached
_GATEWAY_TRANSIENT_RE = re.compile(
    r"^("
    r"Timeout on \d+\.\d+\.\d+\.\d+:\d+"
    r"|Backend HTTP \d+"
    r"|Gateway error: .*"
    r")$"
)

# Retry policy: infinite, exponential backoff capped at 10 minutes. Under
# sustained saturation the cap dominates and gives the server's queue real
# recovery time without piling on extra load.
_GATEWAY_RETRY_BACKOFF_S = (10, 30, 60, 180, 600)


def _should_retry_lean_result(result: dict) -> bool:
    """Decide whether a single per-snippet result is retryable.

    `kind` is authoritative when present: only kinds in
    `RETRYABLE_KINDS` (the snippet never reached a REPL) are retried.
    Without `kind`, fall back to the gateway error-string regex. An
    unknown `kind` fails safe to "no retry" — better to surface an
    unexpected response than to spin forever.
    """
    if not isinstance(result, dict):
        return False
    kind = result.get("kind")
    if kind is None:
        err = result.get("error") or ""
        return bool(_GATEWAY_TRANSIENT_RE.search(err))
    return kind in RETRYABLE_KINDS


async def _check_lean_async_raw(
    code: str,
    lean_server_url: str,
    timeout: int = 600,
) -> dict:
    """Send code to the Lean server and return the raw per-snippet result.

    Retries forever on HTTP 429 (server queue full), HTTP 5xx,
    httpx.RequestError, and retryable per-snippet kinds (see
    `RETRYABLE_KINDS`). Any other result is returned as-is — retrying a
    snippet that reached a REPL would risk running the same proof twice.
    """
    snippet_id = str(uuid_mod.uuid4())
    payload = {
        "snippets": [{"id": snippet_id, "code": code}],
        "timeout": timeout,
    }
    attempt = 0
    while True:
        retry_reason = None
        try:
            async with httpx.AsyncClient(timeout=None) as http:
                resp = await http.post(lean_server_url, json=payload)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_reason = f"HTTP {resp.status_code}"
                else:
                    resp.raise_for_status()
                    data = resp.json()
        except httpx.RequestError as e:
            retry_reason = f"{type(e).__name__}: {e}"

        if retry_reason is None:
            results = data.get("results", [])
            last_result = results[0] if results else data
            if not _should_retry_lean_result(last_result):
                return last_result
            retry_reason = (
                last_result.get("kind")
                or last_result.get("error")
                or "retryable"
            )

        wait = _GATEWAY_RETRY_BACKOFF_S[
            min(attempt, len(_GATEWAY_RETRY_BACKOFF_S) - 1)
        ]
        await asyncio.sleep(wait)
        attempt += 1


def analyse_result(result: dict) -> dict:
    """Analyse a lean-server result into a structured summary.

    Returns dict with keys: pass_, complete, errors, warnings, infos,
    sorries, system_error, time.
    """
    if result.get("error"):
        return {
            "pass_": False,
            "complete": False,
            "errors": [{"severity": "error", "data": result["error"]}],
            "warnings": [],
            "sorries": [],
            "system_error": result["error"],
            "time": result.get("time", 0.0),
        }

    response = result.get("response")
    if response is None:
        return {
            "pass_": False,
            "complete": False,
            "errors": [{"severity": "error", "data": "No response from Lean server"}],
            "warnings": [],
            "sorries": [],
            "system_error": "No response from Lean server",
            "time": result.get("time", 0.0),
        }

    if "message" in response:
        return {
            "pass_": False,
            "complete": False,
            "errors": [{"severity": "error", "data": response["message"]}],
            "warnings": [],
            "sorries": [],
            "system_error": response["message"],
            "time": result.get("time", 0.0),
        }

    messages = response.get("messages", []) or []
    errors = [m for m in messages if m.get("severity") == "error"]
    warnings = [m for m in messages if m.get("severity") == "warning"]
    infos = [m for m in messages if m.get("severity") == "info"]
    # Trust `response.sorries` when populated (one entry per sorry token).
    # Older REPL versions only surfaced sorries as `declaration uses 'sorry'`
    # warnings; fall back to mining warnings only when `sorries` is empty,
    # so downstream `target_has_sorry(...)` checks still see something
    # without double-counting (the warning is per-declaration, not per-token).
    sorries = list(response.get("sorries") or [])
    if not sorries:
        sorries.extend(
            m for m in messages
            if m.get("severity") == "warning"
            and (m.get("data") or "").startswith("declaration uses 'sorry'")
        )

    pass_ = len(errors) == 0
    complete = (
        pass_
        and not sorries
        and not any(
            "declaration uses 'sorry'" in w.get("data", "")
            or "failed" in w.get("data", "")
            for w in warnings
        )
    )

    return {
        "pass_": pass_,
        "complete": complete,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "sorries": sorries,
        "system_error": None,
        "time": result.get("time", 0.0),
    }


# ---------------------------------------------------------------------------
#  Client-side wall-clock ceiling + process-wide timeout budget
# ---------------------------------------------------------------------------
#
# The raw client above uses `httpx.AsyncClient(timeout=None)` and passes
# `timeout` to the server only as a hint. The server normally honours it and
# returns "Lean REPL command timed out" as a regular result, but some cases
# defeat that path: `native_decide` (a native binary ignores the cooperative
# cancel), `set_option maxHeartbeats 0`, or a silently dropped connection.
# The wrapper therefore sets a true ceiling at `timeout *
# LEAN_TOOL_CEILING_MULTIPLIER`; on expiry the call site's `except Exception`
# turns it into "Lean server error." feedback and the conversation moves on.
#
# `TIMEOUT_BUDGET` caps how many such client-side timeouts one process
# tolerates (Lean checks and LLM streams share it): if the tools are down,
# every conversation would burn inference producing useless turns, so the
# process fails loudly instead. Rerun with --resume once they are back.
LEAN_TOOL_CEILING_MULTIPLIER = 2.0
TIMEOUT_BUDGET = 100
_timeout_counter = 0


def _on_client_timeout(kind: str, tag: str = "?") -> None:
    """Record one client-side wait_for timeout. Hard-kill if over budget."""
    global _timeout_counter
    _timeout_counter += 1
    log.warning(
        "[%s] client-side TimeoutError (%s): count=%d/%d",
        tag, kind, _timeout_counter, TIMEOUT_BUDGET,
    )
    if _timeout_counter >= TIMEOUT_BUDGET:
        log.error(
            "FATAL: client-side timeout budget %d exhausted — server "
            "appears unresponsive. Killing run to avoid burning "
            "inference on dead tools. Resubmit with --resume once "
            "the tools are healthy again.",
            TIMEOUT_BUDGET,
        )
        # Best-effort flush before hard exit (os._exit skips atexit/cleanup).
        for h in (list(logging.root.handlers)
                  + list(logging.getLogger("goedel_architect").handlers)):
            try:
                h.flush()
            except Exception:
                pass
        os._exit(2)


async def _wait_for_with_count(coro, timeout: float, *,
                                kind: str, tag: str = "?"):
    """asyncio.wait_for that bumps the client-side timeout counter on fire."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        _on_client_timeout(kind, tag)
        raise


async def check_lean_async(code, lean_server_url, timeout):
    """Compile `code` on the Lean server with a hard client-side ceiling."""
    return await _wait_for_with_count(
        _check_lean_async_raw(code, lean_server_url, timeout),
        timeout=timeout * LEAN_TOOL_CEILING_MULTIPLIER,
        kind="lean_check",
    )


# ---------------------------------------------------------------------------
#  Synchronous client
# ---------------------------------------------------------------------------

_SYNC_RETRY_BACKOFF_S = (600,)


def check_lean_sync(code: str, lean_server_url: str, timeout: int) -> dict:
    """Blocking variant of the Lean client (used by the Lean-backed
    dead-node pruner). Retries forever on HTTP 429/5xx, transport errors, and
    gateway transport-error strings, sleeping `_SYNC_RETRY_BACKOFF_S`."""
    payload = {
        "snippets": [{"id": str(uuid_mod.uuid4()), "code": code}],
        "timeout": timeout,
    }
    attempt = 0
    while True:
        retry_reason = None
        try:
            with httpx.Client(timeout=None) as http:
                resp = http.post(lean_server_url, json=payload)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_reason = f"HTTP {resp.status_code}"
                else:
                    resp.raise_for_status()
                    data = resp.json()
        except httpx.RequestError as e:
            retry_reason = f"{type(e).__name__}: {e}"

        if retry_reason is None:
            results = data.get("results", [])
            last_result = results[0] if results else data
            err = last_result.get("error") if isinstance(last_result, dict) else None
            if not (err and _GATEWAY_TRANSIENT_RE.match(err)):
                return last_result
            retry_reason = err

        wait = _SYNC_RETRY_BACKOFF_S[min(attempt, len(_SYNC_RETRY_BACKOFF_S) - 1)]
        time.sleep(wait)
        attempt += 1
