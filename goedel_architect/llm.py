"""LLM client layer shared by the blueprint, prover, and refinement stages.

The stages talk to an OpenAI-compatible chat-completions endpoint (OpenRouter
by default) with native tool calling:

  * streaming chat completions, assembled into a response object with
    content, reasoning, tool calls, and usage (prompt / completion / cached
    tokens and billed cost);
  * retry on transient failures: HTTP 429 waits (honouring
    `retry_after_seconds`) without limit, other transients (5xx, dropped
    connections, empty streams) back off exponentially up to `max_retries`;
  * prompt-token bookkeeping with a Hugging Face tokenizer when available
    (character estimate otherwise), used to size each request's
    `max_tokens` within the model's context window.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import APIConnectionError, APITimeoutError

from .lean import _wait_for_with_count

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Token counting (HF tokenizer if available, else char-based fallback)
# ---------------------------------------------------------------------------

DEFAULT_MAX_MODEL_LEN = 262144  # 256K
CONTEXT_SAFETY_OFFSET = 512


class CharFallbackTokenizer:
    """Tokenizer-like shim used when no HF tokenizer is available.

    Estimates ~1 token per 4 characters. Used only for prompt-token
    bookkeeping, not for actual model encoding."""
    vocab_size = 0

    def encode(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=True):
        parts: list[str] = []
        if tools:
            parts.append("<|tools|>" + json.dumps(tools, ensure_ascii=False))
        for m in messages:
            role = m.get("role", "")
            content = m.get("content") or ""
            parts.append(f"<|{role}|>\n{content}\n")
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                parts.append(f"<tool_call>{fn.get('name','')}({args})</tool_call>\n")
            if m.get("tool_call_id"):
                parts.append(f"<tool_call_id>{m['tool_call_id']}</tool_call_id>\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)


def load_tokenizer(model_path: str | None):
    if not model_path:
        log.info("No --tokenizer-path provided, using character-based fallback")
        return CharFallbackTokenizer()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        log.info("Loaded HF tokenizer from %s (vocab_size=%d)",
                 model_path, tok.vocab_size)
        return tok
    except Exception as e:
        log.warning("Could not load tokenizer from %s (%s); using char fallback",
                    model_path, e)
        return CharFallbackTokenizer()


def _prepare_messages_for_template(messages: list[dict]) -> list[dict]:
    def _parse_args(raw):
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"_raw_arguments": raw}

    out = []
    for msg in messages:
        if "tool_calls" not in msg:
            out.append(msg)
            continue
        msg = {**msg}
        msg["tool_calls"] = [
            {
                **tc,
                "function": {
                    **tc["function"],
                    "arguments": _parse_args(tc["function"]["arguments"]),
                },
            }
            for tc in msg["tool_calls"]
        ]
        out.append(msg)
    return out


def count_prompt_tokens(tokenizer, messages: list[dict],
                        tools_overhead: int = 0) -> int:
    try:
        template_msgs = _prepare_messages_for_template(messages)
        text = tokenizer.apply_chat_template(
            template_msgs, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        text = json.dumps(messages, ensure_ascii=False, default=str)
    return len(tokenizer.encode(text)) + tools_overhead


def compute_max_tokens(prompt_tokens: int,
                       max_model_len: int = DEFAULT_MAX_MODEL_LEN,
                       max_output_tokens: int | None = None) -> int:
    available = max_model_len - prompt_tokens - CONTEXT_SAFETY_OFFSET
    if available < 512:
        return 0
    if max_output_tokens is None:
        return available
    return min(max_output_tokens, available)


def estimate_tools_overhead(tokenizer, tools: list[dict]) -> int:
    test_msgs = [{"role": "user", "content": "hello"}]
    try:
        with_tools = tokenizer.apply_chat_template(
            test_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        without_tools = tokenizer.apply_chat_template(
            test_msgs, tokenize=False, add_generation_prompt=True,
        )
        overhead = len(tokenizer.encode(with_tools)) - len(tokenizer.encode(without_tools))
        log.info("Measured tools token overhead: %d", overhead)
        return max(overhead, 0)
    except Exception as e:
        log.warning("Could not measure tools overhead (%s), using estimate=800", e)
        return 800


# ---------------------------------------------------------------------------
#  Streaming chat-completion + retry
# ---------------------------------------------------------------------------

def _extract_retry_after(exc: Exception) -> float | None:
    try:
        body = exc.response.json()
        meta = body.get("error", {}).get("metadata", {})
        ra = meta.get("retry_after_seconds")
        if ra is not None:
            return float(ra)
    except Exception:
        pass
    return None


def _http_status(e: Exception) -> int | None:
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    return status


def _is_transient(e: Exception) -> bool:
    if isinstance(e, json.JSONDecodeError):
        return True
    if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout,
                      httpx.ReadError, httpx.RemoteProtocolError,
                      httpx.WriteError, httpx.WriteTimeout,
                      httpx.PoolTimeout, httpx.NetworkError,
                      httpx.ConnectError, asyncio.TimeoutError)):
        return True
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(e, RuntimeError) and "Stream ended with no content" in str(e):
        return True
    status = _http_status(e)
    if status in (408, 425, 429):
        return True
    if status is not None and 500 <= status < 600:
        return True
    return False


async def call_chat_streaming(client, **kwargs):
    """Call chat.completions.create with stream=True and assemble the result."""
    kwargs = dict(kwargs)
    kwargs["stream"] = True
    kwargs.setdefault("stream_options", {"include_usage": True})

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    cost_usd: float = 0.0
    saw_content = False

    stream = await client.chat.completions.create(**kwargs)
    try:
        async for chunk in stream:
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
                saw_content = True
            for attr in ("reasoning", "reasoning_content"):
                r = getattr(delta, attr, None)
                if r:
                    reasoning_parts.append(r)
                    saw_content = True
                    break
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tool_calls.setdefault(idx, {
                        "id": None, "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if tc.id:
                        slot["id"] = tc.id
                    if getattr(tc, "type", None):
                        slot["type"] = tc.type
                    # Gemini (OpenAI-compat) attaches a thought_signature
                    # under extra_content; it MUST be echoed back on the
                    # assistant message or the next request 400s.
                    ec = getattr(tc, "extra_content", None)
                    if ec:
                        slot["extra_content"] = ec
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments
                        saw_content = True
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    if not saw_content and finish_reason is None:
        raise RuntimeError("Stream ended with no content and no finish_reason")

    tc_list = []
    for idx in sorted(tool_calls.keys()):
        s = tool_calls[idx]
        tc_list.append(SimpleNamespace(
            id=s["id"],
            type=s["type"],
            function=SimpleNamespace(
                name=s["function"]["name"],
                arguments=s["function"]["arguments"],
            ),
            extra_content=s.get("extra_content"),
        ))

    msg = SimpleNamespace(
        content="".join(content_parts) or None,
        reasoning="".join(reasoning_parts) or None,
        tool_calls=tc_list if tc_list else None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


RATE_LIMIT_BASE_WAIT = 1.0


async def chat_with_retry(client, *, max_retries: int = 12, tag: str,
                          hard_timeout: float | None = None, **kwargs):
    """Call `call_chat_streaming` with retry on transient upstream failures.

    `max_retries` bounds the per-turn retry budget for non-429 transients
    (5xx, transport drops, empty stream, JSONDecodeError). 429s are
    retried forever above. With backoff 1->2->4->8->16->30 (cap), 12
    attempts give ~5 min of wall time before the conversation gives up.

    `hard_timeout` (seconds), when set, is a wall-clock ceiling on a single
    streaming call; hitting it counts toward the process-wide timeout
    budget (see `lean.TIMEOUT_BUDGET`) and is retried as a transient. The
    prover sets it; blueprint generation and refinement, whose long
    outputs can legitimately stream for many minutes, leave it off."""
    delay = 1.0
    bounded_attempts = 0
    rl_attempts = 0
    while True:
        t0 = time.monotonic()
        try:
            if hard_timeout is None:
                return await call_chat_streaming(client, **kwargs)
            # Hard ceiling on a single streaming call. asyncio.wait_for
            # is the only timer in this stack that doesn't depend on
            # httpx's byte-level liveness check (which fails to fire
            # under HTTP/2 multiplexing or partial-frame trickle), so
            # this is what actually rescues a silently-hung stream.
            # On timeout: TimeoutError → counted in the global budget,
            # _is_transient → retry on a fresh request.
            return await _wait_for_with_count(
                call_chat_streaming(client, **kwargs),
                timeout=hard_timeout,
                kind="stream",
                tag=tag,
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            if _http_status(e) == 429:
                rl_attempts += 1
                wait = max(_extract_retry_after(e) or 0.0, RATE_LIMIT_BASE_WAIT)
                if rl_attempts <= 3 or rl_attempts % 12 == 0:
                    log.warning(
                        "[%s] 429 rate-limited after %.1fs — waiting %.1fs "
                        "(attempt #%d, no cap): %s",
                        tag, elapsed, wait, rl_attempts, str(e)[:120],
                    )
                await asyncio.sleep(wait)
                continue
            if not _is_transient(e) or bounded_attempts >= max_retries:
                raise
            bounded_attempts += 1
            wait = _extract_retry_after(e) or delay
            wait = max(wait, 0.5)
            # Surface the underlying httpx exception so the wall-time and
            # exception kind together reveal what stalled: ConnectTimeout
            # (~15s, edge slow to handshake), ReadTimeout (~600s, stream
            # stall), PoolTimeout (client-side queue), ConnectError (TCP
            # refused/reset, fast), RemoteProtocolError (closed mid-stream),
            # JSONDecodeError (truncated SSE chunk), etc.
            cause = e.__cause__ or e.__context__
            cause_name = type(cause).__name__ if cause is not None else "—"
            status = _http_status(e)
            status_str = f" status={status}" if status is not None else ""
            log.warning(
                "[%s] %s(%s) after %.1fs%s: %s — retrying in %.1fs "
                "(attempt %d/%d)",
                tag, type(e).__name__, cause_name, elapsed, status_str,
                str(e)[:160], wait, bounded_attempts, max_retries,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30.0)


# ---------------------------------------------------------------------------
#  Model / provider options shared by the stage CLIs
# ---------------------------------------------------------------------------

def add_model_args(parser, *, x_title: str) -> None:
    """Add the model, endpoint, provider, and reasoning options."""
    parser.add_argument("--model-url", default="https://openrouter.ai/api/v1",
                        help="OpenRouter base URL (or any OpenAI-compatible endpoint)")
    parser.add_argument("--model-name", default="deepseek/deepseek-v4-flash",
                        help="Model slug on the endpoint")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                        help="Env var name to read the API key from")
    parser.add_argument("--http-referer", default="",
                        help="Optional HTTP-Referer header for OpenRouter attribution")
    parser.add_argument("--x-title", default=x_title,
                        help="Optional X-Title header for OpenRouter attribution")
    parser.add_argument("--provider", default="DeepSeek",
                        help="OpenRouter provider name to pin (e.g. 'DeepSeek' "
                             "for the official first-party endpoint). Empty "
                             "string disables pinning.")
    parser.add_argument("--allow-fallbacks", action="store_true",
                        help="Allow OpenRouter to fall back to other providers "
                             "if the pinned one is unavailable")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["off", "low", "medium", "high"],
                        help="Reasoning effort for the request (OpenRouter "
                             "`reasoning.effort`; DeepSeek-V4 maps it to its "
                             "thinking mode). Use 'off' to disable.")
    parser.add_argument("--reasoning-max-tokens", type=int, default=None,
                        help="Optional cap on reasoning tokens "
                             "(passes reasoning.max_tokens).")


def read_api_key(args) -> str:
    """Return the API key from `--api-key-env`, or exit with an error."""
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        log.error("API key env var %s is empty or unset. Set it before running.",
                  args.api_key_env)
        sys.exit(1)
    return api_key


def default_headers(args) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        headers["X-Title"] = args.x_title
    return headers or None


def build_extra_body(args) -> dict:
    """Request fields beyond the OpenAI schema.

    On OpenRouter: surface cached-token and billed-cost fields in `usage`,
    pin the provider, and set the reasoning effort. Other OpenAI-compatible
    endpoints reject unknown body fields, so they only get the standard
    `reasoning_effort` parameter.
    """
    if "openrouter.ai" not in (args.model_url or ""):
        extra_body: dict = {}
        if args.reasoning_effort != "off":
            extra_body["reasoning_effort"] = args.reasoning_effort
        log.info("Non-OpenRouter endpoint: extra_body=%s", extra_body)
        return extra_body

    extra_body = {
        # Surface cache-hit and billed-cost fields in usage. Without this,
        # usage.prompt_tokens_details.cached_tokens and usage.cost are
        # not populated by OpenRouter.
        "usage": {"include": True},
    }
    if args.provider:
        extra_body["provider"] = {
            "order": [args.provider],
            "allow_fallbacks": bool(args.allow_fallbacks),
        }
        log.info("Pinning provider: %s (allow_fallbacks=%s)",
                 args.provider, bool(args.allow_fallbacks))
    if args.reasoning_effort != "off" or args.reasoning_max_tokens is not None:
        reasoning_cfg: dict = {}
        if args.reasoning_effort != "off":
            reasoning_cfg["effort"] = args.reasoning_effort
        if args.reasoning_max_tokens is not None:
            reasoning_cfg["max_tokens"] = args.reasoning_max_tokens
        extra_body["reasoning"] = reasoning_cfg
        log.info("Reasoning: %s", reasoning_cfg)
    else:
        log.info("Reasoning: disabled")
    return extra_body


# ---------------------------------------------------------------------------
#  Logging: file (DEBUG) + stderr (INFO), both flushed per record
# ---------------------------------------------------------------------------

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(out_dir: Path) -> Path:
    """Send every `goedel_architect.*` logger to `<out_dir>/session.log`
    (DEBUG) and stderr (INFO)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "session.log"

    root = logging.getLogger("goedel_architect")
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = FlushFileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = FlushStreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    return log_path
