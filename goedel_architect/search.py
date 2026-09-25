"""Mathlib search: the `mathlib_search` tool offered to the model.

Two backends, chosen by URL:

  * LeanSearch (https://leansearch.net, the default): the public semantic
    search engine over Mathlib. Request `{"query": [q], "num_results": k}`.
  * A self-hosted retrieval gateway speaking `{"query", "top_k"}` ->
    `{"records": [...]}` (e.g. a LeanSearch deployment behind a small
    gateway); any URL that is not leansearch.net is treated this way.

Search is optional. `resolve_search_server` probes the endpoint once at
startup; if it is unreachable (or the URL is "none"), the stage runs without
the tool and its prompts drop the search guidance. A search that fails
mid-run returns empty results or an error string to the model and never
counts toward the process-wide timeout budget.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

DEFAULT_SEARCH_SERVER = "https://leansearch.net/search"
_DISABLED_VALUES = {"", "none", "off", "false", "0"}


MATHLIB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "mathlib_search",
        "description": (
            "A lookup helper for specific Mathlib lemmas you need while "
            "executing your proof plan. Mathlib does NOT contain the "
            "solution to your problem directly — do not use this tool to "
            "'find the proof' or to search for an exact bound stated in "
            "the goal; such queries return nothing useful and waste turns. "
            "Use it along the way: when your plan needs a specific lemma "
            "(by name, signature, or hypothesis pattern), or to recover "
            "the correct name after an 'Unknown constant' / 'Unknown "
            "identifier' error. Returns theorem names (usable in tactics "
            "like `exact`, `apply`, `simp [...]`), their type signatures, "
            "and natural-language descriptions, ranked by relevance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language description of the theorem or "
                        "mathematical concept to search for. Be specific "
                        "about the mathematical property, e.g. "
                        "'triangle inequality for absolute value', "
                        "'monotonicity of natural number addition', "
                        "'Cauchy-Schwarz inequality', "
                        "'if p is prime and p divides a*b then p divides a or p divides b'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Number of results to return. Default: 5. Use more only if needed."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
#  Search feedback formatting (mathlib_search tool result)
# ---------------------------------------------------------------------------

def format_search_response(result: dict) -> str:
    """Format lean-search results as the mathlib_search tool response string."""
    records = result.get("records", [])
    if not records:
        return "No matching theorems found in Mathlib for this query."

    parts = [f"Found {len(records)} relevant Mathlib result(s):\n"]
    for rec in records:
        name = rec.get("name_pp", "unknown")
        signature = rec.get("signature", "").strip()
        kind = rec.get("kind", "theorem")
        module = rec.get("module_name_pp", "")
        description = rec.get("informal_description", "")

        parts.append(f"{name}: {kind} {name} {signature}")
        if module:
            parts.append(f"  Module: {module}")
        if description:
            parts.append(f"  {description}")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  HTTP transport
# ---------------------------------------------------------------------------
#
# Tighter backoff than the Lean client: search calls are sub-second, so a
# transient blip should not stall a turn for minutes. The schedule sticks at
# the last value; after `_SEARCH_MAX_ATTEMPTS` attempts (or immediately on a
# refused connection) the call degrades to "no results".
_SEARCH_RETRY_BACKOFF_S = (5, 15, 60)
_SEARCH_MAX_ATTEMPTS = 4


async def _search_gateway(
    query: str,
    search_server_url: str,
    top_k: int = 5,
    timeout: int = 60,
) -> dict:
    """Query a self-hosted retrieval gateway (`{"query", "top_k"}` protocol).

    Returns the raw JSON response with 'records' and 'count' keys. Retries
    transport errors (httpx.RequestError) and HTTP 5xx using
    `_SEARCH_RETRY_BACKOFF_S`, up to `_SEARCH_MAX_ATTEMPTS` attempts.
    """
    payload = {
        "query": query,
        "top_k": max(1, min(top_k, 20)),
    }
    attempt = 0
    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                resp = await http.post(search_server_url, json=payload)
                if not 500 <= resp.status_code < 600:  # 5xx: retry below
                    resp.raise_for_status()
                    return resp.json()
        except httpx.ConnectError:
            # Nothing listening. Retrying can't help and the blueprint /
            # refinement call sites use this function without a ceiling, so
            # degrade to "no results" and let the conversation continue.
            return {"records": [], "count": 0}
        except httpx.RequestError:
            pass

        if attempt >= _SEARCH_MAX_ATTEMPTS - 1:
            return {"records": [], "count": 0}
        wait = _SEARCH_RETRY_BACKOFF_S[
            min(attempt, len(_SEARCH_RETRY_BACKOFF_S) - 1)
        ]
        await asyncio.sleep(wait)
        attempt += 1


def _is_leansearch(search_server_url: str) -> bool:
    return (urlparse(search_server_url).hostname or "").endswith("leansearch.net")


def _leansearch_record(item) -> dict:
    """Map one leansearch.net result onto the gateway record shape that
    `format_search_response` reads (`name_pp`, `module_name_pp`, `kind`,
    `signature`, `informal_description`)."""
    rec = item.get("result", item) if isinstance(item, dict) else {}
    name = rec.get("name") or []
    module = rec.get("module_name") or []
    return {
        "name_pp": ".".join(map(str, name)) if isinstance(name, list) else str(name),
        "module_name_pp": ".".join(map(str, module)) if isinstance(module, list) else str(module),
        "kind": rec.get("kind") or "theorem",
        "signature": rec.get("signature") or "",
        "informal_name": rec.get("informal_name") or "",
        "informal_description": rec.get("informal_description") or "",
    }


# The same query recurs across lemmas and samples; results are deterministic,
# so cache them per process to spare the public service.
_LEANSEARCH_CACHE: dict[tuple[str, int], dict] = {}
_LEANSEARCH_CACHE_MAX = 4096


async def _search_leansearch(
    query: str,
    search_server_url: str,
    top_k: int = 5,
    timeout: int = 60,
) -> dict:
    """Query leansearch.net and return `{"records": [...], "count": n}`.

    Same retry schedule as the gateway client; HTTP 429 (rate limit) is
    retried like a 5xx.
    """
    top_k = max(1, min(top_k, 20))
    key = (query, top_k)
    if key in _LEANSEARCH_CACHE:
        return _LEANSEARCH_CACHE[key]
    payload = {"query": [query], "num_results": top_k}
    attempt = 0
    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                resp = await http.post(search_server_url, json=payload)
                if resp.status_code != 429 and not 500 <= resp.status_code < 600:
                    resp.raise_for_status()
                    data = resp.json()
                    hits = data[0] if isinstance(data, list) and data else []
                    records = [_leansearch_record(h) for h in hits]
                    result = {"records": records, "count": len(records)}
                    if len(_LEANSEARCH_CACHE) < _LEANSEARCH_CACHE_MAX:
                        _LEANSEARCH_CACHE[key] = result
                    return result
        except httpx.ConnectError:
            return {"records": [], "count": 0}
        except httpx.RequestError:
            pass

        if attempt >= _SEARCH_MAX_ATTEMPTS - 1:
            return {"records": [], "count": 0}
        wait = _SEARCH_RETRY_BACKOFF_S[
            min(attempt, len(_SEARCH_RETRY_BACKOFF_S) - 1)
        ]
        await asyncio.sleep(wait)
        attempt += 1


async def search_mathlib_async(
    query: str,
    search_server_url: str,
    top_k: int = 5,
    timeout: int = 60,
    ceiling: float | None = None,
) -> dict:
    """Run one `mathlib_search` query against the configured backend.

    With `ceiling`, the whole call (retries included) is cut after that many
    seconds and raises `asyncio.TimeoutError`; call sites report it to the
    model as a search error. Search timeouts never count toward the Lean /
    LLM timeout budget.
    """
    backend = _search_leansearch if _is_leansearch(search_server_url) else _search_gateway
    coro = backend(query, search_server_url, top_k, timeout=timeout)
    if ceiling is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=ceiling)


def resolve_search_server(url: str | None, *, timeout: float = 30.0) -> str | None:
    """Return `url` if the search endpoint answers a probe query, else None.

    None / "" / "none" / "off" disable search explicitly. A failed probe
    logs a warning and disables search for this run, so the stage proceeds
    without the `mathlib_search` tool instead of feeding the model errors.
    """
    if url is None or url.strip().lower() in _DISABLED_VALUES:
        log.info("Mathlib search: disabled")
        return None
    probe = "sum of two even numbers is even"
    try:
        if _is_leansearch(url):
            resp = httpx.post(url, json={"query": [probe], "num_results": 1},
                              timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            ok = isinstance(data, list)
        else:
            resp = httpx.post(url, json={"query": probe, "top_k": 1},
                              timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            ok = isinstance(data, dict) and "records" in data
        if not ok:
            raise ValueError(f"unexpected response shape: {str(data)[:120]}")
    except Exception as e:
        log.warning(
            "Mathlib search at %s is unavailable (%s: %s) — running WITHOUT "
            "the mathlib_search tool.", url, type(e).__name__, e,
        )
        return None
    log.info("Mathlib search: %s", url)
    return url
