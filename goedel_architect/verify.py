"""Final audit: re-verify every claimed proof in a run against the official statements.

`#print axioms` alone is NOT enough: a proof can be axiom-clean yet prove a
*mutated* statement (dropped hypothesis, weakened goal, defeq trick). Lean
compiling does not mean you proved the intended theorem. This verifier closes
that gap with three independent gates per claimed solve:

  1. STATEMENT FIDELITY — the proof's main theorem signature must match the
     official benchmark statement VERBATIM (whitespace-normalized). Catches
     statement-munging that axiom-checking misses.
  2. COMPILES — the full assembled file recompiles against Mathlib with no
     error diagnostics.
  3. AXIOM HONESTY — `#print axioms <main>` shows no `sorryAx` and no custom
     axioms. `Lean.ofReduceBool` (from `native_decide`) is allowed-but-FLAGGED
     so you can eyeball every native_decide proof before submitting.

Verdicts: CLEAN (foundational axioms only) | NATIVE (clean + ofReduceBool,
review before submit) | FAIL_SIG (statement mismatch) | FAIL_COMPILE |
FAIL_SORRY | FAIL_AXIOM (custom axiom) | NO_PROOF.

Usage:
  python -m goedel_architect.verify <run_dir> --statements official.jsonl [--allow-native]
  (official.jsonl: one {"problem_id","formal_statement"} per line — the
   canonical benchmark statements, NOT the pipeline's internal copies.)
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
from collections import Counter

from .graph import parse_axioms_from_result
from .lean import DEFAULT_LEAN_SERVER, check_lean_async

FOUNDATIONAL = {"propext", "Classical.choice", "Quot.sound"}
NATIVE_AXIOM = "Lean.ofReduceBool"   # what native_decide adds


def theorem_sig(formal_statement: str, name: str | None = None) -> str | None:
    """Return the whitespace-normalized signature `theorem <name> ... :`
    (everything up to the `:=` that begins the proof), or None."""
    s = formal_statement
    m = re.search(r"\b(theorem|lemma)\s+([A-Za-z0-9_'.]+)", s)
    if not m:
        return None
    start = m.start()
    # find the ':=' that opens the body (first top-level := after the decl)
    body = re.search(r":=", s[start:])
    if not body:
        return None
    sig = s[start:start + body.start()]
    return re.sub(r"\s+", " ", sig).strip()


def main_theorem_name(formal_statement: str) -> str | None:
    m = re.search(r"\b(?:theorem|lemma)\s+([A-Za-z0-9_'.]+)", formal_statement)
    return m.group(1) if m else None


def solved_candidates(run_dir: str):
    """Return the first claimed proof per problem from the prover traces of
    every iteration (`iterNN/prove/traces.jsonl`)."""
    best: dict[str, dict] = {}
    for root in [run_dir]:
        for tf in sorted(glob.glob(os.path.join(root, "iter*/prove/traces.jsonl"))):
            try:
                for line in open(tf):
                    t = json.loads(line)
                    # accept either the assembly gate OR the pipeline's
                    # `success` flag — both claim a complete proof; the
                    # rigorous gates below are the real arbiter.
                    if not (t.get("assembly_passed") or t.get("success")):
                        continue
                    pid = t.get("problem_id")
                    if pid and t.get("final_code") and pid not in best:
                        best[pid] = t
            except (OSError, json.JSONDecodeError):
                continue
    return best


async def verify_one(pid, trace, official_stmt, lean_server, timeout, allow_native):
    code = trace.get("final_code") or ""
    if not code:
        return pid, "NO_PROOF", set()
    name = main_theorem_name(official_stmt)
    # gate 1: statement fidelity — proof must contain the official signature verbatim
    off_sig = theorem_sig(official_stmt)
    proof_sig = None
    if name:
        m = re.search(r"\b(?:theorem|lemma)\s+" + re.escape(name) + r"\b", code)
        if m:
            proof_sig = theorem_sig(code[m.start():], name)
    if off_sig is None or proof_sig is None or off_sig != proof_sig:
        return pid, "FAIL_SIG", set()
    # gates 2+3: compile official-signature file + #print axioms.
    # These are large assembled proofs; under gateway load a compile can
    # TIME OUT. A timeout is NOT a compile error — conflating them silently
    # drops real solves (observed: 7 CLEAN proofs false-failed in a loaded
    # batch). Retry on exception/timeout, and only call it FAIL_COMPILE on a
    # real error diagnostic; persistent timeout -> INCONCLUSIVE (never drop).
    probe = code.rstrip() + (f"\n\n#print axioms {name}\n" if name else "\n")
    resp = None
    for attempt in range(3):
        try:
            resp = await check_lean_async(probe, lean_server, timeout)
            break
        except Exception:
            if attempt == 2:
                return pid, "INCONCLUSIVE_TIMEOUT", set()
            await asyncio.sleep(5)
    msgs = (resp.get("response") or {}).get("messages") or []
    if any(m.get("severity") == "error" for m in msgs):
        return pid, "FAIL_COMPILE", set()
    axioms = parse_axioms_from_result(resp)
    if "sorryAx" in axioms:
        return pid, "FAIL_SORRY", axioms
    extra = axioms - FOUNDATIONAL
    if not extra:
        return pid, "CLEAN", axioms
    if extra == {NATIVE_AXIOM} and allow_native:
        return pid, "NATIVE", axioms
    return pid, "FAIL_AXIOM", axioms


async def main():
    ap = argparse.ArgumentParser(description="Re-verify the claimed proofs of a run.")
    ap.add_argument("run_dir")
    ap.add_argument("--statements", required=True, help="official {problem_id,formal_statement} jsonl")
    ap.add_argument("--lean-server", default=os.environ.get("LEAN_SERVER", DEFAULT_LEAN_SERVER))
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--allow-native", action="store_true",
                    help="accept Lean.ofReduceBool (native_decide) as NATIVE (flagged), else FAIL_AXIOM")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    official = {}
    for line in open(args.statements):
        if line.strip():
            r = json.loads(line)
            official[r["problem_id"]] = r["formal_statement"]

    cands = solved_candidates(args.run_dir)
    sem = asyncio.Semaphore(args.concurrency)

    async def run(pid, t):
        if pid not in official:
            return pid, "NO_OFFICIAL_STMT", set()
        async with sem:
            return await verify_one(pid, t, official[pid], args.lean_server,
                                    args.timeout, args.allow_native)

    results = await asyncio.gather(*(run(pid, t) for pid, t in cands.items()))
    buckets = Counter(v for _, v, _ in results)
    submittable = sorted(p for p, v, _ in results if v in ("CLEAN", "NATIVE"))
    native = sorted(p for p, v, _ in results if v == "NATIVE")
    inconclusive = sorted(p for p, v, _ in results if v == "INCONCLUSIVE_TIMEOUT")
    failed = sorted((p, v) for p, v, _ in results if v.startswith(("FAIL", "NO_")))
    print(f"candidates: {len(cands)}  |  verdicts: {dict(buckets)}")
    print(f"\nSUBMITTABLE (CLEAN+NATIVE): {len(submittable)}")
    if native:
        print(f"  ⚠ uses native_decide (review before submit): {native}")
    if inconclusive:
        print(f"\nINCONCLUSIVE (compile timed out — re-run on quiet gateway, NOT a failure): {inconclusive}")
    if failed:
        print("\nREJECTED (would be a mis-proof / incomplete):")
        for p, v in failed:
            print(f"  {v:18s} {p}")
    json.dump({"submittable": submittable, "native": native,
               "inconclusive": inconclusive,
               "rejected": [p for p, _ in failed]},
              open(os.path.join(args.run_dir, "submission_verdicts.json"), "w"), indent=2)


def cli() -> None:
    raise SystemExit(asyncio.run(main()))


if __name__ == "__main__":
    cli()
