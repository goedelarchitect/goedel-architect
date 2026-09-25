"""Goedel-Architect: generate a blueprint, then alternate theorem proving and refinement.

    problems ─(optional NL proof)─► blueprint generation ─► prove (iter00)
                                                               │ unproved lemmas?
                            ┌──────── refine (iter01) ◄────────┘
                            └──► prove (iter01) ──► refine (iter02) ──► ...

Each stage runs as its own process (`python -m goedel_architect.<stage>`) and
hands off to the next through `traces.jsonl` files under the run directory:

    <run>/nl_proof/        natural-language proofs            (only with --nl-proof)
    <run>/blueprint/       blueprint generation               (blueprint.py)
    <run>/iter00/prove/    theorem proving on the blueprint   (prover.py)
    <run>/iterNN/refine/   refinement of iter(NN-1)'s failures (refine.py)
    <run>/iterNN/prove/    theorem proving on the refined blueprints

Usage:
    python -m goedel_architect examples/problems.jsonl --run-dir runs/demo
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx

from .lean import DEFAULT_LEAN_SERVER
from .search import DEFAULT_SEARCH_SERVER, resolve_search_server

# Options that a resumed run takes from its frozen `run_config.json`
# instead of the command line.
_NOT_FROZEN = {"problems", "run_dir", "resume"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m goedel_architect",
        description="Prove the formal statements in a JSONL file with "
                    "Goedel-Architect (blueprint generation, then "
                    "alternating theorem proving and blueprint refinement).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("problems", help="JSONL with `problem_id` and `formal_statement` "
                                    "(`problem`, the informal statement, is optional)")
    p.add_argument("--run-dir", default=None,
                   help="output directory (default: runs/<model>_<timestamp>)")
    p.add_argument("--resume", action="store_true",
                   help="resume an interrupted run in --run-dir (reuses its "
                        "frozen run_config.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N problems")

    g = p.add_argument_group("model (any model on OpenRouter, or any OpenAI-compatible endpoint)")
    g.add_argument("--model", default="deepseek/deepseek-v4-flash")
    g.add_argument("--model-url", default="https://openrouter.ai/api/v1")
    g.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="environment variable holding the API key")
    g.add_argument("--provider", default="DeepSeek",
                   help="OpenRouter provider to pin ('' = let OpenRouter route)")
    g.add_argument("--reasoning-effort", default="high",
                   choices=["off", "low", "medium", "high"])
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V4-Pro",
                   help="Hugging Face tokenizer for prompt-token counting "
                        "('' = 4-characters-per-token estimate)")

    g = p.add_argument_group("tools")
    g.add_argument("--lean-server", default=os.environ.get("LEAN_SERVER", DEFAULT_LEAN_SERVER),
                   help="kimina-lean-server check endpoint whose Lean project "
                        "has Mathlib + LeanArchitect (env LEAN_SERVER)")
    g.add_argument("--search-server", default=os.environ.get("SEARCH_SERVER", DEFAULT_SEARCH_SERVER),
                   help="Mathlib search endpoint; 'none' disables search. "
                        "Search is turned off automatically if the endpoint "
                        "is unreachable (env SEARCH_SERVER)")
    g.add_argument("--lean-timeout", type=int, default=300,
                   help="per-compile timeout in seconds")

    g = p.add_argument_group("natural-language proof guidance (optional)")
    nl = g.add_mutually_exclusive_group()
    nl.add_argument("--nl-proof", action="store_true",
                    help="first write natural-language proofs (nl_proof.py) "
                         "and use them to guide blueprint generation")
    nl.add_argument("--nl-proof-from-input", action="store_true",
                    help="guide blueprint generation with the `nl_proof` "
                         "field already present in the problems file")
    g.add_argument("--include-failed-nl", action="store_true",
                   help="also use NL proofs the grader did not accept")
    g.add_argument("--nl-model", default=None, help="NL solver/refiner model (default: --model)")
    g.add_argument("--nl-grader-model", default=None, help="NL grader model (default: --model)")
    g.add_argument("--nl-model-url", default=None, help="default: --model-url")
    g.add_argument("--nl-provider", default=None, help="default: --provider")
    g.add_argument("--nl-api-key-env", default=None, help="default: --api-key-env")
    g.add_argument("--nl-reasoning-effort", default=None,
                   choices=["off", "minimal", "low", "medium", "high", "xhigh"],
                   help="default: --reasoning-effort")
    g.add_argument("--nl-temperature", type=float, default=None, help="default: --temperature")
    g.add_argument("--nl-top-p", type=float, default=1.0)
    g.add_argument("--nl-k-solvers", type=int, default=4)
    g.add_argument("--nl-max-rounds", type=int, default=3)
    g.add_argument("--nl-top-k-refine", type=int, default=2)
    g.add_argument("--nl-early-stop-score", type=int, default=9)
    g.add_argument("--nl-consensus-passes", type=int, default=2)
    g.add_argument("--nl-solve-threshold", type=int, default=8)
    g.add_argument("--nl-concurrency", type=int, default=1024)
    g.add_argument("--nl-max-output-tokens", type=int, default=None)
    g.add_argument("--nl-request-timeout", type=float, default=None,
                   help="per-chunk streaming timeout in seconds (default 600)")

    g = p.add_argument_group("blueprint generation")
    g.add_argument("--blueprint-samples", type=int, default=8,
                   help="attempts per problem (stops at the first valid blueprint)")
    g.add_argument("--blueprint-max-turns", type=int, default=64)
    g.add_argument("--blueprint-concurrency", type=int, default=1024)
    g.add_argument("--blueprint-max-model-len", type=int, default=262144)
    g.add_argument("--blueprint-max-output-tokens", type=int, default=None)
    g.add_argument("--blueprint-patience", type=int, default=None,
                   help="stop a sample after N turns without fewer compile errors")

    g = p.add_argument_group("theorem proving")
    g.add_argument("--prove-samples", type=int, default=1)
    g.add_argument("--prove-max-turns", type=int, default=128)
    g.add_argument("--node-retries", type=int, default=4, help="attempts per lemma")
    g.add_argument("--main-retries", type=int, default=None,
                   help="attempts for the main theorem (default: --node-retries)")
    g.add_argument("--prove-concurrency", type=int, default=1024)
    g.add_argument("--prove-max-model-len", type=int, default=65536)
    g.add_argument("--prove-max-output-tokens", type=int, default=None)
    g.add_argument("--proof-sketch", action="store_true",
                   help="show each lemma's informal proof sketch to the prover")

    g = p.add_argument_group("blueprint refinement")
    g.add_argument("--refine-iterations", type=int, default=7,
                   help="refine+prove cycles after the initial proving pass (0 = none)")
    g.add_argument("--refine-samples", type=int, default=8,
                   help="attempts per problem (stops at the first valid revision)")
    g.add_argument("--refine-max-turns", type=int, default=64)
    g.add_argument("--refine-concurrency", type=int, default=1024)
    g.add_argument("--refine-max-model-len", type=int, default=262144)
    g.add_argument("--refine-max-output-tokens", type=int, default=None)

    g = p.add_argument_group("misc")
    g.add_argument("--no-early-stop", action="store_true",
                   help="keep sampling after a problem is solved")
    g.add_argument("--no-prune-dead", action="store_true",
                   help="keep nodes unreachable from the main theorem")
    g.add_argument("--allow-native-decide", action="store_true",
                   help="permit `native_decide` (adds the Lean.ofReduceBool axiom)")
    g.add_argument("--dep-block-feedback-turns", type=int, default=2,
                   help="repair turns for a blueprint whose per-lemma context "
                        "does not compile before the sample is abandoned")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
#  Run configuration (frozen on the first run, replayed on --resume)
# ---------------------------------------------------------------------------

def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.run_dir is None:
        if args.resume:
            sys.exit("--resume needs --run-dir")
        tag = args.model.replace("/", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = f"runs/{tag}_{stamp}"
    run_dir = Path(args.run_dir)
    config_path = run_dir / "run_config.json"
    if args.resume and config_path.exists():
        frozen = json.loads(config_path.read_text())
        for k, v in frozen.items():
            if k not in _NOT_FROZEN:
                setattr(args, k, v)
        print(f"[resume] replaying options from {config_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config = {k: v for k, v in vars(args).items() if k not in _NOT_FROZEN}
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    return args


# ---------------------------------------------------------------------------
#  Preflight checks
# ---------------------------------------------------------------------------

def check_fd_limit(args: argparse.Namespace) -> None:
    """Each stage opens up to `concurrency` sockets to the LLM endpoint plus
    Lean / search connections and log files. Raise the soft open-file limit
    to the hard limit, and abort if a stage's concurrency still would not
    fit (500 descriptors are reserved for everything else)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            soft = hard
        except (ValueError, OSError):
            pass
    budget = soft - 500
    for name in ["nl_concurrency", "blueprint_concurrency",
                 "prove_concurrency", "refine_concurrency"]:
        c = getattr(args, name)
        if c > budget:
            sys.exit(f"ERROR: --{name.replace('_', '-')} {c} exceeds the "
                     f"open-file budget {budget} (ulimit -n = {soft}); lower it.")


def check_llm(args: argparse.Namespace) -> None:
    """The API key must be set (and, on OpenRouter, accepted)."""
    for env_var in {args.api_key_env, args.nl_api_key_env or args.api_key_env}:
        if not os.environ.get(env_var):
            sys.exit(f"ERROR: set {env_var} to your API key.")
    key = os.environ[args.api_key_env]
    base = args.model_url.rstrip("/")
    if "openrouter.ai" in base:
        # /models answers without a valid key; /key validates it.
        try:
            r = httpx.get(base + "/key", headers={"Authorization": f"Bearer {key}"}, timeout=15)
        except httpx.HTTPError as e:
            sys.exit(f"ERROR: cannot reach {base}: {type(e).__name__}: {e}")
        if r.status_code != 200:
            sys.exit(f"ERROR: OpenRouter rejected the key in {args.api_key_env} "
                     f"(HTTP {r.status_code}: {r.text[:200]})")
        return
    try:
        r = httpx.get(base + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
        code = r.status_code
    except httpx.HTTPError as e:
        code = type(e).__name__
    if code != 200:
        # Other endpoints may gate /models; the first real model call is
        # the source of truth there.
        print(f"[preflight] WARNING: {base}/models returned {code} — proceeding")


def check_lean(args: argparse.Namespace) -> None:
    """The Lean server must compile a file importing Mathlib + LeanArchitect.

    One direct request (no retries); the generous timeout covers a cold
    server loading Mathlib for the first time."""
    code = "import Mathlib\nimport Architect\n\n#check Nat\n"
    payload = {"snippets": [{"id": "preflight", "code": code}], "timeout": 600}
    try:
        r = httpx.post(args.lean_server, json=payload, timeout=900)
        r.raise_for_status()
        results = r.json().get("results") or [{}]
        result = results[0]
    except Exception as e:
        sys.exit(f"ERROR: Lean server {args.lean_server} unreachable: "
                 f"{type(e).__name__}: {e}")
    messages = (result.get("response") or {}).get("messages") or []
    errors = [m.get("data", "") for m in messages if m.get("severity") == "error"]
    if result.get("error") or errors:
        sys.exit(f"ERROR: Lean server {args.lean_server} cannot compile "
                 f"`import Mathlib` + `import Architect`: "
                 f"{result.get('error') or errors[0][:300]}\n"
                 f"Its Lean project must include Mathlib and LeanArchitect "
                 f"(see lean/ in this repository).")


# ---------------------------------------------------------------------------
#  Stage runners
# ---------------------------------------------------------------------------

def _opt(flag: str, value) -> list[str]:
    return [] if value is None else [flag, str(value)]


def run_stage(module: str, stage_args: list[str], env: dict) -> int:
    cmd = [sys.executable, "-m", f"goedel_architect.{module}", *stage_args]
    print(f"\n[pipeline] $ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, env=env).returncode


def model_args(args: argparse.Namespace) -> list[str]:
    return [
        "--model-url", args.model_url,
        "--model-name", args.model,
        "--api-key-env", args.api_key_env,
        "--provider", args.provider,
        "--reasoning-effort", args.reasoning_effort,
        "--temperature", str(args.temperature),
        "--tokenizer-path", args.tokenizer,
    ]


def tool_args(args: argparse.Namespace, search: str) -> list[str]:
    return [
        "--lean-server", args.lean_server,
        "--search-server", search,
        "--lean-timeout", str(args.lean_timeout),
    ]


def run_nl_proof(args, run_dir: Path, env: dict) -> int:
    a = [
        "--input", args.problems,
        "--output", str(run_dir / "nl_proof"),
        "--model-url", args.nl_model_url or args.model_url,
        "--api-key-env", args.nl_api_key_env or args.api_key_env,
        "--model-name", args.nl_model or args.model,
        "--grader-model", args.nl_grader_model or args.model,
        "--provider", args.provider if args.nl_provider is None else args.nl_provider,
        "--reasoning-effort", args.nl_reasoning_effort or args.reasoning_effort,
        "--temperature", str(args.temperature if args.nl_temperature is None else args.nl_temperature),
        "--top-p", str(args.nl_top_p),
        "--k-solvers", str(args.nl_k_solvers),
        "--max-rounds", str(args.nl_max_rounds),
        "--top-k-refine", str(args.nl_top_k_refine),
        "--early-stop-score", str(args.nl_early_stop_score),
        "--consensus-passes", str(args.nl_consensus_passes),
        "--solve-threshold", str(args.nl_solve_threshold),
        "--concurrency", str(args.nl_concurrency),
    ]
    a += _opt("--limit", args.limit)
    a += _opt("--max-output-tokens", args.nl_max_output_tokens)
    a += _opt("--request-timeout", args.nl_request_timeout)
    if args.resume:
        a.append("--resume")
    return run_stage("nl_proof", a, env)


def build_nl_guided_input(args, run_dir: Path) -> Path:
    """Rows {problem_id, problem, formal_statement, nl_proof} for every
    problem whose NL proof is non-empty and accepted by the grader (or any
    non-empty proof with --include-failed-nl)."""
    problems = {}
    for line in open(args.problems):
        if line.strip():
            r = json.loads(line)
            problems[r.get("uuid") or r.get("problem_id")] = r
    out_path = run_dir / "blueprint" / "nl_guided_input.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(out_path, "w") as f:
        for path in sorted(glob.glob(str(run_dir / "nl_proof" / "*" / "result.json"))):
            n_in += 1
            r = json.load(open(path))
            if not r.get("proof"):
                continue
            if not args.include_failed_nl and r.get("status") != "solved":
                continue
            pid = r.get("uuid") or r.get("problem_id")
            prob = problems.get(pid, {})
            formal = prob.get("formal_statement") or r.get("formal_statement", "")
            if not formal:
                print(f"  warn: no formal_statement for {pid}; skipping", file=sys.stderr)
                continue
            f.write(json.dumps({
                "problem_id": pid,
                "problem": prob.get("problem", r.get("problem", "")),
                "formal_statement": formal,
                "nl_proof": r["proof"],
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[pipeline] {n_out}/{n_in} NL proofs carried into blueprint generation")
    if n_out == 0:
        sys.exit("ERROR: no usable NL proofs (try --include-failed-nl).")
    return out_path


def run_blueprint(args, run_dir: Path, search: str, env: dict, input_path: str,
                  nl_guided: bool) -> int:
    a = ["--input", input_path, "--output", str(run_dir / "blueprint"),
         *model_args(args), *tool_args(args, search),
         "--max-turns", str(args.blueprint_max_turns),
         "--num-samples", str(args.blueprint_samples),
         "--max-model-len", str(args.blueprint_max_model_len),
         "--concurrency", str(args.blueprint_concurrency)]
    a += _opt("--limit", args.limit)
    a += _opt("--max-output-tokens", args.blueprint_max_output_tokens)
    a += _opt("--patience", args.blueprint_patience)
    if args.resume:
        a.append("--resume")
    if not args.no_early_stop:
        a.append("--early-stop")
    if args.no_prune_dead:
        a.append("--no-prune-dead")
    if nl_guided:
        a.append("--nl-proof-decompose")
    return run_stage("blueprint", a, env)


def run_prove(args, blueprint_dir: Path, out_dir: Path, search: str, env: dict) -> int:
    if not (blueprint_dir / "traces.jsonl").exists():
        sys.exit(f"ERROR: {blueprint_dir}/traces.jsonl missing")
    a = ["--skeleton-results", str(blueprint_dir), "--output", str(out_dir),
         "--input-jsonl", args.problems,
         *model_args(args), *tool_args(args, search),
         "--max-turns", str(args.prove_max_turns),
         "--num-samples", str(args.prove_samples),
         "--concurrency", str(args.prove_concurrency),
         "--max-model-len", str(args.prove_max_model_len),
         "--node-retries", str(args.node_retries),
         # Always on: the refinement stage reads each lemma's forfeit
         # diagnosis and formal-disproof signal.
         "--allow-negation", "--allow-forfeit"]
    a += _opt("--limit", args.limit)
    a += _opt("--max-output-tokens", args.prove_max_output_tokens)
    a += _opt("--main-retries", args.main_retries)
    if args.resume:
        a.append("--resume")
    if args.no_early_stop:
        a.append("--no-early-stop")
    if args.no_prune_dead:
        a.append("--no-prune-dead")
    if not args.proof_sketch:
        a.append("--skip-proof-sketch")
    return run_stage("prover", a, env)


def run_refine(args, prover_traces: Path, out_dir: Path, search: str, env: dict) -> int:
    if not prover_traces.exists():
        print(f"[pipeline] {prover_traces} missing — cannot refine")
        return 1
    a = ["--prover-traces", str(prover_traces), "--output", str(out_dir),
         *model_args(args), *tool_args(args, search),
         "--max-turns", str(args.refine_max_turns),
         "--num-samples", str(args.refine_samples),
         "--max-model-len", str(args.refine_max_model_len),
         "--concurrency", str(args.refine_concurrency)]
    a += _opt("--limit", args.limit)
    a += _opt("--max-output-tokens", args.refine_max_output_tokens)
    if args.resume:
        a.append("--resume")
    if not args.no_early_stop:
        a.append("--early-stop")
    if args.no_prune_dead:
        a.append("--no-prune-dead")
    return run_stage("refine", a, env)


# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------

def stage_summary(name: str, d: Path) -> None:
    path = d / "summary.json"
    if not path.exists():
        print(f"[{name}] summary.json not found (stage skipped or failed)")
        return
    s = json.loads(path.read_text())
    print(f"[{name}]")
    if "total_input_tokens" in s:
        print(f"  tokens: input={s['total_input_tokens']:,} "
              f"cached={s.get('total_cached_tokens', 0):,} "
              f"({s.get('cache_hit_rate', 0) * 100:.1f}% hit) "
              f"output={s.get('total_output_tokens', 0):,}")
    print(f"  cost:   ${s.get('total_cost_usd', 0):.4f}   wall: {s.get('elapsed_seconds', 0):.0f}s")
    if "solved" in s and "completed" in s:
        print(f"  NL proofs accepted: {s['solved']}/{s['completed']}")
    if "solved_any_sample" in s and "total_problems" in s:
        print(f"  solved (any sample): {s['solved_any_sample']}/{s['total_problems']} "
              f"({s.get('solve_rate_any', 0) * 100:.1f}%)")
    if "assembled_any_sample" in s and "total_problems" in s:
        print(f"  assembled (any sample): {s['assembled_any_sample']}/{s['total_problems']} "
              f"({s.get('assembly_rate_any', 0) * 100:.1f}%)")


def rollup(run_dir: Path, n_iters: int) -> dict:
    """Best proving result per problem across iterations (the earliest
    iteration that solved it)."""
    best: dict[str, tuple[str, bool]] = {}
    for k in range(n_iters + 1):
        path = run_dir / f"iter{k:02d}" / "prove" / "traces.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            if not line.strip():
                continue
            t = json.loads(line)
            pid = t.get("problem_id")
            if not pid:
                continue
            cur = best.get(pid)
            ok = bool(t.get("success"))
            # Prefer success over non-success; among equals keep the
            # earliest iteration.
            if cur is None or (ok and not cur[1]):
                best[pid] = (f"iter{k:02d}", ok)
    solved = sorted(p for p, (_, ok) in best.items() if ok)
    first_solved: dict[str, int] = {}
    for _, (it, ok) in best.items():
        if ok:
            first_solved[it] = first_solved.get(it, 0) + 1
    return {"problems": len(best), "solved": len(solved),
            "solved_ids": solved, "first_solved_by_iteration": first_solved}


def main(argv: list[str] | None = None) -> None:
    args = resolve_config(parse_args(argv))
    run_dir = Path(args.run_dir)
    print(f"[pipeline] run dir: {run_dir}")
    print(f"[pipeline] model:   {args.model} via {args.model_url}"
          + (f" (provider {args.provider})" if args.provider else ""))
    print(f"[pipeline] Lean:    {args.lean_server}")

    check_fd_limit(args)
    check_llm(args)
    check_lean(args)
    search = resolve_search_server(args.search_server) or "none"
    print(f"[pipeline] search:  {search}")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["ALLOW_NATIVE_DECIDE"] = "1" if args.allow_native_decide else "0"
    env["DEP_BLOCK_FEEDBACK_TURNS"] = str(args.dep_block_feedback_turns)

    # Natural-language proof guidance (optional).
    nl_guided = bool(args.nl_proof or args.nl_proof_from_input)
    blueprint_input = args.problems
    if args.nl_proof:
        if run_nl_proof(args, run_dir, env) != 0:
            sys.exit("ERROR: NL proof stage failed")
        blueprint_input = str(build_nl_guided_input(args, run_dir))

    # Blueprint generation.
    if run_blueprint(args, run_dir, search, env, blueprint_input, nl_guided) != 0:
        sys.exit("ERROR: blueprint generation failed")

    # Initial proving pass (fatal on failure: nothing to fall back to).
    if run_prove(args, run_dir / "blueprint", run_dir / "iter00" / "prove", search, env) != 0:
        sys.exit("ERROR: theorem proving (iter00) failed")

    # Refine + re-prove cycles. A failing cycle stops the loop; results of
    # the completed iterations stand.
    n_done = 0
    for k in range(1, args.refine_iterations + 1):
        prev = run_dir / f"iter{k - 1:02d}" / "prove" / "traces.jsonl"
        refine_dir = run_dir / f"iter{k:02d}" / "refine"
        if run_refine(args, prev, refine_dir, search, env) != 0:
            print(f"[iter{k:02d}] refinement failed — stopping")
            break
        traces = refine_dir / "traces.jsonl"
        if not traces.exists() or traces.stat().st_size == 0:
            print(f"[iter{k:02d}] nothing left to refine — stopping")
            break
        if run_prove(args, refine_dir, run_dir / f"iter{k:02d}" / "prove", search, env) != 0:
            print(f"[iter{k:02d}] theorem proving failed — stopping")
            break
        n_done = k

    print("\n" + "=" * 60 + "\n  Pipeline complete\n" + "=" * 60)
    if args.nl_proof:
        stage_summary("nl_proof", run_dir / "nl_proof")
    stage_summary("blueprint", run_dir / "blueprint")
    for k in range(n_done + 1):
        if k >= 1:
            stage_summary(f"iter{k:02d}/refine", run_dir / f"iter{k:02d}" / "refine")
        stage_summary(f"iter{k:02d}/prove", run_dir / f"iter{k:02d}" / "prove")
    summary = rollup(run_dir, args.refine_iterations)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsolved {summary['solved']}/{summary['problems']} problems: "
          f"{', '.join(summary['solved_ids']) or '-'}")
    for it, n in sorted(summary["first_solved_by_iteration"].items()):
        print(f"  first solved at {it}: {n}")
    print(f"proofs: {run_dir}/iterNN/prove/proved_blueprints/*.lean")


if __name__ == "__main__":
    main()
