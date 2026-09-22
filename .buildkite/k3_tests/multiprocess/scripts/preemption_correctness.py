# SPDX-License-Identifier: Apache-2.0
"""End-to-end preemption correctness ladder for the LMCache MP connector.

Drives two OpenAI-compatible vLLM servers -- a baseline without LMCache and
one with ``LMCacheMPConnector`` -- through the same token-id workloads and
compares generated token ids exactly.  Run it once per scheduler mode
(regular and async scheduling) and per transfer mode; the servers must be
started by the caller with ``VLLM_BATCH_INVARIANT=1`` on a model vLLM's
batch-invariant mode is validated for.

Two workloads, because they prove different things:

* **unique**: every prompt starts with its own 64-token tag, so no two
  requests share a single LMCache chunk key.  On a cold cache, any LMCache hit
  during the high-concurrency pass can only be a preempted request loading
  the KV it stored itself.  Proves preempt -> store -> resume/load ownership.
* **shared**: a few 192-token prefixes shared by many requests, so the
  connector must reconcile vLLM's own prefix cache with LMCache
  (``skip_first_n_tokens``) and a corrupted chunk is re-read by other
  requests.  Proves cross-request reuse under preemption.

Passes, in order:

  1. baseline, high concurrency, both workloads  -> reference token ids
     (+ top-2 logprobs) and the A/A noise floor at low concurrency
  2. lmcache, unique workload, high concurrency  -> preemption + resume-load
  3. lmcache, shared workload, high concurrency  -> reuse/overlap under preemption
  4. lmcache, shared workload, low concurrency   -> warm-cache integrity: every
     request must load what pass 3 stored
  (3 and 4 repeat ``--repeats`` times)

Preemption is asserted from vLLM's own ``/metrics`` counter.  Prompts are
token ids with ``ignore_eos`` and a fixed ``max_tokens`` so KV demand is
deterministic; run the servers with ``--num-gpu-blocks-override N`` such that
``concurrency * (prompt + max_tokens) >> N * block_size``.
"""

# Standard
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Third Party
import aiohttp

PREEMPTION_METRIC = "vllm:num_preemptions_total"
RUNNING_METRIC = "vllm:num_requests_running"
WAITING_METRIC = "vllm:num_requests_waiting"
# Connector prefix-cache counters: vLLM records them per admission, so
# re-admissions after preemption are included.
EXTERNAL_QUERIES_METRIC = "vllm:external_prefix_cache_queries_total"
EXTERNAL_HITS_METRIC = "vllm:external_prefix_cache_hits_total"
LOCAL_QUERIES_METRIC = "vllm:prefix_cache_queries_total"
LOCAL_HITS_METRIC = "vllm:prefix_cache_hits_total"
TRACKED_METRICS = (
    EXTERNAL_QUERIES_METRIC,
    EXTERNAL_HITS_METRIC,
    LOCAL_QUERIES_METRIC,
    LOCAL_HITS_METRIC,
)
# One LMCache chunk; the unique-workload tag must cover at least one so the
# very first chunk key already differs between requests.
TAG_TOKENS = 64


@dataclass
class Prompt:
    request_id: str
    token_ids: list[int]
    max_tokens: int


@dataclass
class Completion:
    request_id: str
    token_ids: list[int]
    text: str
    finish_reason: str
    latency_s: float
    lmcache_cached_tokens: int | None = None
    error: str | None = None
    # Per output position: (best token id, best logprob, runner-up token id,
    # runner-up logprob).  Only collected for the reference pass.
    top2: list[tuple[int, float, int, float]] = field(default_factory=list)


@dataclass
class PassResult:
    name: str
    completions: dict[str, Completion] = field(default_factory=dict)
    preemptions_before: float = 0.0
    preemptions_after: float = 0.0
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0
    drained: bool = True

    @property
    def preemptions(self) -> float:
        return self.preemptions_after - self.preemptions_before

    @property
    def errors(self) -> list[Completion]:
        return [c for c in self.completions.values() if c.error]

    @property
    def not_length_capped(self) -> list[Completion]:
        return [
            c
            for c in self.completions.values()
            if not c.error and c.finish_reason != "length"
        ]

    def delta(self, metric: str) -> float:
        return self.metrics_after.get(metric, float("nan")) - self.metrics_before.get(
            metric, float("nan")
        )


_TEXT = (
    "The history of computing is a history of people trying to make machines "
    "do arithmetic faster and more reliably than they could by hand. Early "
    "mechanical calculators used gears and levers; later designs replaced "
    "them with relays, then vacuum tubes, then transistors. Each generation "
    "was smaller, faster, and cheaper than the last, and each opened up new "
    "kinds of problems that could be attacked with computation. Programming "
    "languages evolved alongside the hardware, moving from raw machine codes "
    "to assembly mnemonics to high level notations that let a person describe "
    "an algorithm in something closer to ordinary prose. Operating systems "
    "appeared to share expensive machines among many users, and networks "
    "connected those machines into systems that spanned buildings, cities, "
    "and eventually the whole planet. Storage grew from punched cards and "
    "paper tape to magnetic drums, disks, and solid state memory, with "
    "capacities that increased by orders of magnitude every decade. Along "
    "the way, ideas from mathematics and logic gave the field its theoretical "
    "foundation: models of computation, notions of complexity, and proofs "
    "about what can and cannot be decided by any machine at all. "
)


def _text_token_pool(model: str, need: int) -> list[int] | None:
    """Tokenize real prose with the model's tokenizer; None if unavailable."""
    try:
        # Third Party
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
    except Exception:  # noqa: BLE001 - fall back to random ids
        return None
    ids: list[int] = []
    paragraph = 0
    while len(ids) < need:
        # Vary the paragraph so repeated text does not create spurious
        # shared prefixes between unrelated requests.
        ids.extend(
            tok.encode(f"Section {paragraph}. " + _TEXT, add_special_tokens=False)
        )
        paragraph += 1
    return ids


@dataclass
class WorkloadSpec:
    name: str
    num_requests: int
    prompt_min: int
    prompt_max: int
    max_tokens: int
    num_shared_prefixes: int
    shared_prefix_len: int
    vocab: int
    seed: int


def build_workload(model: str, spec: WorkloadSpec) -> list[Prompt]:
    """Token-id prompts from real prose (random ids if no tokenizer).

    ``num_shared_prefixes == 0`` builds the *unique* workload: each prompt
    starts with a request-specific ``TAG_TOKENS``-long tag drawn from a
    request-specific pool region, so no two requests (and no two seeds)
    share their first chunk and therefore no LMCache chunk key.  Otherwise
    ``num_shared_prefixes`` prefixes of ``shared_prefix_len`` tokens are shared
    round-robin across requests.
    """
    rng = random.Random(spec.seed)
    need = (
        spec.num_requests * (spec.prompt_max + TAG_TOKENS) + spec.shared_prefix_len * 4
    )
    pool = _text_token_pool(model, need=need)

    def draw(n: int) -> list[int]:
        if pool is None:
            return [rng.randrange(10, spec.vocab) for _ in range(n)]
        start = rng.randrange(0, max(1, len(pool) - n))
        return pool[start : start + n]

    prompts = []
    if spec.num_shared_prefixes > 0:
        prefixes = [
            draw(spec.shared_prefix_len) for _ in range(spec.num_shared_prefixes)
        ]
        for i in range(spec.num_requests):
            total = rng.randint(spec.prompt_min, spec.prompt_max)
            prefix = prefixes[i % spec.num_shared_prefixes]
            tail = draw(max(1, total - len(prefix)))
            prompts.append(
                Prompt(f"{spec.name}-{spec.seed}-{i}", prefix + tail, spec.max_tokens)
            )
    else:
        for i in range(spec.num_requests):
            total = rng.randint(spec.prompt_min, spec.prompt_max)
            # Deterministic, request- and seed-specific first chunk: real
            # tokens from a disjoint pool slice when possible, so the model
            # still sees text rather than noise.
            if pool is not None and len(pool) >= (i + 2) * TAG_TOKENS:
                base = (
                    (spec.seed * spec.num_requests + i)
                    * TAG_TOKENS
                    % (len(pool) - TAG_TOKENS)
                )
                tag = pool[base : base + TAG_TOKENS]
            else:
                tag = [
                    10 + (spec.seed * 100_003 + i * 1_009 + k) % (spec.vocab - 10)
                    for k in range(TAG_TOKENS)
                ]
            tail = draw(max(1, total - TAG_TOKENS))
            prompts.append(
                Prompt(f"{spec.name}-{spec.seed}-{i}", tag + tail, spec.max_tokens)
            )
    src = "random token ids" if pool is None else "tokenized prose"
    print(f"workload {spec.name}: {len(prompts)} prompts, source={src}")
    return prompts


def scrape_metric(base_url: str, name: str) -> float:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
            body = resp.read().decode()
    except Exception:
        return float("nan")
    total = 0.0
    found = False
    for line in body.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            total += float(line.rsplit(" ", 1)[1])
            found = True
    return total if found else float("nan")


def wait_for_drain(base_url: str, timeout_s: float = 30.0) -> bool:
    """True once the server reports no running and no waiting requests."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        running = scrape_metric(base_url, RUNNING_METRIC)
        waiting = scrape_metric(base_url, WAITING_METRIC)
        if running == 0 and waiting == 0:
            return True
        time.sleep(0.5)
    return False


async def _one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: Prompt,
    sem: asyncio.Semaphore,
    want_cache_stats: bool,
    want_logprobs: bool,
) -> Completion:
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt.token_ids,
        "max_tokens": prompt.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "return_token_ids": True,
        "request_id": prompt.request_id,
    }
    if want_cache_stats:
        payload["kv_transfer_params"] = {"cached_token_stats": {}}
    if want_logprobs:
        payload["logprobs"] = 2
        payload["return_tokens_as_token_ids"] = True
    async with sem:
        start = time.monotonic()
        try:
            async with session.post(f"{url}/v1/completions", json=payload) as resp:
                body = await resp.json()
                if resp.status != 200:
                    return Completion(
                        prompt.request_id, [], "", "", 0.0, error=json.dumps(body)[:300]
                    )
        except Exception as exc:  # noqa: BLE001 - report, do not crash the run
            return Completion(prompt.request_id, [], "", "", 0.0, error=repr(exc))
        latency = time.monotonic() - start
    choice = body["choices"][0]
    stats = (body.get("kv_transfer_params") or {}).get("cached_token_stats") or {}
    top2: list[tuple[int, float, int, float]] = []
    logprobs = choice.get("logprobs") or {}
    for entry in logprobs.get("top_logprobs") or []:
        ranked = sorted(
            ((_token_id(k), float(v)) for k, v in entry.items()), key=lambda t: -t[1]
        )
        if len(ranked) >= 2:
            top2.append((ranked[0][0], ranked[0][1], ranked[1][0], ranked[1][1]))
        elif ranked:
            top2.append((ranked[0][0], ranked[0][1], -1, float("-inf")))
    return Completion(
        request_id=prompt.request_id,
        token_ids=list(choice.get("token_ids") or []),
        text=choice.get("text", ""),
        finish_reason=choice.get("finish_reason", ""),
        latency_s=latency,
        lmcache_cached_tokens=stats.get("num_lmcache_cached_tokens"),
        top2=top2,
    )


def _token_id(key: str) -> int:
    """``return_tokens_as_token_ids`` renders keys as ``token_id:123``."""
    if key.startswith("token_id:"):
        return int(key.split(":", 1)[1])
    return -1


async def run_pass(
    name: str,
    url: str,
    model: str,
    prompts: list[Prompt],
    concurrency: int,
    want_cache_stats: bool,
    want_logprobs: bool = False,
) -> PassResult:
    result = PassResult(name=name)
    result.preemptions_before = scrape_metric(url, PREEMPTION_METRIC)
    result.metrics_before = {m: scrape_metric(url, m) for m in TRACKED_METRICS}
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    start = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _one(session, url, model, p, sem, want_cache_stats, want_logprobs)
            for p in prompts
        ]
        for coro in asyncio.as_completed(tasks):
            c = await coro
            result.completions[c.request_id] = c
    result.wall_s = time.monotonic() - start
    result.preemptions_after = scrape_metric(url, PREEMPTION_METRIC)
    result.metrics_after = {m: scrape_metric(url, m) for m in TRACKED_METRICS}
    result.drained = wait_for_drain(url)
    return result


@dataclass
class Divergence:
    request_id: str
    position: int
    ref_token: int
    got_token: int
    gap: float  # reference top-1 minus top-2 logprob at ``position``
    got_is_runner_up: bool
    cached_tokens: int | None

    def benign(self, near_tie_gap: float) -> bool:
        """Numerical noise flips a near-tie to the runner-up; nothing else."""
        return self.got_is_runner_up and self.gap <= near_tie_gap

    def describe(self, total: int) -> str:
        kind = "near-tie" if self.got_is_runner_up else "not in top-2"
        return (
            f"{self.request_id}: diverges at output token {self.position}/{total} "
            f"ref={self.ref_token} got={self.got_token} top1-top2 gap={self.gap:.4f} "
            f"({kind}, lmcache_cached_tokens={self.cached_tokens})"
        )


def compare(
    reference: PassResult, other: PassResult
) -> tuple[list[Divergence], list[str]]:
    """Classify every request whose token ids differ from the reference.

    Returns the divergences plus lines for requests that are missing or
    errored (always hard failures).
    """
    divergences: list[Divergence] = []
    hard: list[str] = []
    for rid, ref in reference.completions.items():
        got = other.completions.get(rid)
        if got is None or got.error:
            hard.append(f"{rid}: missing or errored ({got.error if got else 'absent'})")
            continue
        if ref.token_ids == got.token_ids:
            continue
        first = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(ref.token_ids, got.token_ids, strict=False)
                )
                if a != b
            ),
            min(len(ref.token_ids), len(got.token_ids)),
        )
        ref_tok = ref.token_ids[first] if first < len(ref.token_ids) else -1
        got_tok = got.token_ids[first] if first < len(got.token_ids) else -1
        gap = float("inf")
        runner_up = False
        if first < len(ref.top2):
            _t1, lp1, t2, lp2 = ref.top2[first]
            gap = lp1 - lp2
            runner_up = got_tok == t2
        divergences.append(
            Divergence(
                rid, first, ref_tok, got_tok, gap, runner_up, got.lmcache_cached_tokens
            )
        )
    return divergences, hard


def summarize(result: PassResult) -> str:
    n = len(result.completions)
    errors = len(result.errors)
    lengths = sum(c.finish_reason == "length" for c in result.completions.values())
    cached = [
        c.lmcache_cached_tokens
        for c in result.completions.values()
        if c.lmcache_cached_tokens is not None
    ]
    lat = sorted(c.latency_s for c in result.completions.values() if not c.error)
    p99 = lat[int(0.99 * (len(lat) - 1))] if lat else float("nan")
    cache_line = ""
    if cached:
        served = sum(1 for c in cached if c > 0)
        cache_line = (
            f"\n    LMCache-served requests: {served}/{len(cached)}, "
            f"mean served tokens {sum(cached) / len(cached):.0f}"
        )
    ext_hits = result.delta(EXTERNAL_HITS_METRIC)
    ext_queries = result.delta(EXTERNAL_QUERIES_METRIC)
    loc_hits = result.delta(LOCAL_HITS_METRIC)
    loc_queries = result.delta(LOCAL_QUERIES_METRIC)
    return (
        f"[{result.name}] requests={n} errors={errors} finish=length:{lengths} "
        f"preemptions={result.preemptions:.0f} wall={result.wall_s:.1f}s "
        f"p99_latency={p99:.2f}s drained={result.drained}{cache_line}\n"
        f"    prefix-cache tokens: vLLM APC {loc_hits:.0f}/{loc_queries:.0f}, "
        f"LMCache {ext_hits:.0f}/{ext_queries:.0f} (queries include re-admissions)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-url", required=True)
    ap.add_argument("--lmcache-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--hot-concurrency", type=int, default=64)
    ap.add_argument("--replay-concurrency", type=int, default=4)
    ap.add_argument("--prompt-min", type=int, default=256)
    ap.add_argument("--prompt-max", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--shared-prefixes", type=int, default=4)
    ap.add_argument("--shared-prefix-len", type=int, default=192)
    ap.add_argument("--vocab", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1, help="shared hot+replay rounds")
    ap.add_argument(
        "--min-replay-hit-fraction",
        type=float,
        default=0.8,
        help="the warm replay must report at least this fraction of prompt "
        "tokens served from LMCache, else it proves nothing",
    )
    ap.add_argument(
        "--near-tie-gap",
        type=float,
        default=0.1,
        help="a divergence is benign only if the reference's top-1/top-2 logprob "
        "gap at that position is at most this many nats and the other run "
        "produced the runner-up token",
    )
    ap.add_argument(
        "--max-slowdown-percent",
        type=float,
        default=None,
        help="fail if an LMCache high-concurrency pass is slower than its "
        "baseline by more than this (unset: report only)",
    )
    ap.add_argument("--skip-unique", action="store_true", help="skip pass 2")
    ap.add_argument("--skip-shared", action="store_true", help="skip passes 3 and 4")
    ap.add_argument("--output-dir", type=Path, default=Path("preemption_results"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    results: list[PassResult] = []

    def record(result: PassResult) -> PassResult:
        results.append(result)
        print(summarize(result))
        if result.errors:
            failures.append(f"{result.name}: {len(result.errors)} errored requests")
        if result.not_length_capped:
            failures.append(
                f"{result.name}: {len(result.not_length_capped)} requests did not "
                f"finish with finish_reason=length"
            )
        if not result.drained:
            failures.append(
                f"{result.name}: server still reports running/waiting requests "
                f"30 s after the last response"
            )
        return result

    def judge(label: str, reference: PassResult, other: PassResult) -> None:
        divergences, hard = compare(reference, other)
        benign = [d for d in divergences if d.benign(args.near_tie_gap)]
        real = [d for d in divergences if not d.benign(args.near_tie_gap)]
        print(
            f"    {label}: {len(divergences)} divergent requests "
            f"({len(benign)} near-tie, {len(real)} hard), {len(hard)} missing/errored"
        )
        for d in divergences[:10]:
            print("      ", d.describe(args.max_tokens))
        for line in hard:
            failures.append(f"{label}: {line}")
        if real:
            failures.append(f"{label}: {len(real)} hard divergences from reference")

    def slowdown(label: str, reference: PassResult, other: PassResult) -> None:
        pct = (
            (other.wall_s / reference.wall_s - 1.0) * 100.0 if reference.wall_s else 0.0
        )
        print(
            f"    {label}: wall {other.wall_s:.1f}s vs baseline "
            f"{reference.wall_s:.1f}s ({pct:+.1f}%)"
        )
        if args.max_slowdown_percent is not None and pct > args.max_slowdown_percent:
            failures.append(
                f"{label}: {pct:.1f}% slower than baseline "
                f"(limit {args.max_slowdown_percent}%)"
            )

    def run(name: str, url: str, prompts: list[Prompt], conc: int, **kw) -> PassResult:
        return record(asyncio.run(run_pass(name, url, args.model, prompts, conc, **kw)))

    common = dict(
        num_requests=args.num_requests,
        prompt_min=args.prompt_min,
        prompt_max=args.prompt_max,
        max_tokens=args.max_tokens,
        vocab=args.vocab,
        seed=args.seed,
    )
    unique = build_workload(
        args.model,
        WorkloadSpec(
            name="unique", num_shared_prefixes=0, shared_prefix_len=0, **common
        ),
    )
    shared = build_workload(
        args.model,
        WorkloadSpec(
            name="shared",
            num_shared_prefixes=args.shared_prefixes,
            shared_prefix_len=args.shared_prefix_len,
            **common,
        ),
    )
    for wl in (unique, shared):
        demand = sum(len(p.token_ids) + p.max_tokens for p in wl)
        label = wl[0].request_id.split("-")[0]
        print(f"  {label} workload KV demand: {demand} tokens")

    # ---- 1. references and A/A floors ----------------------------------------
    refs: dict[str, PassResult] = {}
    for label, wl in (("unique", unique), ("shared", shared)):
        if (label == "unique" and args.skip_unique) or (
            label == "shared" and args.skip_shared
        ):
            continue
        ref = run(
            f"baseline-{label}-hot",
            args.baseline_url,
            wl,
            args.hot_concurrency,
            want_cache_stats=False,
            want_logprobs=True,
        )
        if not all(c.token_ids for c in ref.completions.values() if not c.error):
            failures.append(f"{ref.name}: no token ids; vLLM lacks return_token_ids?")
        if not all(c.top2 for c in ref.completions.values() if not c.error):
            print("    WARNING: no top-2 logprobs; every divergence counts as hard")
        refs[label] = ref
        floor = run(
            f"baseline-{label}-lowconc",
            args.baseline_url,
            wl,
            args.replay_concurrency,
            want_cache_stats=False,
        )
        divergences, _hard = compare(ref, floor)
        benign = sum(d.benign(args.near_tie_gap) for d in divergences)
        print(
            f"    A/A floor ({label}): {len(divergences)} divergent, "
            f"{benign} near-tie, {len(divergences) - benign} would count as hard"
        )
        for d in divergences[:5]:
            print("      ", d.describe(args.max_tokens))

    # ---- 2. unique workload: ownership / resume-load -------------------------
    if not args.skip_unique:
        hot = run(
            "lmcache-unique-hot",
            args.lmcache_url,
            unique,
            args.hot_concurrency,
            want_cache_stats=True,
        )
        if not hot.preemptions > 0:
            failures.append(f"{hot.name}: no preemptions observed; workload too small")
        served = [
            c for c in hot.completions.values() if (c.lmcache_cached_tokens or 0) > 0
        ]
        print(
            f"    resume-load: {len(served)}/{len(hot.completions)} requests loaded "
            f"KV from LMCache on a cold cache with unique prompts; every such load "
            f"is a preempted request reading its own earlier KV"
        )
        if not served:
            failures.append(
                f"{hot.name}: no request loaded KV from LMCache despite "
                f"{hot.preemptions:.0f} preemptions; resume-load is not working"
            )
        judge(f"T2/T6 {hot.name}", refs["unique"], hot)
        slowdown(f"T1 {hot.name}", refs["unique"], hot)

    # ---- 3 + 4. shared workload: reuse under preemption, then warm replay ----
    if not args.skip_shared:
        for rnd in range(args.repeats):
            hot = run(
                f"lmcache-shared-hot-{rnd}",
                args.lmcache_url,
                shared,
                args.hot_concurrency,
                want_cache_stats=True,
            )
            if not hot.preemptions > 0:
                failures.append(
                    f"{hot.name}: no preemptions observed; workload too small"
                )
            judge(f"T2 {hot.name}", refs["shared"], hot)
            slowdown(f"T1 {hot.name}", refs["shared"], hot)

            replay = run(
                f"lmcache-shared-replay-{rnd}",
                args.lmcache_url,
                shared,
                args.replay_concurrency,
                want_cache_stats=True,
            )
            if replay.preemptions > 0:
                failures.append(
                    f"{replay.name}: {replay.preemptions:.0f} preemptions during the "
                    f"low-concurrency replay; raise the pool or lower concurrency"
                )
            cached = [
                (c.lmcache_cached_tokens or 0) / len(p.token_ids)
                for p in shared
                if (c := replay.completions.get(p.request_id)) is not None
            ]
            hit_fraction = sum(cached) / len(cached) if cached else 0.0
            print(
                f"    replay LMCache hit fraction of prompt tokens: {hit_fraction:.2f}"
            )
            if hit_fraction < args.min_replay_hit_fraction:
                failures.append(
                    f"{replay.name}: hit fraction {hit_fraction:.2f} < "
                    f"{args.min_replay_hit_fraction}; the replay did not read the cache"
                )
            judge(f"T3 {replay.name}", refs["shared"], replay)

    for res in results:
        with open(args.output_dir / f"{res.name}.json", "w") as f:
            json.dump(
                {rid: c.__dict__ for rid, c in res.completions.items()}, f, indent=1
            )

    if failures:
        print("FAILED:")
        for line in failures:
            print("  -", line)
        return 1
    print(
        "PASSED: no crash or wedge, unique-prompt resume-load matches the reference, "
        "shared-prefix hot pass matches, warm replay matches"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
