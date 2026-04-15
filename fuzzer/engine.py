"""
fuzzer/engine.py — Core fuzzing loop.

Three phases:

Phase 1 — Deterministic bootstrap
  For every seed prompt × every transform in the configured groups:
    - Deterministic transforms (tools_pool): apply directly, no composer needed
    - LLM-based transforms (strategies_pool): composer with single-strategy chain
  All results feed into the shared corpus.

Phase 2 — Random havoc (UCB1-guided)
  UCB1-weighted sampling of strategy + tool chains.
  Composer generates the mutated prompt using the composite annotation model.
  Runs until plateau_threshold consecutive non-novel iterations triggers phase 3.

Phase 3 — LLM-guided composition
  Composer proposes chains from score history.
  Interleaves with phase 2: resets to phase 2 when a novel entry is found.

Concurrency:
  asyncio.Semaphore(max_concurrent) over the inner loop body.
  Corpus and UCB1 weights are protected by asyncio.Lock inside their classes.
  Results from concurrent workers are processed as they complete.

Budget tracking:
  _estimate_cost() uses a simple $/million-token table for common OpenAI models.
  Unknown models default to gpt-4o pricing (conservative overestimate).
  state.budget_spent_usd is updated after every LLM API response that returns
  usage data (composer, scorer, seed_generator). Target calls are not counted
  because the target is the system under test, not our infrastructure cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich import box

from transforms import TRANSFORMS  # type: ignore[import]

from .chain import ChainBuilder
from .composer import Composer
from .corpus import Chain, Corpus, CorpusEntry, make_entry
from .scorer import Scorer, ScoreResult
from .target import Target

if TYPE_CHECKING:
    from .campaign import CampaignConfig

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Budget / cost estimation
# ---------------------------------------------------------------------------

# $/million tokens — input and output pricing for common models.
# Figures approximate as of 2025; update as needed.
_COST_TABLE: dict[str, tuple[float, float]] = {
    # model-prefix          input $/M    output $/M
    "gpt-4o-mini":         (0.15,        0.60),
    "gpt-4o":              (2.50,       10.00),
    "gpt-4-turbo":         (10.00,      30.00),
    "gpt-4":               (30.00,      60.00),
    "gpt-3.5-turbo":       (0.50,        1.50),
    "o1-mini":             (1.10,        4.40),
    "o1":                  (15.00,      60.00),
    "o3-mini":             (1.10,        4.40),
    "claude-3-5-sonnet":   (3.00,       15.00),
    "claude-3-opus":       (15.00,      75.00),
    "claude-3-haiku":      (0.25,        1.25),
}
_DEFAULT_COST = (2.50, 10.00)  # gpt-4o fallback


def _model_cost(model: str) -> tuple[float, float]:
    """Return (input_per_M, output_per_M) for a model name (prefix match)."""
    m = model.lower()
    for prefix, prices in _COST_TABLE.items():
        if m.startswith(prefix):
            return prices
    return _DEFAULT_COST


def _estimate_cost(model: str, usage) -> float:
    """
    Estimate USD cost from an OpenAI CompletionUsage object.
    Returns 0.0 if usage is None or missing fields.
    """
    if usage is None:
        return 0.0
    inp  = getattr(usage, "prompt_tokens",     0) or 0
    out  = getattr(usage, "completion_tokens", 0) or 0
    pi, po = _model_cost(model)
    return (inp * pi + out * po) / 1_000_000


# ---------------------------------------------------------------------------
# Iteration result (passed from worker to main loop)
# ---------------------------------------------------------------------------


@dataclass
class IterResult:
    iteration:      int
    phase:          str           # "phase1_tool" | "phase1_strategy" | "phase2"
    seed_prompt:    str
    chain:          Chain
    mutated_prompt: Optional[str] # None = annotation failure / composer error
    response:       Optional[str] # None = target error
    score:          Optional[ScoreResult]
    admitted:       bool
    crashed:        bool
    elapsed_s:      float


# ---------------------------------------------------------------------------
# Engine state (shared across concurrent workers)
# ---------------------------------------------------------------------------


@dataclass
class EngineState:
    iteration:          int   = 0
    non_novel_streak:   int   = 0
    phase:              str   = "phase1"
    budget_spent_usd:   float = 0.0
    crashes_found:      int   = 0


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------


async def run(
    config: "CampaignConfig",
    corpus: Corpus,
    chain_builder: ChainBuilder,
    composer: Composer,
    scorer: Scorer,
    target: Target,
    state: Optional[EngineState] = None,
    on_checkpoint: Optional[callable] = None,
) -> EngineState:
    """
    Run the full fuzzing campaign. Returns final EngineState.

    on_checkpoint: async callable(corpus, chain_builder, state) — called every
                   checkpoint_interval iterations and on clean exit.
    """
    if state is None:
        state = EngineState()

    recent_results: list[IterResult] = []  # for live display

    with _make_live_display(config, state, recent_results) as live:
        # ---------------------------------------------------------------
        # Phase 1 — deterministic bootstrap
        # ---------------------------------------------------------------
        if state.phase == "phase1":
            await _run_phase1(
                config, corpus, chain_builder, composer, scorer, target,
                state, recent_results, live, on_checkpoint,
            )
            state.phase = "phase2"

        # ---------------------------------------------------------------
        # Phase 2/3 — interleaved havoc + LLM-guided (stub for phase 3)
        # ---------------------------------------------------------------
        await _run_phase2(
            config, corpus, chain_builder, composer, scorer, target,
            state, recent_results, live, on_checkpoint,
        )

    # Final checkpoint
    if on_checkpoint:
        await on_checkpoint(corpus, chain_builder, state)

    return state


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------


async def _run_phase1(
    config, corpus, chain_builder, composer, scorer, target,
    state, recent_results, live, on_checkpoint,
):
    """
    Exhaust all single-transform mutations across all seeds.
    Deterministic tools applied directly; LLM strategies via composer.
    """
    sem = asyncio.Semaphore(config.max_concurrent)

    async def run_one(seed: str, chain: Chain, is_direct: bool, iter_idx: int):
        async with sem:
            t0 = time.monotonic()

            composer_cost = 0.0
            if is_direct:
                # Deterministic tool — apply directly, bypass composer
                tool_id = chain.tool_spec[0]
                t = TRANSFORMS.get(tool_id)
                if t is None or t.requires_llm:
                    return
                try:
                    mutated = await t.apply_async(seed, llm=None, context="")
                except Exception as exc:
                    logger.debug("Direct transform %r failed: %s", tool_id, exc)
                    return
                phase_tag = "phase1_tool"
            else:
                # LLM strategy — use composer
                mutated, composer_cost = await composer.generate_with_cost(
                    seed, chain, config.seed_intent, config.target_context or ""
                )
                state.budget_spent_usd += composer_cost
                if mutated is None:
                    return
                phase_tag = "phase1_strategy"

            # Send to target + score
            try:
                response = await target.send(mutated)
            except Exception as exc:
                logger.warning("Target error (phase1): %s", exc)
                return

            score, scorer_cost = await scorer.score_with_cost(
                config.seed_intent, mutated, response, config.success_criterion
            )
            state.budget_spent_usd += scorer_cost
            entry = make_entry(seed, chain, mutated, score, depth=0)
            admitted = await corpus.add(entry)
            crashed = score.score == 3

            if crashed:
                state.crashes_found += 1

            result = IterResult(
                iteration=iter_idx, phase=phase_tag, seed_prompt=seed,
                chain=chain, mutated_prompt=mutated, response=response,
                score=score, admitted=admitted, crashed=crashed,
                elapsed_s=time.monotonic() - t0,
            )
            _record_result(result, recent_results, state)
            _refresh_live(live, config, state, recent_results)

            if on_checkpoint and state.iteration % config.checkpoint_interval == 0:
                await on_checkpoint(corpus, chain_builder, state)

            if crashed and config.stop_on_first_crash:
                return

    tasks = []
    iter_idx = state.iteration

    for seed in config.seed_prompts:
        # Single-tool chains (deterministic, bypass composer)
        for chain in chain_builder.all_single_tool_chains():
            tasks.append(run_one(seed, chain, is_direct=True, iter_idx=iter_idx))
            iter_idx += 1

        # Single-strategy chains (LLM-based, use composer)
        for chain in chain_builder.all_single_strategy_chains():
            tasks.append(run_one(seed, chain, is_direct=False, iter_idx=iter_idx))
            iter_idx += 1

    # Run all phase 1 tasks concurrently (bounded by semaphore)
    await asyncio.gather(*tasks, return_exceptions=True)
    state.iteration = iter_idx
    logger.info("Phase 1 complete: %d mutations, corpus size=%d", iter_idx, corpus.size())


# ---------------------------------------------------------------------------
# Phase 2/3
# ---------------------------------------------------------------------------


async def _run_phase2(
    config, corpus, chain_builder, composer, scorer, target,
    state, recent_results, live, on_checkpoint,
):
    """
    Random havoc (phase 2) interleaved with LLM-guided (phase 3).

    Phase 3 is triggered when non_novel_streak >= plateau_threshold:
      - Fetch top-k corpus entries
      - Ask composer to propose a chain based on score history
      - Use the proposed chain (or fall back to random if proposal fails)
      - Reset non_novel_streak so phase 2 resumes after any novel discovery

    Workers run concurrently bounded by max_concurrent semaphore.
    """
    sem = asyncio.Semaphore(config.max_concurrent)

    async def run_one_iteration():
        async with sem:
            if _should_stop(config, corpus, state):
                return

            t0 = time.monotonic()

            # Pick parent and seed
            parent = await corpus.pick_parent()
            if parent is not None:
                seed = parent.seed_prompt
                depth = parent.depth + 1
                parent_id = parent.entry_id
            else:
                seed = config.seed_prompts[0]
                depth = 0
                parent_id = None

            # Phase selection: random havoc (2) vs LLM-guided (3)
            if state.non_novel_streak >= config.plateau_threshold:
                # Phase 3 — ask the composer to propose a chain
                state.phase = "phase3"
                top_k = await corpus.get_top_k(10)
                chain = await chain_builder.llm_propose(
                    composer=composer,
                    top_k_entries=top_k,
                    seed_intent=config.seed_intent,
                    target_context=config.target_context or "",
                )
                # Reset streak regardless — avoid hammering the proposal API
                state.non_novel_streak = 0
                logger.info(
                    "Phase 3 proposal: %s",
                    f"strategies={chain.strategies} tools={chain.tool_spec}" if chain else "None → random",
                )
            else:
                state.phase = "phase2"
                chain = chain_builder.random_chain()

            if chain is None:
                state.non_novel_streak += 1
                state.iteration += 1
                return

            # Generate mutated prompt
            mutated, composer_cost = await composer.generate_with_cost(
                seed, chain, config.seed_intent, config.target_context or ""
            )
            state.budget_spent_usd += composer_cost
            if mutated is None:
                # Annotation failure — discard
                chain_builder.update(chain, admitted=False, crashed=False)
                state.non_novel_streak += 1
                state.iteration += 1
                return

            # Target + scorer
            try:
                response = await target.send(mutated)
            except Exception as exc:
                logger.warning("Target error: %s", exc)
                state.non_novel_streak += 1
                state.iteration += 1
                return

            score, scorer_cost = await scorer.score_with_cost(
                config.seed_intent, mutated, response, config.success_criterion
            )
            state.budget_spent_usd += scorer_cost

            entry = make_entry(seed, chain, mutated, score, depth, parent_id)
            admitted = await corpus.add(entry)
            crashed = score.score == 3

            # Update bandit weights
            chain_builder.update(chain, admitted=admitted, crashed=crashed)

            if admitted:
                state.non_novel_streak = 0
                if crashed:
                    state.crashes_found += 1
            else:
                state.non_novel_streak += 1

            state.iteration += 1

            result = IterResult(
                iteration=state.iteration, phase=state.phase, seed_prompt=seed,
                chain=chain, mutated_prompt=mutated, response=response,
                score=score, admitted=admitted, crashed=crashed,
                elapsed_s=time.monotonic() - t0,
            )
            _record_result(result, recent_results, state)
            _refresh_live(live, config, state, recent_results)

            # Periodic maintenance
            if state.iteration % config.checkpoint_interval == 0 and on_checkpoint:
                await on_checkpoint(corpus, chain_builder, state)
            if state.iteration % 100 == 0:
                await corpus.cull()

    # Main loop: keep spawning workers until stop condition
    while not _should_stop(config, corpus, state):
        # Fill a batch of up to max_concurrent concurrent workers
        batch = [
            asyncio.create_task(run_one_iteration())
            for _ in range(config.max_concurrent)
            if not _should_stop(config, corpus, state)
        ]
        if not batch:
            break
        await asyncio.gather(*batch, return_exceptions=True)


# ---------------------------------------------------------------------------
# Stop condition
# ---------------------------------------------------------------------------


def _should_stop(config, corpus: Corpus, state: EngineState) -> bool:
    if state.iteration >= config.max_iterations:
        return True
    if state.budget_spent_usd >= config.budget_usd:
        return True
    if config.stop_on_first_crash and corpus.has_crash():
        return True
    return False


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------


def _record_result(
    result: IterResult,
    recent: list[IterResult],
    state: EngineState,
) -> None:
    recent.append(result)
    if len(recent) > 20:
        recent.pop(0)


# ---------------------------------------------------------------------------
# Rich live display
# ---------------------------------------------------------------------------


def _make_live_display(config, state, recent_results):
    """Return a Rich Live context manager."""
    return Live(
        _render_display(config, state, recent_results),
        console=console,
        refresh_per_second=4,
        transient=False,
    )


def _refresh_live(live: Live, config, state: EngineState, recent: list[IterResult]) -> None:
    live.update(_render_display(config, state, recent))


def _render_display(config, state: EngineState, recent: list[IterResult]) -> Table:
    """Build the Rich renderable for the live display."""
    outer = Table.grid(padding=0)

    # Header bar
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    header.add_column(ratio=1)
    budget_pct = (state.budget_spent_usd / config.budget_usd * 100) if config.budget_usd > 0 else 0
    header.add_row(
        f"[bold cyan]llm-fuzzer[/] · {config.seed_intent[:60]}",
        f"iter [bold]{state.iteration}[/]/{config.max_iterations}  "
        f"phase [bold]{state.phase}[/]  "
        f"${state.budget_spent_usd:.3f}/${config.budget_usd:.2f} "
        f"[dim]({budget_pct:.0f}%)[/]",
        f"crashes [bold {'red' if state.crashes_found else 'white'}]"
        f"{state.crashes_found}[/]  "
        f"streak [bold]{state.non_novel_streak}[/]/{config.plateau_threshold}",
    )
    outer.add_row(header)
    outer.add_row("")

    # Results table
    tbl = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim",
        expand=True,
    )
    tbl.add_column("#",       width=5)
    tbl.add_column("score",   width=7)
    tbl.add_column("cat",     width=20)
    tbl.add_column("chain",   ratio=1)
    tbl.add_column("status",  width=10)

    for r in reversed(recent[-12:]):
        score_val = r.score.score if r.score else "?"
        score_color = {3: "bold red", 2: "yellow", 1: "cyan", 0: "dim"}.get(
            score_val if isinstance(score_val, int) else -1, "dim"
        )
        chain_str = _chain_summary(r.chain)
        cat = r.score.category if r.score else "error"

        if r.crashed:
            status = Text("CRASH", style="bold red on white")
        elif r.admitted:
            status = Text("admitted", style="green")
        elif r.mutated_prompt is None:
            status = Text("discard", style="dim red")
        else:
            status = Text("skip", style="dim")

        tbl.add_row(
            str(r.iteration),
            Text(str(score_val), style=score_color),
            Text(cat, style="dim"),
            chain_str,
            status,
        )

    outer.add_row(tbl)
    return outer


def _chain_summary(chain: Chain) -> str:
    parts = []
    if chain.strategies:
        parts.append("+".join(s.split("-")[-1] for s in chain.strategies))
    if chain.tool_spec:
        parts.append("{" + ",".join(t.split("-")[0] for t in chain.tool_spec) + "}")
    return " ".join(parts) if parts else "∅"
