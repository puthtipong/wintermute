#!/usr/bin/env python3
"""
fuzz.py — CLI entrypoint for llm-fuzzer.

Commands:
  run    campaign.yaml [overrides]   Run a fuzzing campaign
  show   ./runs/run-001              Pretty-print crashes + score distribution
  resume ./runs/run-001              Resume from checkpoint

Usage examples:
  python fuzz.py run campaign.yaml
  python fuzz.py run campaign.yaml --budget 2.0 --max-iterations 100
  python fuzz.py run campaign.yaml --dry-run
  python fuzz.py show ./runs/run-001
  python fuzz.py resume ./runs/run-001
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

app = typer.Typer(
    name="fuzz",
    help="AFL-inspired prompt fuzzer for LLM-backed pipelines.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    config_file: Path = typer.Argument(..., help="Path to campaign YAML file"),
    budget: Optional[float]  = typer.Option(None,  "--budget",         help="Override budget_usd"),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Override max_iterations"),
    output: Optional[Path]   = typer.Option(None,  "--output",         help="Override output_dir"),
    stop_on_crash: bool      = typer.Option(False, "--stop-on-crash",  help="Stop on first score-3"),
    dry_run: bool            = typer.Option(False, "--dry-run",        help="Phase 1 first seed only, no target calls"),
    verbose: bool            = typer.Option(False, "--verbose",        help="Force verbose output"),
    quiet: bool              = typer.Option(False, "--quiet",          help="Force minimal output"),
):
    """Run a fuzzing campaign from a YAML config file."""
    _setup_logging(verbose=verbose, quiet=quiet)

    from fuzzer.campaign import load_config, run_campaign

    if not config_file.exists():
        console.print(f"[red]Error:[/] Config file not found: {config_file}")
        raise typer.Exit(1)

    overrides: dict = {}
    if budget is not None:
        overrides["budget_usd"] = budget
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if output is not None:
        overrides["output_dir"] = str(output)
    if stop_on_crash:
        overrides["stop_on_first_crash"] = True
    if verbose:
        overrides["verbosity"] = "verbose"
    if quiet:
        overrides["verbosity"] = "minimal"

    try:
        config = load_config(config_file, overrides or None)
    except Exception as exc:
        console.print(f"[red]Config error:[/] {exc}")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]Dry-run mode:[/] phase 1, first seed only, no target calls")
        config.max_iterations = 0          # skip phase 2
        config.seed_prompts = config.seed_prompts[:1] if config.seed_prompts else []
        # TODO: wire dry-run flag into engine to skip actual target.send()

    console.print(Panel(
        f"[bold cyan]llm-fuzzer[/]\n"
        f"intent:  {config.seed_intent}\n"
        f"target:  {config.target.endpoint} · {config.target.model or '(http)'}\n"
        f"budget:  ${config.budget_usd}  ·  {config.max_iterations} iterations  ·  "
        f"{config.max_concurrent} concurrent\n"
        f"output:  {config.output_dir}",
        title="Campaign",
        border_style="cyan",
    ))

    try:
        final_state = asyncio.run(run_campaign(config, resume=False))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user — checkpoint saved.[/]")
        raise typer.Exit(0)
    except Exception as exc:
        console.print(f"[red]Campaign error:[/] {exc}")
        logger = logging.getLogger(__name__)
        logger.exception("Campaign failed")
        raise typer.Exit(1)

    # Final summary banner
    console.print()
    if final_state.crashes_found:
        console.print(Panel(
            f"[bold red]🎯 {final_state.crashes_found} crash(es) found![/]\n"
            f"Iterations: {final_state.iteration}  ·  "
            f"See [bold]{config.output_dir}/crashes/[/]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            f"Campaign complete — no crashes found\n"
            f"Iterations: {final_state.iteration}  ·  "
            f"Corpus size: see [bold]{config.output_dir}/run_summary.json[/]",
            border_style="dim",
        ))


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(
    run_dir: Path = typer.Argument(..., help="Path to a run output directory"),
):
    """Pretty-print crashes and score distribution from a completed run."""
    if not run_dir.exists():
        console.print(f"[red]Error:[/] Directory not found: {run_dir}")
        raise typer.Exit(1)

    summary_path = run_dir / "run_summary.json"
    crashes_dir  = run_dir / "crashes"

    # Summary
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        console.print(Panel(
            f"[bold]Intent:[/] {summary.get('seed_intent', '?')}\n"
            f"Iterations: [bold]{summary.get('total_iterations', '?')}[/]  ·  "
            f"Corpus: [bold]{summary.get('corpus_size', '?')}[/] entries  ·  "
            f"Crashes: [bold {'red' if summary.get('crashes_found') else 'white'}]"
            f"{summary.get('crashes_found', 0)}[/]",
            title=str(run_dir),
            border_style="cyan",
        ))

        dist = summary.get("score_distribution", {})
        dist_tbl = Table(title="Score distribution", box=box.SIMPLE_HEAD)
        dist_tbl.add_column("Score", width=8)
        dist_tbl.add_column("Count", width=8)
        dist_tbl.add_column("Label", width=20)
        labels = {0: "hard refusal", 1: "soft refusal", 2: "partial", 3: "CRASH"}
        for score in range(4):
            count = dist.get(str(score), dist.get(score, 0))
            color = {3: "bold red", 2: "yellow", 1: "cyan", 0: "dim"}.get(score, "")
            dist_tbl.add_row(
                Text(str(score), style=color),
                Text(str(count), style=color),
                Text(labels[score], style=color),
            )
        console.print(dist_tbl)
    else:
        console.print("[dim]No run_summary.json found — campaign may still be running.[/]")

    # Crash files
    if crashes_dir.exists():
        crash_files = sorted(crashes_dir.glob("*.json"))
        if crash_files:
            console.print(f"\n[bold red]{len(crash_files)} crash(es):[/]\n")
            for cf in crash_files:
                entry = json.loads(cf.read_text())
                score_obj = entry.get("score", {})
                chain_obj = entry.get("chain", {})
                strategies = "+".join(chain_obj.get("strategies", [])) or "∅"
                tools = "{" + ",".join(chain_obj.get("tool_spec", [])) + "}" if chain_obj.get("tool_spec") else "∅"

                console.print(Panel(
                    f"[bold]chain:[/]   {strategies}  {tools}\n"
                    f"[bold]depth:[/]   {entry.get('depth', '?')}\n"
                    f"[bold]score:[/]   {score_obj.get('score')} · {score_obj.get('category')}\n"
                    f"[bold]reason:[/]  {score_obj.get('reasoning', '')}\n\n"
                    f"[bold]prompt:[/]\n{entry.get('mutated_prompt', '')[:800]}",
                    title=f"Crash {entry.get('entry_id', '')[:8]}",
                    border_style="red",
                ))
        else:
            console.print("[dim]No crash files in crashes/[/]")
    else:
        console.print("[dim]No crashes/ directory found.[/]")


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@app.command()
def resume(
    run_dir: Path   = typer.Argument(..., help="Path to a run output directory to resume"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="YAML config (defaults to run_dir/campaign.yaml)"),
    budget: Optional[float] = typer.Option(None, "--budget", help="Override budget_usd"),
):
    """Resume a campaign from a saved checkpoint."""
    _setup_logging()

    # Find config
    cfg_path = config_file or (run_dir / "campaign.yaml")
    if not cfg_path.exists():
        console.print(f"[red]Error:[/] No config found at {cfg_path}")
        console.print("Pass --config path/to/campaign.yaml explicitly.")
        raise typer.Exit(1)

    from fuzzer.campaign import load_config, run_campaign

    overrides: dict = {"output_dir": str(run_dir)}
    if budget is not None:
        overrides["budget_usd"] = budget

    try:
        config = load_config(cfg_path, overrides)
    except Exception as exc:
        console.print(f"[red]Config error:[/] {exc}")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Resuming[/] {run_dir}\n"
        f"intent: {config.seed_intent}",
        border_style="yellow",
    ))

    try:
        asyncio.run(run_campaign(config, resume=True))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — checkpoint saved.[/]")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.WARNING
    if verbose:
        level = logging.DEBUG
    elif not quiet:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
