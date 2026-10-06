"""Research Engineer CLI: End-to-end tooling for Agent Post-Training, SFT, DPO, and Evaluation."""

import sys
from pathlib import Path
from typing import Optional

# Ensure standard streams handle UTF-8 safely on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.agent.graph import BankingAgentRunner
from src.agent.mock_policy import DeterministicPolicyAgent
from src.eval.benchmark import AgentEvaluationHarness, get_standard_eval_suite
from src.eval.trace_inspector import TraceInspector
from src.sft.dataset import build_and_save_sft_dataset
from src.preference.dpo_builder import export_dpo_dataset, build_contrastive_preference_pairs
from src.preference.dpo_loss import compute_dpo_loss
from src.reward.outcome_reward import OutcomeRewardModel
from src.reward.process_reward import ProcessRewardModel
from src.distillation.trajectory_distiller import AgentDistillationPipeline
import torch

app = typer.Typer(
    name="agent-re",
    help="Agent Post-Training & Research Engineering Toolkit (SFT, DPO, Reward Modeling, Tracing)",
    add_completion=False,
)
console = Console(legacy_windows=False)
inspector = TraceInspector(console=console)


@app.command("run")
def run_agent(
    query: str = typer.Argument(..., help="User query for the banking agent"),
    offline: bool = typer.Option(False, "--offline", help="Run deterministic teacher policy without external API"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override"),
):
    """Execute a query through the agent and display the complete trace."""
    mode_label = "Offline Teacher Policy" if offline else "Live LangGraph Model"
    console.print(f"[bold cyan]Running Agent ({mode_label}) with query:[/bold cyan] {query}")
    if offline:
        runner = DeterministicPolicyAgent()
    else:
        runner = BankingAgentRunner()
    trajectory = runner.run(query)
    inspector.render_trajectory(trajectory)


@app.command("benchmark")
def run_benchmark(
    offline: bool = typer.Option(True, "--offline/--live", help="Use deterministic policy vs live LLM API"),
):
    """Run standardized evaluation suite and display comprehensive scorecard."""
    mode_str = "Offline Verified Policy" if offline else "Live LLM API"
    console.print(f"[bold yellow]Launching Agent Evaluation Benchmark Suite ({mode_str})...[/bold yellow]")
    runner = DeterministicPolicyAgent() if offline else BankingAgentRunner()
    harness = AgentEvaluationHarness(runner=runner)
    summary = harness.run_benchmark()

    # Scorecard Table
    scorecard = Table(title="Agent Benchmark Performance Scorecard", header_style="bold green")
    scorecard.add_column("Metric", style="bold")
    scorecard.add_column("Score", justify="right")

    scorecard.add_row("Total Test Cases", str(summary.total_cases))
    scorecard.add_row("Task Success Rate", f"{summary.success_rate:.1f}%")
    scorecard.add_row("Tool F1 Score", f"{summary.tool_f1_score:.2f}")
    scorecard.add_row("Auth / Security Compliance", f"{summary.auth_compliance_rate:.1f}%")
    scorecard.add_row("Hallucination Rate", f"{summary.hallucination_rate:.1f}%")
    scorecard.add_row("Mean Steps per Query", f"{summary.mean_steps:.2f}")
    scorecard.add_row("Mean Duration", f"{summary.mean_duration_ms:.1f}ms")

    console.print(scorecard)

    # Failure Distribution Table
    fail_table = Table(title="Failure Mode Breakdown", header_style="bold red")
    fail_table.add_column("Failure Category", style="cyan")
    fail_table.add_column("Count", justify="right")

    for cat, count in summary.failure_distribution.items():
        fail_table.add_row(cat, str(count))

    console.print(fail_table)


@app.command("sft-prepare")
def prepare_sft():
    """Build multi-turn agent demonstration dataset for Supervised Fine-Tuning."""
    console.print("[bold cyan]📦 Generating curated Agent SFT dataset...[/bold cyan]")
    train_file, val_file = build_and_save_sft_dataset()
    console.print(f"[green]✅ SFT Train set created:[/green] {train_file}")
    console.print(f"[green]✅ SFT Validation set created:[/green] {val_file}")


@app.command("dpo-generate")
def generate_dpo():
    """Synthesize contrastive preference pairs (chosen vs rejected) for agent DPO."""
    console.print("[bold cyan]⚖️ Synthesizing contrastive preference pairs...[/bold cyan]")
    out_path = export_dpo_dataset()
    console.print(f"[green]✅ DPO Dataset exported to:[/green] {out_path}")

    # Compute a quick DPO loss demonstration
    console.print("\n[bold magenta]Simulating DPO Loss with synthetic log-probs:[/bold magenta]")
    pi_w = torch.tensor([-0.8, -1.2])
    pi_l = torch.tensor([-2.5, -3.1])
    ref_w = torch.tensor([-1.5, -1.8])
    ref_l = torch.tensor([-1.6, -1.9])

    loss, metrics = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1)
    console.print(f"  • DPO Loss: {metrics['loss']:.4f}")
    console.print(f"  • Implicit Reward Margin: {metrics['reward_margin']:.4f}")
    console.print(f"  • Preference Accuracy: {metrics['preference_accuracy'] * 100:.1f}%")


@app.command("reward-score")
def score_rewards():
    """Score sample trajectories using Outcome Reward Model and Process Reward Model."""
    console.print("[bold cyan]🎯 Running Reward Models (ORM & PRM)...[/bold cyan]")
    pairs = build_contrastive_preference_pairs()
    orm = OutcomeRewardModel()
    prm = ProcessRewardModel()

    for idx, p in enumerate(pairs, 1):
        console.print(f"\n[bold yellow]Pair {idx}: {p.pair_id}[/bold yellow]")
        
        # Score Chosen
        orm_chosen = orm.score(p.chosen_trajectory)
        prm_chosen = prm.score_trajectory(p.chosen_trajectory)
        console.print(f"  [green]Chosen:[/green] ORM Reward = {orm_chosen['total_reward']}, PRM Cumulative = {prm_chosen['cumulative_return']}")

        # Score Rejected
        orm_rejected = orm.score(p.rejected_trajectory)
        prm_rejected = prm.score_trajectory(p.rejected_trajectory)
        console.print(f"  [red]Rejected ({p.failure_category.value}):[/red] ORM Reward = {orm_rejected['total_reward']}, PRM Cumulative = {prm_rejected['cumulative_return']}")


@app.command("distill")
def run_distillation():
    """Run verifier-based rejection sampling and trajectory distillation pipeline."""
    console.print("[bold cyan]🧪 Executing Trajectory Distillation Pipeline...[/bold cyan]")
    pipeline = AgentDistillationPipeline()
    pairs = build_contrastive_preference_pairs()
    candidates = [p.chosen_trajectory for p in pairs] + [p.rejected_trajectory for p in pairs]

    result = pipeline.process_and_export(candidates)
    console.print(f"[green]✅ Distillation complete![/green]")
    console.print(f"  • Total Candidates: {result['total_candidates']}")
    console.print(f"  • Accepted (Verified Clean): {result['accepted_count']}")
    console.print(f"  • Rejected (Failed Rubric): {result['rejected_count']}")
    console.print(f"  • Acceptance Rate: {result['acceptance_rate'] * 100:.1f}%")
    console.print(f"  • Exported Dataset: {result['output_path']}")


@app.command("inspect-trace")
def inspect_trace(trace_file: Path = typer.Argument(..., help="Path to trajectory JSON trace")):
    """Inspect and visualize a specific trajectory JSON file."""
    if not trace_file.exists():
        console.print(f"[bold red]File not found:[/bold red] {trace_file}")
        raise typer.Exit(1)
    traj = inspector.load_trajectory_from_file(str(trace_file))
    inspector.render_trajectory(traj)


if __name__ == "__main__":
    app()
