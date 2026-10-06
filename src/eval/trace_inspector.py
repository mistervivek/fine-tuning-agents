"""Trace inspector and diagnostic visualizer for agent trajectories."""

import sys
import json
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from src.schemas.trajectory import AgentTrajectory, StepType, FailureCategory


class TraceInspector:
    """Renders execution traces into human-readable tables, inspecting failure modes and outputs."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console(legacy_windows=False)

    def load_trajectory_from_file(self, file_path: str) -> AgentTrajectory:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AgentTrajectory.model_validate(data)

    def render_trajectory(self, trajectory: AgentTrajectory) -> None:
        """Render a full execution trace with step-level telemetry and failure diagnosis."""
        # 1. Trajectory Header
        status_color = "green" if trajectory.is_success else "red"
        status_symbol = "✅ PASSED" if trajectory.is_success else "❌ FAILED"
        if trajectory.is_success is None:
            status_symbol = "ℹ️ UNVERIFIED"
            status_color = "cyan"

        header_text = Text()
        header_text.append(f"Trajectory ID: {trajectory.trajectory_id}\n", style="bold cyan")
        header_text.append(f"User Query: ", style="bold")
        header_text.append(f'"{trajectory.user_query}"\n', style="italic white")
        header_text.append(f"Status: {status_symbol}  |  Steps: {trajectory.total_steps}  |  Runtime: {trajectory.total_duration_ms:.1f}ms\n", style=status_color)
        if trajectory.failure_category != FailureCategory.NONE:
            header_text.append(f"Failure Category: {trajectory.failure_category.value}\n", style="bold red")

        self.console.print(Panel(header_text, title="Agent Execution Trace", border_style=status_color))

        # 2. Steps Table
        table = Table(title="Step Breakdown", show_header=True, header_style="bold magenta")
        table.add_column("Step", justify="center", style="dim", width=6)
        table.add_column("Type", justify="center", width=12)
        table.add_column("Tool / Action", style="cyan", width=22)
        table.add_column("Details / Arguments / Observation", style="white")

        for s in trajectory.steps:
            type_str = s.step_type.value.upper()
            action_str = "-"
            details_str = ""

            if s.step_type == StepType.TOOL_CALL:
                names = [tc.name for tc in s.tool_calls]
                action_str = ", ".join(names)
                args = [json.dumps(tc.arguments) for tc in s.tool_calls]
                details_str = f"Args: {'; '.join(args)}"
                if s.thought:
                    details_str = f"Thought: {s.thought}\n{details_str}"

            elif s.step_type == StepType.TOOL_RESULT:
                names = [tr.name for tr in s.tool_results]
                action_str = ", ".join(names)
                outputs = [json.dumps(tr.output) if isinstance(tr.output, (dict, list)) else str(tr.output) for tr in s.tool_results]
                details_str = f"Obs: {'; '.join(outputs)[:140]}..."

            elif s.step_type == StepType.FINAL_RESPONSE:
                action_str = "Agent Output"
                details_str = (s.content or "")[:180] + ("..." if len(s.content or "") > 180 else "")

            elif s.step_type == StepType.ERROR:
                action_str = "Error"
                details_str = s.content or "Unknown error"

            table.add_row(str(s.step_number), type_str, action_str, details_str)

        self.console.print(table)

        # 3. Final Answer Panel
        if trajectory.final_response:
            self.console.print(
                Panel(
                    trajectory.final_response,
                    title="Delivered Final Response",
                    border_style="green" if trajectory.is_success else "yellow",
                )
            )
