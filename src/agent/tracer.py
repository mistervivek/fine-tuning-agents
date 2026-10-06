"""Execution tracer that records step-level telemetry, tool calls, and outputs."""

from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.schemas.trajectory import (
    AgentStep,
    AgentTrajectory,
    StepType,
    ToolCallRecord,
    ToolResultRecord,
    FailureCategory,
)


class AgentTracer:
    """Instruments an agent execution session and builds structured AgentTrajectory."""

    def __init__(self, trajectory_id: Optional[str] = None, log_dir: str = "artifacts/traces"):
        self.trajectory_id = trajectory_id or f"traj_{uuid.uuid4().hex[:8]}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time: float = 0.0
        self.steps: List[AgentStep] = []
        self.user_query: str = ""
        self.system_prompt: Optional[str] = None
        self.final_response: Optional[str] = None

    def start_trajectory(self, user_query: str, system_prompt: Optional[str] = None) -> None:
        self.start_time = time.perf_counter()
        self.user_query = user_query
        self.system_prompt = system_prompt
        self.steps = []

    def record_thought_or_step(
        self,
        step_type: StepType,
        thought: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        content: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> AgentStep:
        step_idx = len(self.steps) + 1
        t_calls: List[ToolCallRecord] = []
        if tool_calls:
            for tc in tool_calls:
                t_calls.append(
                    ToolCallRecord(
                        call_id=tc.get("id", f"call_{len(t_calls)}"),
                        name=tc.get("name", ""),
                        arguments=tc.get("args", {}) if isinstance(tc.get("args"), dict) else {},
                        raw_arguments=str(tc.get("args", "")),
                    )
                )

        t_results: List[ToolResultRecord] = []
        if tool_results:
            for tr in tool_results:
                t_results.append(
                    ToolResultRecord(
                        call_id=tr.get("call_id", ""),
                        name=tr.get("name", ""),
                        output=tr.get("output", ""),
                        is_error=tr.get("is_error", False),
                    )
                )

        step = AgentStep(
            step_number=step_idx,
            step_type=step_type,
            thought=thought,
            tool_calls=t_calls,
            tool_results=t_results,
            content=content,
            latency_ms=duration_ms,
        )
        self.steps.append(step)
        return step

    def finalize(
        self,
        final_response: str,
        is_success: Optional[bool] = None,
        failure_category: FailureCategory = FailureCategory.NONE,
    ) -> AgentTrajectory:
        total_duration_ms = (time.perf_counter() - self.start_time) * 1000.0 if self.start_time > 0 else 0.0
        self.final_response = final_response

        traj = AgentTrajectory(
            trajectory_id=self.trajectory_id,
            user_query=self.user_query,
            system_prompt=self.system_prompt,
            steps=self.steps,
            final_response=final_response,
            total_steps=len(self.steps),
            total_duration_ms=total_duration_ms,
            is_success=is_success,
            failure_category=failure_category,
        )

        # Save to disk
        trace_file = self.log_dir / f"{self.trajectory_id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(traj.model_dump_json(indent=2))

        return traj
