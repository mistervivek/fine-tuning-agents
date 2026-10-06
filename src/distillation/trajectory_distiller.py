"""Trajectory distillation pipeline converting teacher executions to student training records."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from src.schemas.trajectory import AgentTrajectory, StepType
from src.distillation.rejection_sampler import TrajectoryRejectionSampler


class AgentDistillationPipeline:
    """Manages teacher trajectory collection, verifier filtering, and student dataset export."""

    def __init__(self, sampler: TrajectoryRejectionSampler | None = None):
        self.sampler = sampler or TrajectoryRejectionSampler()

    def convert_trajectory_to_sft_format(self, trajectory: AgentTrajectory) -> Dict[str, Any]:
        """Convert an execution trajectory into standard multi-turn chat messages with tool calls."""
        messages: List[Dict[str, str]] = []

        if trajectory.system_prompt:
            messages.append({"role": "system", "content": trajectory.system_prompt})

        messages.append({"role": "user", "content": trajectory.user_query})

        for step in trajectory.steps:
            if step.step_type == StepType.TOOL_CALL:
                # Format as structured thought + tool call
                payload: Dict[str, Any] = {}
                if step.thought:
                    payload["thought"] = step.thought
                if step.tool_calls:
                    tc = step.tool_calls[0]
                    payload["tool_call"] = {"name": tc.name, "arguments": tc.arguments}
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False)
                })

            elif step.step_type == StepType.TOOL_RESULT:
                for tr in step.tool_results:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tr.output, ensure_ascii=False)
                        if isinstance(tr.output, (dict, list))
                        else str(tr.output)
                    })

        if trajectory.final_response:
            messages.append({"role": "assistant", "content": trajectory.final_response})

        return {
            "distillation_id": f"distill_{trajectory.trajectory_id}",
            "messages": messages,
            "total_steps": len(trajectory.steps),
        }

    def process_and_export(
        self,
        candidate_trajectories: List[AgentTrajectory],
        output_path: str = "data/sft/distilled_student_sft.jsonl",
        expected_facts_map: dict[str, List[str]] | None = None,
    ) -> Dict[str, Any]:
        """Filter candidate teacher runs and save verified records for student training."""
        accepted, rejected = self.sampler.filter_trajectories(
            candidate_trajectories,
            expected_facts_map=expected_facts_map,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        student_records = [self.convert_trajectory_to_sft_format(t) for t in accepted]

        with open(out, "w", encoding="utf-8") as f:
            for rec in student_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return {
            "total_candidates": len(candidate_trajectories),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "acceptance_rate": round(len(accepted) / len(candidate_trajectories), 3) if candidate_trajectories else 0.0,
            "output_path": str(out),
            "rejected_reasons": [r[1] for r in rejected],
        }
