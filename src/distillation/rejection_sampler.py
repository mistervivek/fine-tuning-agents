"""Rejection sampling and verifier filtering for agent trajectory distillation."""

from __future__ import annotations
from typing import List, Tuple
from src.schemas.trajectory import AgentTrajectory, FailureCategory
from src.reward.outcome_reward import OutcomeRewardModel


class TrajectoryRejectionSampler:
    """Filters candidate teacher trajectories to select only verified, high-reward samples for distillation."""

    def __init__(self, min_reward_threshold: float = 0.6, max_allowed_steps: int = 8):
        self.min_reward_threshold = min_reward_threshold
        self.max_allowed_steps = max_allowed_steps
        self.orm = OutcomeRewardModel()

    def filter_trajectories(
        self,
        candidates: List[AgentTrajectory],
        expected_facts_map: dict[str, List[str]] | None = None,
    ) -> Tuple[List[AgentTrajectory], List[Tuple[AgentTrajectory, str]]]:
        """Separate candidate trajectories into accepted (clean) and rejected (flawed).

        Returns:
            (accepted_trajectories, rejected_with_reasons)
        """
        accepted: List[AgentTrajectory] = []
        rejected: List[Tuple[AgentTrajectory, str]] = []

        for traj in candidates:
            # Check 1: Step budget
            if len(traj.steps) > self.max_allowed_steps:
                rejected.append((traj, f"Exceeded max steps ({len(traj.steps)} > {self.max_allowed_steps})"))
                continue

            # Check 2: Severe failure categories
            if traj.failure_category != FailureCategory.NONE:
                rejected.append((traj, f"Failure category detected: {traj.failure_category.value}"))
                continue

            # Check 3: Outcome Reward Score
            facts = (expected_facts_map or {}).get(traj.trajectory_id)
            score_res = self.orm.score(traj, expected_facts=facts)
            if score_res["total_reward"] < self.min_reward_threshold:
                rejected.append(
                    (traj, f"Low outcome reward: {score_res['total_reward']} < {self.min_reward_threshold}")
                )
                continue

            # Passed all verifiers
            traj.is_success = True
            accepted.append(traj)

        return accepted, rejected
