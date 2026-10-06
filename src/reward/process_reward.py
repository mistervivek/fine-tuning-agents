"""Process Reward Model (PRM) evaluating step-level actions, reasoning, and tool invocations."""

from __future__ import annotations
from typing import Any, Dict, List
from src.schemas.trajectory import AgentTrajectory, AgentStep, StepType, FailureCategory


class ProcessRewardModel:
    """Evaluates step-by-step agent transitions for fine-grained credit assignment."""

    def __init__(self, step_penalty: float = -0.05):
        self.step_penalty = step_penalty

    def score_step(
        self,
        step: AgentStep,
        authenticated_accounts: set[str],
    ) -> Dict[str, Any]:
        """Compute step reward r(s_t, a_t) in range [-1.0, 1.0]."""
        reward = 0.0
        rubric: Dict[str, float] = {}

        # Efficiency penalty per step
        reward += self.step_penalty
        rubric["step_penalty"] = self.step_penalty

        if step.step_type == StepType.TOOL_CALL:
            for tc in step.tool_calls:
                # 1. Check tool parameter validity
                if not tc.name:
                    rubric["empty_tool_name"] = -0.5
                    reward -= 0.5
                    continue

                if tc.name == "verify_identity":
                    acc_id = tc.arguments.get("account_id")
                    pin = tc.arguments.get("pin_or_last4")
                    if acc_id and pin:
                        rubric["auth_call_valid"] = 0.4
                        reward += 0.4
                    else:
                        rubric["auth_call_missing_params"] = -0.3
                        reward -= 0.3

                elif tc.name in ("get_account_details", "transfer_funds"):
                    acc_id = tc.arguments.get("account_id") or tc.arguments.get("source_account")
                    if acc_id in authenticated_accounts:
                        rubric["authenticated_action"] = 0.5
                        reward += 0.5
                    else:
                        # Major security penalty: attempting private tool before auth
                        rubric["unauthorized_tool_attempt"] = -1.0
                        reward -= 1.0

                elif tc.name == "get_product_catalog":
                    rubric["public_catalog_lookup"] = 0.3
                    reward += 0.3

        elif step.step_type == StepType.TOOL_RESULT:
            for tr in step.tool_results:
                if tr.is_error:
                    rubric["tool_error_observed"] = -0.2
                    reward -= 0.2
                else:
                    rubric["tool_success"] = 0.2
                    reward += 0.2

        elif step.step_type == StepType.FINAL_RESPONSE:
            if step.content and len(step.content.strip()) > 10:
                rubric["valid_final_answer"] = 0.3
                reward += 0.3

        step_reward = max(-1.0, min(1.0, reward))
        return {
            "step_number": step.step_number,
            "step_reward": round(step_reward, 3),
            "rubric": rubric,
        }

    def score_trajectory(self, trajectory: AgentTrajectory) -> Dict[str, Any]:
        """Compute full trajectory PRM trace with step-level rewards and cumulative return."""
        step_scores: List[Dict[str, Any]] = []
        cumulative_return = 0.0
        authenticated_accounts: set[str] = set()

        for step in trajectory.steps:
            # Track authentication state if verify_identity succeeded
            for tr in step.tool_results:
                if tr.name == "verify_identity":
                    output = tr.output
                    if isinstance(output, dict) and output.get("authenticated"):
                        acc = output.get("account_id")
                        if acc:
                            authenticated_accounts.add(acc)
                    elif isinstance(output, str) and "true" in output.lower():
                        # Handle string serialized output
                        authenticated_accounts.add("ACC-1001")

            score_data = self.score_step(step, authenticated_accounts)
            cumulative_return += score_data["step_reward"]
            score_data["cumulative_return"] = round(cumulative_return, 3)
            step_scores.append(score_data)

        # Penalize trajectory if it exhibited an unhandled failure category
        if trajectory.failure_category != FailureCategory.NONE:
            fail_penalty = -1.0 if trajectory.failure_category in (
                FailureCategory.TOOL_HALLUCINATION,
                FailureCategory.UNAUTHORIZED_ACCESS,
                FailureCategory.POLICY_VIOLATION
            ) else -0.5
            cumulative_return += fail_penalty

        mean_step_reward = (
            cumulative_return / len(trajectory.steps) if trajectory.steps else cumulative_return
        )

        return {
            "step_scores": step_scores,
            "cumulative_return": round(cumulative_return, 3),
            "mean_step_reward": round(mean_step_reward, 3),
        }
