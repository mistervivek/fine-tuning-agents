"""Mathematical implementation of Direct Preference Optimization (DPO) loss for Agent policies."""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute exact DPO loss, implicit rewards, and optimization metrics.

    Mathematical formulation (Rafailov et al., 2023):
        L_DPO = - log(sigmoid(beta * (log(pi(yw)/pi_ref(yw)) - log(pi(yl)/pi_ref(yl)))))

    Args:
        policy_chosen_logps: Log probabilities of chosen trajectories under current policy pi_theta
        policy_rejected_logps: Log probabilities of rejected trajectories under current policy pi_theta
        reference_chosen_logps: Log probabilities of chosen trajectories under reference model pi_ref
        reference_rejected_logps: Log probabilities of rejected trajectories under reference model pi_ref
        beta: Temperature hyperparameter controlling KL penalty / policy deviation
        label_smoothing: Optional conservative label smoothing factor

    Returns:
        loss: Scalar tensor representing average DPO loss
        metrics: Dictionary containing implicit rewards, reward margin, and preference accuracy
    """
    # 1. Compute policy log-ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # 2. Compute implicit rewards: r_hat = beta * (log pi(y) - log pi_ref(y))
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    reward_margins = chosen_rewards - rejected_rewards

    # 3. Compute DPO cross-entropy loss
    if label_smoothing == 0.0:
        losses = -F.logsigmoid(beta * logits)
    else:
        # Conservative DPO with label smoothing
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    loss = losses.mean()

    # 4. Compute telemetry metrics
    with torch.no_grad():
        accuracy = (reward_margins > 0).float().mean().item()
        mean_chosen_reward = chosen_rewards.mean().item()
        mean_rejected_reward = rejected_rewards.mean().item()
        mean_margin = reward_margins.mean().item()

    metrics = {
        "loss": loss.item(),
        "reward_margin": mean_margin,
        "chosen_reward": mean_chosen_reward,
        "rejected_reward": mean_rejected_reward,
        "preference_accuracy": accuracy,
    }

    return loss, metrics
