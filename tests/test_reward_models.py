"""Unit tests for OutcomeRewardModel and ProcessRewardModel."""

from src.reward.outcome_reward import OutcomeRewardModel
from src.reward.process_reward import ProcessRewardModel
from src.preference.dpo_builder import build_contrastive_preference_pairs


def test_outcome_and_process_reward_models():
    pairs = build_contrastive_preference_pairs()
    pair = pairs[0]  # Tool hallucination pair

    orm = OutcomeRewardModel()
    prm = ProcessRewardModel()

    # Chosen trajectory should receive higher reward than rejected
    orm_chosen = orm.score(pair.chosen_trajectory)
    orm_rejected = orm.score(pair.rejected_trajectory)

    assert orm_chosen["total_reward"] > orm_rejected["total_reward"]
    assert orm_chosen["is_acceptable"] is True

    # PRM step evaluation
    prm_chosen = prm.score_trajectory(pair.chosen_trajectory)
    prm_rejected = prm.score_trajectory(pair.rejected_trajectory)

    assert prm_chosen["cumulative_return"] > prm_rejected["cumulative_return"]
    assert len(prm_chosen["step_scores"]) == len(pair.chosen_trajectory.steps)
