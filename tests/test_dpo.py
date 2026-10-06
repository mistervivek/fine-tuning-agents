"""Unit tests for Direct Preference Optimization loss and pair synthesis."""

import torch
from src.preference.dpo_loss import compute_dpo_loss
from src.preference.dpo_builder import build_contrastive_preference_pairs, export_dpo_dataset


def test_dpo_loss_computation():
    # If policy favors chosen over rejected more than ref model, loss should be low
    pi_w = torch.tensor([-0.5, -0.6])
    pi_l = torch.tensor([-2.0, -2.5])
    ref_w = torch.tensor([-1.2, -1.3])
    ref_l = torch.tensor([-1.4, -1.5])

    loss, metrics = compute_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1)

    assert loss.item() > 0.0
    assert metrics["reward_margin"] > 0.0
    assert metrics["preference_accuracy"] == 1.0


def test_dpo_dataset_export(tmp_path):
    out_file = tmp_path / "dpo_test.jsonl"
    res_path = export_dpo_dataset(str(out_file))

    assert res_path.exists()
    pairs = build_contrastive_preference_pairs()
    assert len(pairs) >= 3
