"""Preference optimization package (DPO, contrastive pair generation)."""

from src.preference.dpo_builder import (
    build_contrastive_preference_pairs,
    export_dpo_dataset,
)
from src.preference.dpo_loss import compute_dpo_loss

__all__ = [
    "build_contrastive_preference_pairs",
    "export_dpo_dataset",
    "compute_dpo_loss",
]
