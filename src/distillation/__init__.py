"""Distillation package (Teacher-student trajectory extraction, rejection sampling)."""

from src.distillation.rejection_sampler import TrajectoryRejectionSampler
from src.distillation.trajectory_distiller import AgentDistillationPipeline

__all__ = ["TrajectoryRejectionSampler", "AgentDistillationPipeline"]
