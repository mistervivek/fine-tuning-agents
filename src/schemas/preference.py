"""Schemas for preference optimization pairs (chosen vs rejected) and DPO datasets."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.schemas.trajectory import AgentTrajectory, FailureCategory


class TrajectoryPair(BaseModel):
    pair_id: str = Field(..., description="Unique ID of comparison pair")
    prompt: str = Field(..., description="User prompt or task instruction")
    system_prompt: Optional[str] = Field(default=None)
    chosen_trajectory: AgentTrajectory = Field(..., description="The preferred (y_w) trajectory")
    rejected_trajectory: AgentTrajectory = Field(..., description="The dispreferred (y_l) trajectory")
    failure_category: FailureCategory = Field(
        ..., description="Specific failure mode exhibited by rejected trajectory"
    )
    rationale: str = Field(..., description="Research engineer reason for preference judgment")
    margin: float = Field(default=1.0, description="Preference confidence margin")


class DPOFormattedPair(BaseModel):
    """Ready-to-train format for Hugging Face TRL or custom DPO loss."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
