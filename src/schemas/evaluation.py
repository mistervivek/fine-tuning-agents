"""Schemas for evaluation benchmarks, metrics, and failure diagnostics."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.schemas.trajectory import FailureCategory


class EvaluationCase(BaseModel):
    case_id: str
    category: str
    user_query: str
    expected_tools: List[str] = Field(default_factory=list)
    forbidden_tools: List[str] = Field(default_factory=list)
    requires_auth: bool = False
    ground_truth_assertions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaseEvaluationResult(BaseModel):
    case_id: str
    is_success: bool
    failure_category: FailureCategory
    tool_precision: float
    tool_recall: float
    auth_compliance: bool
    hallucination_detected: bool
    step_count: int
    duration_ms: float
    error_message: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkSummary(BaseModel):
    total_cases: int
    success_rate: float
    mean_steps: float
    mean_duration_ms: float
    tool_f1_score: float
    auth_compliance_rate: float
    hallucination_rate: float
    failure_distribution: Dict[str, int] = Field(default_factory=dict)
    detailed_results: List[CaseEvaluationResult] = Field(default_factory=list)
