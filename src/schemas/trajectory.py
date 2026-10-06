"""Pydantic schemas for agent trajectories, steps, and tool interactions."""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StepType(str, Enum):
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"


class FailureCategory(str, Enum):
    NONE = "none"
    UNAUTHORIZED_ACCESS = "unauthorized_access"  # Attempted private action without auth
    TOOL_HALLUCINATION = "tool_hallucination"    # Fabricated data without calling tool
    ARGUMENT_INVALID = "argument_invalid"        # Malformed or impossible parameters
    REDUNDANT_TOOL_CALL = "redundant_tool_call"  # Repeating identical query
    PREMATURE_TERMINATION = "premature_termination" # Stopped before answering prompt
    POLICY_VIOLATION = "policy_violation"        # Violating business rules (e.g. transfer limit)
    SCHEMA_VIOLATION = "schema_violation"        # Output failed JSON/tool-call parsing


class ToolCallRecord(BaseModel):
    call_id: str = Field(..., description="Unique ID of tool invocation")
    name: str = Field(..., description="Name of tool called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments passed to tool")
    raw_arguments: Optional[str] = Field(default=None, description="Raw JSON string from model")


class ToolResultRecord(BaseModel):
    call_id: str = Field(..., description="ID matching the tool call")
    name: str = Field(..., description="Name of tool")
    output: Any = Field(..., description="Returned result from environment")
    is_error: bool = Field(default=False, description="Whether tool execution threw error")


class AgentStep(BaseModel):
    step_number: int = Field(..., description="1-indexed step in the execution trace")
    step_type: StepType = Field(..., description="Classification of this step")
    thought: Optional[str] = Field(default=None, description="Internal reasoning or rationale")
    tool_calls: List[ToolCallRecord] = Field(default_factory=list, description="Tool calls generated")
    tool_results: List[ToolResultRecord] = Field(default_factory=list, description="Tool outputs observed")
    content: Optional[str] = Field(default=None, description="Message text or response content")
    latency_ms: float = Field(default=0.0, description="Execution time for this step in ms")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentTrajectory(BaseModel):
    trajectory_id: str = Field(..., description="Unique ID for this run")
    user_query: str = Field(..., description="Original user prompt or task")
    system_prompt: Optional[str] = Field(default=None, description="System instructions provided")
    steps: List[AgentStep] = Field(default_factory=list, description="Ordered trace of steps")
    final_response: Optional[str] = Field(default=None, description="Final output delivered to user")
    total_steps: int = Field(default=0, description="Number of steps taken")
    total_duration_ms: float = Field(default=0.0, description="Total runtime in ms")
    is_success: Optional[bool] = Field(default=None, description="Ground truth outcome verification")
    failure_category: FailureCategory = Field(default=FailureCategory.NONE)
    tags: List[str] = Field(default_factory=list)
