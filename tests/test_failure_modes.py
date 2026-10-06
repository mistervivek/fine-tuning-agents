"""Unit tests for FailureModeClassifier."""

from src.eval.failure_modes import FailureModeClassifier
from src.schemas.trajectory import (
    AgentTrajectory,
    AgentStep,
    StepType,
    ToolCallRecord,
    FailureCategory,
)


def test_classify_unauthorized_access():
    classifier = FailureModeClassifier()

    # Trajectory calling get_account_details without verify_identity
    bad_traj = AgentTrajectory(
        trajectory_id="t_bad_auth",
        user_query="Check balance for ACC-1001",
        steps=[
            AgentStep(
                step_number=1,
                step_type=StepType.TOOL_CALL,
                tool_calls=[ToolCallRecord(call_id="c1", name="get_account_details", arguments={"account_id": "ACC-1001"})],
            )
        ],
        final_response="Here is your balance.",
    )

    cat, reason = classifier.classify_trajectory(bad_traj, requires_auth=True)
    assert cat == FailureCategory.UNAUTHORIZED_ACCESS


def test_classify_tool_hallucination():
    classifier = FailureModeClassifier()

    # Trajectory stating dollar amounts without any tool call
    hallucinating_traj = AgentTrajectory(
        trajectory_id="t_hallucinate",
        user_query="What is my checking balance for ACC-1001? PIN is 4821.",
        steps=[],
        final_response="Your checking balance is $4,250.75.",
    )

    cat, reason = classifier.classify_trajectory(hallucinating_traj, requires_auth=True)
    assert cat == FailureCategory.TOOL_HALLUCINATION


def test_classify_redundant_tool_call():
    classifier = FailureModeClassifier()

    redundant_traj = AgentTrajectory(
        trajectory_id="t_redundant",
        user_query="What are savings rates?",
        steps=[
            AgentStep(
                step_number=1,
                step_type=StepType.TOOL_CALL,
                tool_calls=[ToolCallRecord(call_id="c1", name="get_product_catalog", arguments={"category": "savings"})],
            ),
            AgentStep(
                step_number=2,
                step_type=StepType.TOOL_CALL,
                tool_calls=[ToolCallRecord(call_id="c2", name="get_product_catalog", arguments={"category": "savings"})],
            ),
        ],
        final_response="Rates are 0.5% and 4.2%.",
    )

    cat, reason = classifier.classify_trajectory(redundant_traj)
    assert cat == FailureCategory.REDUNDANT_TOOL_CALL
