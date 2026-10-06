"""LangGraph-powered agent with instrumented trajectory recording and environment integration."""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.environment.bank_env import BankEnvironment
from src.environment.tools import create_banking_tools
from src.agent.tracer import AgentTracer
from src.schemas.trajectory import AgentTrajectory, StepType

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are a compliant banking assistant. "
    "MANDATORY SECURITY RULES:\n"
    "1. Never disclose private account balances or transfer money without verifying identity first using 'verify_identity'.\n"
    "2. If identity is not verified, ask the user for their account ID and PIN/last-4 digits.\n"
    "3. Never hallucinate or invent account balances or transaction outcomes.\n"
    "4. For transfers of $10,000 or greater, warn the user of AML compliance review requirements."
)


def get_default_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Instantiate ChatOpenAI using environment variables (supports OpenRouter & OpenAI)."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model = model_name or os.getenv("MODEL_NAME", "qwen/qwen3.8-27b:free")

    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
    )


class BankingAgentRunner:
    """Executes queries against a LangGraph ReAct agent and generates full execution traces."""

    def __init__(
        self,
        env: Optional[BankEnvironment] = None,
        llm: Optional[ChatOpenAI] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        trace_dir: str = "artifacts/traces",
    ):
        self.env = env or BankEnvironment()
        self.tools = create_banking_tools(self.env)
        self.llm = llm or get_default_llm()
        self.system_prompt = system_prompt
        self.trace_dir = trace_dir

        # Initialize LangGraph React agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    def run(self, query: str, trajectory_id: Optional[str] = None) -> AgentTrajectory:
        """Run agent on query, trace every tool call and model reasoning step, return Trajectory."""
        tracer = AgentTracer(trajectory_id=trajectory_id, log_dir=self.trace_dir)
        tracer.start_trajectory(user_query=query, system_prompt=self.system_prompt)

        start_time = time.perf_counter()
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ]

        try:
            result = self.agent.invoke({"messages": messages})
            result_messages = result.get("messages", [])

            # Parse through messages to record structured trajectory steps
            final_content = ""
            for idx, msg in enumerate(result_messages):
                if isinstance(msg, AIMessage):
                    # Check for tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tracer.record_thought_or_step(
                            step_type=StepType.TOOL_CALL,
                            thought=msg.content if msg.content else None,
                            tool_calls=msg.tool_calls,
                            duration_ms=(time.perf_counter() - start_time) * 1000.0,
                        )
                    elif msg.content:
                        final_content = msg.content
                elif isinstance(msg, ToolMessage):
                    tracer.record_thought_or_step(
                        step_type=StepType.TOOL_RESULT,
                        tool_results=[{
                            "call_id": getattr(msg, "tool_call_id", ""),
                            "name": getattr(msg, "name", "tool"),
                            "output": msg.content,
                            "is_error": getattr(msg, "status", "") == "error",
                        }],
                    )

            if not final_content:
                # Extract last AIMessage if any
                for msg in reversed(result_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_content = msg.content
                        break

            tracer.record_thought_or_step(
                step_type=StepType.FINAL_RESPONSE,
                content=final_content,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
            )

            trajectory = tracer.finalize(final_response=final_content)
            return trajectory

        except Exception as e:
            tracer.record_thought_or_step(
                step_type=StepType.ERROR,
                content=f"Execution error: {str(e)}",
            )
            return tracer.finalize(
                final_response=f"Error executing agent: {str(e)}",
                is_success=False,
            )
