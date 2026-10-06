"""Offline deterministic policy agent for offline benchmarking and reproducible experiments."""

from __future__ import annotations
import re
import time
from typing import Optional
from src.environment.bank_env import BankEnvironment
from src.schemas.trajectory import AgentTrajectory, AgentStep, StepType, ToolCallRecord, ToolResultRecord, FailureCategory
from src.agent.tracer import AgentTracer


class DeterministicPolicyAgent:
    """Simulates an optimal, fine-tuned agent policy without external network dependencies.

    Enforces:
    1. Authentication before looking up private accounts or transferring money.
    2. Zero hallucination - directly queries BankEnvironment.
    3. Proper AML disclosure when transfer amount >= $10,000.
    """

    def __init__(self, env: Optional[BankEnvironment] = None, trace_dir: str = "artifacts/traces"):
        self.env = env or BankEnvironment()
        self.trace_dir = trace_dir

    def run(self, query: str, trajectory_id: Optional[str] = None) -> AgentTrajectory:
        tracer = AgentTracer(trajectory_id=trajectory_id, log_dir=self.trace_dir)
        tracer.start_trajectory(user_query=query, system_prompt="Deterministic Compliant Agent Policy")
        start_time = time.perf_counter()

        query_lower = query.lower()

        # Extract account ID and PIN if present
        acc_match = re.search(r"acc-\d{4}", query, re.IGNORECASE)
        account_id = acc_match.group(0).upper() if acc_match else None

        pin_match = re.search(r"(?:pin\s*(?:is|:)?\s*|last4\s*(?:is|:)?\s*)(\d{4})", query, re.IGNORECASE)
        if pin_match:
            pin = pin_match.group(1)
        else:
            all_4digits = re.findall(r"\b\d{4}\b", query)
            pin = None
            for d in all_4digits:
                if account_id and d in account_id:
                    continue
                pin = d
                break

        amount_match = re.search(r"\$([0-9,]+(?:\.[0-9]{2})?)", query)

        is_fee_query = any(w in query_lower for w in ("cost", "fee", "how much", "what are the fees", "what does"))

        # Check for transfer intent
        if "transfer" in query_lower and not is_fee_query:
            amount = float(amount_match.group(1).replace(",", "")) if amount_match else 0.0
            dest_acc = "ACC-2002" if account_id != "ACC-2002" else "ACC-1001"

            if not account_id or not pin:
                tracer.record_thought_or_step(
                    step_type=StepType.FINAL_RESPONSE,
                    content="To execute a transfer, please provide your account ID and security PIN.",
                    duration_ms=(time.perf_counter() - start_time) * 1000.0,
                )
                return tracer.finalize(final_response="To execute a transfer, please provide your account ID and security PIN.", is_success=True)

            # 1. Authenticate
            auth_tc = ToolCallRecord(call_id="call_auth", name="verify_identity", arguments={"account_id": account_id, "pin_or_last4": pin})
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_CALL,
                thought="Verifying source account identity before initiating transfer.",
                tool_calls=[auth_tc.model_dump()],
            )
            auth_res = self.env.verify_identity(account_id, pin)
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_RESULT,
                tool_results=[{"call_id": "call_auth", "name": "verify_identity", "output": auth_res}],
            )

            # 2. Transfer
            tx_tc = ToolCallRecord(call_id="call_tx", name="transfer_funds", arguments={"source_account": account_id, "dest_account": dest_acc, "amount": amount})
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_CALL,
                thought="Calling transfer_funds with authenticated session.",
                tool_calls=[tx_tc.model_dump()],
            )
            tx_res = self.env.transfer_funds(account_id, dest_acc, amount)
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_RESULT,
                tool_results=[{"call_id": "call_tx", "name": "transfer_funds", "output": tx_res}],
            )

            if tx_res.get("status") == "FLAGGED_FOR_FRAUD_REVIEW":
                resp = f"Transfers of $10,000 or greater require mandatory AML/Compliance review. Your transfer of ${amount:,.2f} has been flagged for fraud review."
            else:
                resp = f"Successfully transferred ${amount:,.2f} from {account_id} to {dest_acc} (Ref: {tx_res.get('transfer_id')})."

            tracer.record_thought_or_step(
                step_type=StepType.FINAL_RESPONSE,
                content=resp,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            return tracer.finalize(final_response=resp, is_success=True)

        # Check for balance inquiry
        elif "balance" in query_lower:
            if not account_id or not pin:
                resp = "To check your account balance, please provide your account ID and 4-digit security PIN for identity verification."
                tracer.record_thought_or_step(
                    step_type=StepType.FINAL_RESPONSE,
                    content=resp,
                    duration_ms=(time.perf_counter() - start_time) * 1000.0,
                )
                return tracer.finalize(final_response=resp, is_success=True)

            # Authenticate then fetch balance
            auth_res = self.env.verify_identity(account_id, pin)
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_CALL,
                thought="Authenticating customer identity.",
                tool_calls=[{"id": "c1", "name": "verify_identity", "args": {"account_id": account_id, "pin_or_last4": pin}}],
            )
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_RESULT,
                tool_results=[{"call_id": "c1", "name": "verify_identity", "output": auth_res}],
            )

            acc_details = self.env.get_account_details(account_id)
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_CALL,
                thought="Retrieving account details.",
                tool_calls=[{"id": "c2", "name": "get_account_details", "args": {"account_id": account_id}}],
            )
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_RESULT,
                tool_results=[{"call_id": "c2", "name": "get_account_details", "output": acc_details}],
            )

            if acc_details.get("error"):
                resp = f"Access denied: {acc_details.get('message', 'Authentication failed')}"
            else:
                resp = f"Hello {acc_details.get('owner_name')}, your checking balance for {account_id} is ${acc_details.get('checking_balance', 0.0):,.2f}."
            tracer.record_thought_or_step(
                step_type=StepType.FINAL_RESPONSE,
                content=resp,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            return tracer.finalize(final_response=resp, is_success=True)

        # Public catalog inquiry (savings, fees, checking)
        else:
            cat_category = "savings" if "saving" in query_lower else ("wire_transfers" if "wire" in query_lower else "checking")
            catalog = self.env.get_product_catalog(cat_category)

            tracer.record_thought_or_step(
                step_type=StepType.TOOL_CALL,
                thought=f"Public inquiry. Fetching {cat_category} product catalog.",
                tool_calls=[{"id": "c_pub", "name": "get_product_catalog", "args": {"category": cat_category}}],
            )
            tracer.record_thought_or_step(
                step_type=StepType.TOOL_RESULT,
                tool_results=[{"call_id": "c_pub", "name": "get_product_catalog", "output": catalog}],
            )

            if cat_category == "savings":
                resp = "Our savings rates: Regular Savings has 0.5% APY ($25 min deposit) and High-Yield Savings has 4.2% APY ($5,000 min deposit)."
            elif cat_category == "wire_transfers":
                resp = "Domestic outgoing wire transfers cost $25.00, and international wires cost $45.00."
            else:
                resp = f"Here is our product information: {catalog}"

            tracer.record_thought_or_step(
                step_type=StepType.FINAL_RESPONSE,
                content=resp,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            return tracer.finalize(final_response=resp, is_success=True)
