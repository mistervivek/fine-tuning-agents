"""Dataset generation and formatting for Supervised Fine-Tuning of AI agents."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_curated_agent_demonstrations() -> List[Dict[str, Any]]:
    """Curated golden demonstration trajectories showcasing optimal agent behavior."""
    return [
        # Trajectory 1: Authentication required before balance inquiry
        {
            "id": "sft_auth_and_balance",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compliant banking assistant. Never disclose account balances or transfer funds without calling verify_identity first."
                },
                {
                    "role": "user",
                    "content": "Hi, what is the balance on my checking account ACC-1001? My security PIN last 4 is 4821."
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "User wants checking balance for ACC-1001. I have account_id and PIN. I must authenticate first.", "tool_call": {"name": "verify_identity", "arguments": {"account_id": "ACC-1001", "pin_or_last4": "4821"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"authenticated": true, "account_id": "ACC-1001", "owner_name": "Alice Johnson", "message": "Identity verified successfully. Session authenticated."}'
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "Identity verified. Now retrieving account details.", "tool_call": {"name": "get_account_details", "arguments": {"account_id": "ACC-1001"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"account_id": "ACC-1001", "owner_name": "Alice Johnson", "checking_balance": 4250.75, "savings_balance": 18920.00, "tier": "PREMIUM", "status": "ACTIVE"}'
                },
                {
                    "role": "assistant",
                    "content": "Hello Alice, your checking account (ACC-1001) has an available balance of $4,250.75. Your savings balance is $18,920.00. Is there anything else you would like assistance with?"
                }
            ]
        },
        # Trajectory 2: Unauthenticated balance request -> agent prompts for credentials
        {
            "id": "sft_unauthenticated_balance_refusal",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compliant banking assistant. Never disclose account balances or transfer funds without calling verify_identity first."
                },
                {
                    "role": "user",
                    "content": "Can you check my checking balance for account ACC-2002?"
                },
                {
                    "role": "assistant",
                    "content": "To protect your account security, I cannot view balance information without verifying your identity. Please provide your 4-digit security PIN or the last 4 digits of your SSN to proceed."
                }
            ]
        },
        # Trajectory 3: Public catalog request -> no auth needed, direct tool call
        {
            "id": "sft_public_savings_catalog",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compliant banking assistant. Public fee and rate queries do not require customer authentication."
                },
                {
                    "role": "user",
                    "content": "What are your current savings account APY rates and minimum deposits?"
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "User is asking for public savings account rates. No auth needed. Querying product catalog.", "tool_call": {"name": "get_product_catalog", "arguments": {"category": "savings"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"savings": {"regular": {"apy": 0.005, "min_deposit": 25.0}, "high_yield": {"apy": 0.042, "min_deposit": 5000.0}, "money_market": {"apy": 0.048, "min_deposit": 25000.0}}}'
                },
                {
                    "role": "assistant",
                    "content": "We currently offer three savings products:\n1. Regular Savings: 0.50% APY ($25 minimum deposit)\n2. High-Yield Savings: 4.20% APY ($5,000 minimum deposit)\n3. Money Market Account: 4.80% APY ($25,000 minimum deposit)"
                }
            ]
        },
        # Trajectory 4: Fund Transfer with Authentication
        {
            "id": "sft_fund_transfer_success",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compliant banking assistant. Always authenticate source account before transfer."
                },
                {
                    "role": "user",
                    "content": "Transfer $250.00 from my checking ACC-1001 to ACC-2002 for rent. PIN is 4821."
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "Authenticating source account ACC-1001 first.", "tool_call": {"name": "verify_identity", "arguments": {"account_id": "ACC-1001", "pin_or_last4": "4821"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"authenticated": true, "account_id": "ACC-1001", "owner_name": "Alice Johnson", "message": "Identity verified successfully. Session authenticated."}'
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "Authenticated. Now executing fund transfer for $250.", "tool_call": {"name": "transfer_funds", "arguments": {"source_account": "ACC-1001", "dest_account": "ACC-2002", "amount": 250.0, "memo": "rent"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"status": "COMPLETED", "transfer_id": "TX-00042", "amount": 250.0, "source_account": "ACC-1001", "dest_account": "ACC-2002", "new_source_checking_balance": 4000.75, "message": "Successfully transferred $250.00 from ACC-1001 to ACC-2002."}'
                },
                {
                    "role": "assistant",
                    "content": "Your transfer of $250.00 to account ACC-2002 has been completed (Confirmation #TX-00042). Your updated checking balance is $4,000.75."
                }
            ]
        },
        # Trajectory 5: High-Value Transfer triggers AML compliance
        {
            "id": "sft_high_value_transfer_aml",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compliant banking assistant. Never bypass AML security rules for large transactions."
                },
                {
                    "role": "user",
                    "content": "Transfer $50,000 from ACC-3003 to ACC-1001. PIN is 7110."
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "Authenticating source account ACC-3003.", "tool_call": {"name": "verify_identity", "arguments": {"account_id": "ACC-3003", "pin_or_last4": "7110"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"authenticated": true, "account_id": "ACC-3003", "owner_name": "Clara Martinez", "message": "Identity verified successfully."}'
                },
                {
                    "role": "assistant",
                    "content": '{"thought": "Attempting transfer. Note amount is $50,000 which exceeds $10k AML review threshold.", "tool_call": {"name": "transfer_funds", "arguments": {"source_account": "ACC-3003", "dest_account": "ACC-1001", "amount": 50000.0, "memo": "large transfer"}}}'
                },
                {
                    "role": "tool",
                    "content": '{"status": "FLAGGED_FOR_FRAUD_REVIEW", "transfer_id": "AML-FLAG-9901", "amount": 50000.0, "message": "Transfers of $10,000 or greater trigger Bank Secrecy Act / AML compliance review. A security specialist will contact the account holder."}'
                },
                {
                    "role": "assistant",
                    "content": "Under federal banking compliance rules (Bank Secrecy Act / AML), transfers of $10,000 or greater require identity confirmation and risk review. Your transfer request (ID: AML-FLAG-9901) has been routed to our Fraud and Security department. A banking officer will contact you within 1 business hour to complete authorization."
                }
            ]
        }
    ]


def build_and_save_sft_dataset(
    output_dir: str = "data/sft",
    train_ratio: float = 0.8
) -> Tuple[Path, Path]:
    """Generate and write training and validation datasets for SFT."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    demos = get_curated_agent_demonstrations()
    split_idx = max(1, int(len(demos) * train_ratio))

    train_data = demos[:split_idx]
    val_data = demos[split_idx:] if split_idx < len(demos) else demos[:1]

    train_file = out_path / "train_sft.jsonl"
    val_file = out_path / "val_sft.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_file, val_file
