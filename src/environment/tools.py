"""LangChain tools bound to BankEnvironment instance."""

from __future__ import annotations
from typing import Any, Dict
from langchain_core.tools import tool
from src.environment.bank_env import BankEnvironment


def create_banking_tools(env: BankEnvironment):
    """Factory creating tools closures bound to a specific BankEnvironment instance."""

    @tool
    def verify_identity(account_id: str, pin_or_last4: str) -> Dict[str, Any]:
        """Authenticate customer before accessing private accounts or moving money.
        Args:
            account_id: The customer account identifier (e.g. 'ACC-1001')
            pin_or_last4: Security PIN or last 4 digits of SSN
        """
        return env.verify_identity(account_id=account_id, pin_or_last4=pin_or_last4)

    @tool
    def get_account_details(account_id: str) -> Dict[str, Any]:
        """Look up sensitive account balances (checking/savings) and account status.
        IMPORTANT: verify_identity MUST be called and succeed before calling this tool.
        Args:
            account_id: The customer account identifier (e.g. 'ACC-1001')
        """
        return env.get_account_details(account_id=account_id)

    @tool
    def transfer_funds(source_account: str, dest_account: str, amount: float, memo: str = "") -> Dict[str, Any]:
        """Initiate transfer of funds from source account to destination account.
        Requires prior authentication of source account. High amounts (> $10k) trigger AML review.
        Args:
            source_account: Sending account ID
            dest_account: Receiving account ID
            amount: Amount in USD (positive number)
            memo: Optional description or note
        """
        return env.transfer_funds(
            source_account=source_account,
            dest_account=dest_account,
            amount=amount,
            memo=memo
        )

    @tool
    def get_product_catalog(category: str = "all") -> Dict[str, Any]:
        """Get official bank product information, interest rates (APY), and fee schedules.
        Public info - does NOT require customer authentication.
        Args:
            category: 'checking', 'savings', 'wire_transfers', or 'all'
        """
        return env.get_product_catalog(category=category)

    return [verify_identity, get_account_details, transfer_funds, get_product_catalog]
