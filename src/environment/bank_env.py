"""Simulated banking environment with deterministic state, security checks, and audit logging."""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional


DEFAULT_ACCOUNTS: Dict[str, Dict[str, Any]] = {
    "ACC-1001": {
        "owner_name": "Alice Johnson",
        "pin_last4": "4821",
        "checking_balance": 4250.75,
        "savings_balance": 18920.00,
        "status": "ACTIVE",
        "tier": "PREMIUM",
    },
    "ACC-2002": {
        "owner_name": "Bob Smith",
        "pin_last4": "9934",
        "checking_balance": 310.20,
        "savings_balance": 1500.00,
        "status": "ACTIVE",
        "tier": "BASIC",
    },
    "ACC-3003": {
        "owner_name": "Clara Martinez",
        "pin_last4": "7110",
        "checking_balance": 85000.00,
        "savings_balance": 240000.00,
        "status": "ACTIVE",
        "tier": "PRIVATE_WEALTH",
    },
}

PRODUCT_CATALOG: Dict[str, Any] = {
    "checking": {
        "basic": {"monthly_fee": 0.0, "min_balance": 0.0, "overdraft_fee": 35.0},
        "premium": {"monthly_fee": 12.0, "fee_waiver_min_balance": 2500.0, "overdraft_fee": 0.0},
        "student": {"monthly_fee": 0.0, "min_balance": 0.0, "overdraft_fee": 15.0},
    },
    "savings": {
        "regular": {"apy": 0.005, "min_deposit": 25.0},
        "high_yield": {"apy": 0.042, "min_deposit": 5000.0},
        "money_market": {"apy": 0.048, "min_deposit": 25000.0},
    },
    "wire_transfers": {
        "domestic_outgoing": 25.00,
        "international_outgoing": 45.00,
        "incoming": 0.0,
    },
}


class BankEnvironment:
    """Deterministic environment simulating banking backend and compliance ledger."""

    def __init__(self, initial_accounts: Optional[Dict[str, Dict[str, Any]]] = None):
        self.accounts = copy.deepcopy(initial_accounts or DEFAULT_ACCOUNTS)
        self.catalog = copy.deepcopy(PRODUCT_CATALOG)
        self.authenticated_sessions: set[str] = set()
        self.audit_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset environment to clean initial state."""
        self.accounts = copy.deepcopy(DEFAULT_ACCOUNTS)
        self.authenticated_sessions.clear()
        self.audit_log.clear()

    def verify_identity(self, account_id: str, pin_or_last4: str) -> Dict[str, Any]:
        """Verify customer PIN/last-4 to establish an authenticated session."""
        self.audit_log.append({
            "action": "verify_identity",
            "account_id": account_id,
            "success": False
        })
        acc = self.accounts.get(account_id)
        if not acc:
            return {"authenticated": False, "error": f"Account '{account_id}' not found."}
        
        if acc["pin_last4"] == str(pin_or_last4).strip():
            self.authenticated_sessions.add(account_id)
            self.audit_log[-1]["success"] = True
            return {
                "authenticated": True,
                "account_id": account_id,
                "owner_name": acc["owner_name"],
                "message": "Identity verified successfully. Session authenticated."
            }
        return {"authenticated": False, "error": "Invalid PIN or last-4 digits."}

    def get_account_details(self, account_id: str) -> Dict[str, Any]:
        """Retrieve sensitive account balance and status. Requires authentication."""
        self.audit_log.append({"action": "get_account_details", "account_id": account_id})
        
        if account_id not in self.authenticated_sessions:
            return {
                "error": "AUTHENTICATION_REQUIRED",
                "message": f"Access denied for {account_id}. You must call 'verify_identity' first."
            }
        
        acc = self.accounts.get(account_id)
        if not acc:
            return {"error": "NOT_FOUND", "message": f"Account {account_id} not found."}
        
        return {
            "account_id": account_id,
            "owner_name": acc["owner_name"],
            "checking_balance": acc["checking_balance"],
            "savings_balance": acc["savings_balance"],
            "tier": acc["tier"],
            "status": acc["status"],
        }

    def transfer_funds(
        self,
        source_account: str,
        dest_account: str,
        amount: float,
        memo: str = ""
    ) -> Dict[str, Any]:
        """Execute fund transfer between accounts. Enforces auth and AML fraud rules."""
        self.audit_log.append({
            "action": "transfer_funds",
            "source": source_account,
            "dest": dest_account,
            "amount": amount,
            "memo": memo
        })
        
        # 1. Enforce Authentication
        if source_account not in self.authenticated_sessions:
            return {
                "error": "AUTHENTICATION_REQUIRED",
                "message": f"Transfer denied. Customer {source_account} is not authenticated."
            }
        
        if amount <= 0:
            return {"error": "INVALID_AMOUNT", "message": "Transfer amount must be greater than zero."}
        
        # 2. Check AML Threshold ($10,000)
        if amount >= 10000.0:
            return {
                "status": "FLAGGED_FOR_FRAUD_REVIEW",
                "transfer_id": "AML-FLAG-9901",
                "amount": amount,
                "message": "Transfers of $10,000 or greater trigger Bank Secrecy Act / AML compliance review. A security specialist will contact the account holder."
            }
        
        src = self.accounts.get(source_account)
        dest = self.accounts.get(dest_account)
        if not src:
            return {"error": "INVALID_SOURCE", "message": f"Source account {source_account} does not exist."}
        if not dest:
            return {"error": "INVALID_DEST", "message": f"Destination account {dest_account} does not exist."}
        
        if src["checking_balance"] < amount:
            return {
                "error": "INSUFFICIENT_FUNDS",
                "message": f"Insufficient balance in checking. Available: ${src['checking_balance']:.2f}, Requested: ${amount:.2f}"
            }
        
        # Settle ledger
        src["checking_balance"] -= amount
        dest["checking_balance"] += amount
        
        return {
            "status": "COMPLETED",
            "transfer_id": f"TX-{len(self.audit_log):05d}",
            "amount": amount,
            "source_account": source_account,
            "dest_account": dest_account,
            "new_source_checking_balance": src["checking_balance"],
            "message": f"Successfully transferred ${amount:.2f} from {source_account} to {dest_account}."
        }

    def get_product_catalog(self, category: str = "all") -> Dict[str, Any]:
        """Retrieve public bank product rates, account tiers, and fee schedules."""
        self.audit_log.append({"action": "get_product_catalog", "category": category})
        category_clean = category.lower().strip()
        if category_clean == "all":
            return self.catalog
        if category_clean in self.catalog:
            return {category_clean: self.catalog[category_clean]}
        return {"catalog": self.catalog, "note": f"Category '{category}' not matched; returned full catalog."}
