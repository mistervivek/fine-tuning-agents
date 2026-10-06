"""Unit tests for BankEnvironment and security invariants."""

import pytest
from src.environment.bank_env import BankEnvironment


def test_bank_environment_auth_flow():
    env = BankEnvironment()

    # 1. Unauthenticated lookup should be denied
    res_unauth = env.get_account_details("ACC-1001")
    assert res_unauth.get("error") == "AUTHENTICATION_REQUIRED"

    # 2. Invalid PIN should fail
    auth_fail = env.verify_identity("ACC-1001", "0000")
    assert auth_fail["authenticated"] is False

    # 3. Valid PIN should succeed
    auth_success = env.verify_identity("ACC-1001", "4821")
    assert auth_success["authenticated"] is True

    # 4. Authenticated lookup should now succeed
    res_auth = env.get_account_details("ACC-1001")
    assert "error" not in res_auth
    assert res_auth["checking_balance"] == 4250.75


def test_fund_transfer_limits_and_aml():
    env = BankEnvironment()
    env.verify_identity("ACC-1001", "4821")

    # Transfer $250
    tx_res = env.transfer_funds(
        source_account="ACC-1001",
        dest_account="ACC-2002",
        amount=250.0
    )
    assert tx_res["status"] == "COMPLETED"
    assert tx_res["new_source_checking_balance"] == 4000.75

    # Transfer >= $10,000 should trigger AML review
    aml_res = env.transfer_funds(
        source_account="ACC-1001",
        dest_account="ACC-2002",
        amount=15000.0
    )
    assert aml_res["status"] == "FLAGGED_FOR_FRAUD_REVIEW"


def test_product_catalog():
    env = BankEnvironment()
    cat = env.get_product_catalog("savings")
    assert "savings" in cat
    assert "regular" in cat["savings"]
