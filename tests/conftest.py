import pytest


@pytest.fixture
def risk_data() -> dict:
    # Fresh dict per test, so tests can modify it freely.
    return {
        "id": "R-001",
        "title": "Third-party billing provider outage",
        "description": "Billing provider had two outages last quarter; invoices delayed up to 48h.",
        "category": "vendor",
        "likelihood": 3,
        "impact": 4,
        "owner": "Billing Team",
        "mitigation": "Evaluate secondary provider; add SLA penalties.",
        "expert_opinions": [
            {"expert": "A. Meier", "rating": "high", "comment": "Customer-facing impact."},
            {"expert": "J. Berzins", "rating": "medium", "comment": "Outages were short."},
        ],
    }
