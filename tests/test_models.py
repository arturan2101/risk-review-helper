import pytest
from pydantic import ValidationError

from risk_helper.models import CompletenessResult, DecisionPackage, Risk


def test_valid_risk_parses(risk_data):
    risk = Risk.model_validate(risk_data)
    assert risk.id == "R-001"
    assert risk.likelihood == 3
    assert risk.expert_opinions[0].rating == "high"


def test_minimal_risk_uses_defaults():
    risk = Risk.model_validate({"id": "R-2", "title": "T", "description": "D"})
    assert risk.category is None
    assert risk.likelihood is None
    assert risk.owner is None
    assert risk.expert_opinions == []


def test_null_likelihood_and_impact_allowed(risk_data):
    risk_data["likelihood"] = None
    risk_data["impact"] = None
    risk = Risk.model_validate(risk_data)
    assert risk.likelihood is None and risk.impact is None


@pytest.mark.parametrize("field", ["likelihood", "impact"])
@pytest.mark.parametrize("value", [0, 6])
def test_out_of_range_scale_rejected(risk_data, field, value):
    risk_data[field] = value
    with pytest.raises(ValidationError):
        Risk.model_validate(risk_data)


def test_bad_category_rejected(risk_data):
    risk_data["category"] = "finance"
    with pytest.raises(ValidationError):
        Risk.model_validate(risk_data)


@pytest.mark.parametrize("field", ["id", "title", "description"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_required_string_rejected(risk_data, field, value):
    risk_data[field] = value
    with pytest.raises(ValidationError):
        Risk.model_validate(risk_data)


@pytest.mark.parametrize("field", ["id", "title", "description"])
def test_missing_required_field_rejected(risk_data, field):
    del risk_data[field]
    with pytest.raises(ValidationError):
        Risk.model_validate(risk_data)


def test_bad_expert_rating_rejected(risk_data):
    risk_data["expert_opinions"][0]["rating"] = "critical"
    with pytest.raises(ValidationError):
        Risk.model_validate(risk_data)


def test_unknown_field_rejected(risk_data):
    risk_data["likelyhood"] = 3
    with pytest.raises(ValidationError):
        Risk.model_validate(risk_data)


def _package(**overrides) -> DecisionPackage:
    fields = {
        "risk_id": "R-001",
        "completeness": CompletenessResult(score=1.0, missing_fields=[], warnings=[]),
        "risk_score": 12,
        "risk_level": "medium",
        "expert_consensus": "split",
        "summary": "s",
        "draft_recommendation": "mitigate",
        "rationale": "r",
        "llm_status": "ok",
    }
    return DecisionPackage(**(fields | overrides))


def test_decision_package_always_requires_human():
    assert _package().requires_human_decision is True
    with pytest.raises(ValidationError):
        _package(requires_human_decision=False)


def test_decision_package_rejects_unknown_recommendation():
    with pytest.raises(ValidationError):
        _package(draft_recommendation="ignore")
