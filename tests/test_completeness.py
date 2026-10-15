import pytest

from risk_helper.completeness import (
    check_completeness,
    compute_risk_score,
    expert_consensus,
    risk_level,
)
from risk_helper.models import ExpertOpinion, Risk


def test_full_risk_is_complete(risk_data):
    result = check_completeness(Risk.model_validate(risk_data))
    assert result.score == 1.0
    assert result.missing_fields == []
    assert result.warnings == []


def test_missing_fields_listed_and_scored(risk_data):
    del risk_data["owner"]
    del risk_data["mitigation"]
    risk_data["expert_opinions"] = []
    risk_data["impact"] = 2  # keep the high-impact warning out of this test

    result = check_completeness(Risk.model_validate(risk_data))
    assert result.missing_fields == ["owner", "mitigation", "expert_opinions"]
    assert result.score == 0.5


def test_score_is_rounded_to_two_decimals(risk_data):
    del risk_data["category"]
    result = check_completeness(Risk.model_validate(risk_data))
    assert result.score == 0.83  # 5 of 6


def test_minimal_risk_has_all_fields_missing():
    risk = Risk.model_validate({"id": "R-2", "title": "T", "description": "D" * 40})
    result = check_completeness(risk)
    assert result.score == 0.0
    assert len(result.missing_fields) == 6


def test_blank_owner_counts_as_missing(risk_data):
    risk_data["owner"] = "   "
    result = check_completeness(Risk.model_validate(risk_data))
    assert result.missing_fields == ["owner"]


def test_short_description_warns_without_affecting_score(risk_data):
    risk_data["description"] = "Too short."
    result = check_completeness(Risk.model_validate(risk_data))
    assert result.score == 1.0
    assert len(result.warnings) == 1
    assert "shorter than 30" in result.warnings[0]


def test_high_impact_without_mitigation_warns(risk_data):
    del risk_data["mitigation"]
    result = check_completeness(Risk.model_validate(risk_data))
    assert len(result.warnings) == 1
    assert "no mitigation" in result.warnings[0]


@pytest.mark.parametrize(
    ("impact", "mitigation"),
    [(3, None), (4, "Has a plan."), (None, None)],
)
def test_no_mitigation_warning_when_not_needed(risk_data, impact, mitigation):
    risk_data["impact"] = impact
    risk_data["mitigation"] = mitigation
    result = check_completeness(Risk.model_validate(risk_data))
    assert not any("mitigation" in w for w in result.warnings)


def test_both_warnings_can_trigger_together(risk_data):
    risk_data["description"] = "Short."
    risk_data["impact"] = 5
    del risk_data["mitigation"]
    result = check_completeness(Risk.model_validate(risk_data))
    assert len(result.warnings) == 2


def test_risk_score_is_product(risk_data):
    assert compute_risk_score(Risk.model_validate(risk_data)) == 12


@pytest.mark.parametrize("field", ["likelihood", "impact"])
def test_risk_score_none_when_input_missing(risk_data, field):
    risk_data[field] = None
    assert compute_risk_score(Risk.model_validate(risk_data)) is None


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1, "low"),
        (6, "low"),
        (7, "medium"),
        (14, "medium"),
        (15, "high"),
        (25, "high"),
        (None, "unknown"),
    ],
)
def test_risk_level_boundaries(score, expected):
    assert risk_level(score) == expected


def _opinions(*ratings: str) -> list[ExpertOpinion]:
    return [ExpertOpinion(expert=f"E{i}", rating=r) for i, r in enumerate(ratings)]


@pytest.mark.parametrize(
    ("ratings", "expected"),
    [
        ((), "none"),
        (("high",), "agreed"),
        (("medium", "medium", "medium"), "agreed"),
        (("high", "medium"), "split"),
        (("low", "low", "high"), "split"),
    ],
)
def test_expert_consensus(ratings, expected):
    assert expert_consensus(_opinions(*ratings)) == expected
