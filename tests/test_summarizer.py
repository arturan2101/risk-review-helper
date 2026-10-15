import json
import sys
from typing import get_args

import pytest

from risk_helper.completeness import check_completeness, compute_risk_score
from risk_helper.llm import FakeLLM, HuggingFaceLLM
from risk_helper.models import Recommendation, Risk
from risk_helper.summarizer import (
    RECOMMENDATION_HELP,
    STRICT_SUFFIX,
    build_prompt,
    parse_llm_output,
    summarize,
)

VALID = {
    "summary": "Vendor outages delayed invoices.",
    "draft_recommendation": "mitigate",
    "rationale": "Impact is high and a secondary provider is feasible.",
}
VALID_JSON = json.dumps(VALID)
GARBAGE = "Sure! Here is my assessment: the risk seems quite serious."


def _inputs(risk_data: dict):
    risk = Risk.model_validate(risk_data)
    return risk, check_completeness(risk), compute_risk_score(risk)


# --- prompt -----------------------------------------------------------------


def test_prompt_contains_risk_facts_and_missing_fields(risk_data):
    del risk_data["owner"]
    prompt = build_prompt(*_inputs(risk_data))
    assert risk_data["title"] in prompt
    assert "MISSING FIELDS: owner" in prompt
    for opinion in risk_data["expert_opinions"]:
        assert opinion["comment"] in prompt


def test_prompt_lists_all_allowed_recommendations(risk_data):
    prompt = build_prompt(*_inputs(risk_data))
    for value in get_args(Recommendation):
        assert f"- {value}:" in prompt


def test_recommendation_help_matches_model():
    assert set(RECOMMENDATION_HELP) == set(get_args(Recommendation))


def test_prompt_marks_unknown_values(risk_data):
    risk_data["likelihood"] = None
    risk_data["expert_opinions"] = []
    prompt = build_prompt(*_inputs(risk_data))
    assert "Likelihood (1-5): not provided" in prompt
    assert "Risk score (likelihood x impact): unknown" in prompt
    assert "- none provided" in prompt


# --- parsing ----------------------------------------------------------------


def test_clean_json_parses():
    assert parse_llm_output(VALID_JSON) == VALID


@pytest.mark.parametrize(
    "text",
    [
        f"Here is the assessment:\n{VALID_JSON}\nLet me know if you need more.",
        f"```json\n{VALID_JSON}\n```",
        f"Answer:\n```\n{json.dumps(VALID, indent=2)}\n```",
        json.dumps([VALID]),  # first {...} block is taken even inside an array
    ],
)
def test_json_wrapped_in_prose_or_fence_parses(text):
    assert parse_llm_output(text) == VALID


def test_braces_inside_strings_do_not_break_parsing():
    data = VALID | {"summary": "Risk {R-001} affects billing."}
    assert parse_llm_output(f"Result: {json.dumps(data)}") == data


def test_recommendation_is_normalized():
    data = VALID | {"draft_recommendation": " Mitigate "}
    assert parse_llm_output(json.dumps(data))["draft_recommendation"] == "mitigate"


def test_extra_keys_are_dropped():
    data = VALID | {"confidence": 0.9}
    assert parse_llm_output(json.dumps(data)) == VALID


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(VALID | {"draft_recommendation": "ignore"}),
        json.dumps({k: v for k, v in VALID.items() if k != "rationale"}),
        json.dumps(VALID | {"summary": "   "}),
        json.dumps(VALID | {"summary": 42}),
        "{not valid json}",
        GARBAGE,
        "",
    ],
    ids=[
        "bad-recommendation",
        "missing-key",
        "blank-summary",
        "non-string-summary",
        "broken-json",
        "prose",
        "empty",
    ],
)
def test_invalid_output_rejected(text):
    assert parse_llm_output(text) is None


# --- summarize --------------------------------------------------------------


def test_valid_output_first_try(risk_data):
    llm = FakeLLM(VALID_JSON)
    draft, status = summarize(*_inputs(risk_data), llm)
    assert status == "ok"
    assert draft == VALID
    assert len(llm.prompts) == 1


def test_garbage_twice_falls_back(risk_data):
    llm = FakeLLM(GARBAGE)
    draft, status = summarize(*_inputs(risk_data), llm)
    assert status == "fallback"
    assert draft["draft_recommendation"] == "undecided"
    assert risk_data["title"] in draft["summary"]
    assert len(llm.prompts) == 2


def test_retry_uses_stricter_prompt(risk_data):
    llm = FakeLLM(GARBAGE)
    summarize(*_inputs(risk_data), llm)
    assert STRICT_SUFFIX not in llm.prompts[0]
    assert llm.prompts[1].endswith(STRICT_SUFFIX)


def test_garbage_then_valid_is_ok(risk_data):
    llm = FakeLLM(GARBAGE, VALID_JSON)
    draft, status = summarize(*_inputs(risk_data), llm)
    assert status == "ok"
    assert draft == VALID


def test_no_llm_is_skipped(risk_data):
    del risk_data["owner"]
    draft, status = summarize(*_inputs(risk_data), None)
    assert status == "skipped"
    assert draft["draft_recommendation"] == "undecided"
    assert "Missing: owner" in draft["summary"]


# --- llm module -------------------------------------------------------------


def test_fake_llm_repeats_last_response():
    llm = FakeLLM("a", "b")
    assert [llm.generate("p") for _ in range(3)] == ["a", "b", "b"]


def test_fake_llm_needs_a_response():
    with pytest.raises(ValueError):
        FakeLLM()


def test_core_does_not_import_transformers():
    assert "transformers" not in sys.modules


def test_huggingface_llm_explains_missing_dependency(monkeypatch):
    # A None entry in sys.modules makes the import fail, even if transformers is installed.
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="requirements-llm.txt"):
        HuggingFaceLLM()
