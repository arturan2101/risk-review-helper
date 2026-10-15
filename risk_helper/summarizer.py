import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from risk_helper.completeness import risk_level
from risk_helper.llm import LLM
from risk_helper.models import CompletenessResult, LLMStatus, Recommendation, Risk

# Short definitions help a small model pick a sensible value. Keys must match Recommendation.
RECOMMENDATION_HELP = {
    "accept": "the risk is tolerable, no action needed",
    "mitigate": "reduce the likelihood or the impact",
    "transfer": "shift the risk to a third party (insurance, contract, vendor)",
    "avoid": "stop the activity that causes the risk",
    "undecided": "not enough information to recommend anything",
}

STRICT_SUFFIX = (
    "\n\nSTRICT FORMAT: reply with exactly one JSON object and nothing else. "
    "No markdown, no code fences, no text before or after it. "
    "Keys: summary, draft_recommendation, rationale."
)


class _LLMDraft(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(min_length=1)
    draft_recommendation: Recommendation
    rationale: str = Field(min_length=1)

    @field_validator("draft_recommendation", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        # Models often answer "Mitigate" or " mitigate"; accept those, reject anything else.
        return value.strip().lower() if isinstance(value, str) else value


def _or_missing(value: object) -> str:
    return "not provided" if value is None or value == "" else str(value)


def _score_text(risk_score: int | None) -> str:
    if risk_score is None:
        return "unknown (likelihood or impact missing)"
    return f"{risk_score} of 25 ({risk_level(risk_score)})"


def build_prompt(risk: Risk, completeness: CompletenessResult, risk_score: int | None) -> str:
    opinions = [
        f"- {o.expert} (rating: {o.rating}): {o.comment or 'no comment'}"
        for o in risk.expert_opinions
    ] or ["- none provided"]
    recommendations = [f"- {name}: {meaning}" for name, meaning in RECOMMENDATION_HELP.items()]

    lines = [
        "You are drafting a risk assessment for a human reviewer, who makes the final decision.",
        "Use only the facts below. Do not invent numbers, dates, names or events.",
        "If important information is missing, say so instead of guessing.",
        "",
        "RISK",
        f"ID: {risk.id}",
        f"Title: {risk.title}",
        f"Description: {risk.description}",
        f"Category: {_or_missing(risk.category)}",
        f"Likelihood (1-5): {_or_missing(risk.likelihood)}",
        f"Impact (1-5): {_or_missing(risk.impact)}",
        f"Risk score (likelihood x impact): {_score_text(risk_score)}",
        f"Owner: {_or_missing(risk.owner)}",
        f"Mitigation: {_or_missing(risk.mitigation)}",
        "",
        "EXPERT OPINIONS",
        *opinions,
        "",
        f"MISSING FIELDS: {', '.join(completeness.missing_fields) or 'none'}",
        f"WARNINGS: {' '.join(completeness.warnings) or 'none'}",
        "",
        "TASK",
        "Reply with JSON only, using exactly these keys:",
        '{"summary": "2-3 sentences", "draft_recommendation": "one allowed value", '
        '"rationale": "1-2 sentences"}',
        "Allowed values for draft_recommendation:",
        *recommendations,
    ]
    return "\n".join(lines)


def _first_json_object(text: str) -> dict | None:
    # raw_decode parses one JSON value and ignores whatever follows, so prose or a closing
    # ``` fence after the object is fine, and braces inside strings don't confuse it
    # (unlike a regex). Try each "{" until one parses; starting at "{" the result is a dict.
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            continue
    return None


def parse_llm_output(text: str) -> dict | None:
    data = _first_json_object(text)
    if data is None:
        return None
    try:
        return _LLMDraft.model_validate(data).model_dump()
    except ValidationError:
        return None


def _rule_based_draft(
    risk: Risk, completeness: CompletenessResult, risk_score: int | None, rationale: str
) -> dict:
    score = "unknown" if risk_score is None else f"{risk_score} ({risk_level(risk_score)})"
    summary = (
        f"{risk.title} [{risk.category or 'no category'}]: risk score {score}, "
        f"{len(risk.expert_opinions)} expert opinion(s)."
    )
    if completeness.missing_fields:
        summary += f" Missing: {', '.join(completeness.missing_fields)}."
    return {"summary": summary, "draft_recommendation": "undecided", "rationale": rationale}


def summarize(
    risk: Risk,
    completeness: CompletenessResult,
    risk_score: int | None,
    llm: LLM | None,
) -> tuple[dict, LLMStatus]:
    # llm=None means the LLM step was skipped on purpose (--no-llm).
    if llm is None:
        rationale = "LLM step was skipped; no draft recommendation."
        return _rule_based_draft(risk, completeness, risk_score, rationale), "skipped"

    prompt = build_prompt(risk, completeness, risk_score)
    for attempt in (prompt, prompt + STRICT_SUFFIX):
        draft = parse_llm_output(llm.generate(attempt))
        if draft is not None:
            return draft, "ok"

    rationale = "LLM output could not be parsed after a retry; recommendation left to the reviewer."
    return _rule_based_draft(risk, completeness, risk_score, rationale), "fallback"
