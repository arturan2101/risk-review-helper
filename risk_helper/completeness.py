from risk_helper.models import CompletenessResult, Consensus, ExpertOpinion, Risk, RiskLevel

CHECKED_FIELDS = ("category", "likelihood", "impact", "owner", "mitigation", "expert_opinions")
MIN_DESCRIPTION_LENGTH = 30
HIGH_IMPACT = 4


def _is_missing(value: object) -> bool:
    # None, empty strings and empty lists all mean "not provided".
    return value is None or value == "" or value == []


def check_completeness(risk: Risk) -> CompletenessResult:
    missing = [name for name in CHECKED_FIELDS if _is_missing(getattr(risk, name))]
    score = round((len(CHECKED_FIELDS) - len(missing)) / len(CHECKED_FIELDS), 2)

    # Warnings flag quality issues for the reviewer but do not affect the score.
    warnings = []
    if len(risk.description) < MIN_DESCRIPTION_LENGTH:
        warnings.append(f"Description is shorter than {MIN_DESCRIPTION_LENGTH} characters.")
    if risk.impact is not None and risk.impact >= HIGH_IMPACT and _is_missing(risk.mitigation):
        warnings.append(f"Impact is {risk.impact} but no mitigation is defined.")

    return CompletenessResult(score=score, missing_fields=missing, warnings=warnings)


def compute_risk_score(risk: Risk) -> int | None:
    if risk.likelihood is None or risk.impact is None:
        return None
    return risk.likelihood * risk.impact


def risk_level(score: int | None) -> RiskLevel:
    if score is None:
        return "unknown"
    if score <= 6:
        return "low"
    if score <= 14:
        return "medium"
    return "high"


def expert_consensus(opinions: list[ExpertOpinion]) -> Consensus:
    if not opinions:
        return "none"
    return "agreed" if len({o.rating for o in opinions}) == 1 else "split"
