from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["it", "vendor", "security", "compliance", "operational"]
Rating = Literal["low", "medium", "high"]
RiskLevel = Literal["low", "medium", "high", "unknown"]
Consensus = Literal["none", "agreed", "split"]
Recommendation = Literal["accept", "mitigate", "transfer", "avoid", "undecided"]
LLMStatus = Literal["ok", "fallback", "skipped"]

# Strip whitespace so "   " fails min_length; forbid extra keys so a typo like
# "likelyhood" is an error instead of silently counting as a missing field.
_INPUT_CONFIG = ConfigDict(str_strip_whitespace=True, extra="forbid")


class ExpertOpinion(BaseModel):
    model_config = _INPUT_CONFIG

    expert: str = Field(min_length=1)
    rating: Rating
    comment: str = ""


class Risk(BaseModel):
    model_config = _INPUT_CONFIG

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    # Optional so the completeness check can report it as missing instead of rejecting input.
    category: Category | None = None
    likelihood: int | None = Field(default=None, ge=1, le=5)
    impact: int | None = Field(default=None, ge=1, le=5)
    owner: str | None = None
    mitigation: str | None = None
    expert_opinions: list[ExpertOpinion] = Field(default_factory=list)


class CompletenessResult(BaseModel):
    score: float = Field(ge=0, le=1)
    missing_fields: list[str]
    warnings: list[str]


class DecisionPackage(BaseModel):
    risk_id: str
    completeness: CompletenessResult
    risk_score: int | None
    risk_level: RiskLevel
    expert_consensus: Consensus
    summary: str
    draft_recommendation: Recommendation
    rationale: str
    llm_status: LLMStatus
    # Literal[True]: the package can never claim that no human decision is needed.
    requires_human_decision: Literal[True] = True
