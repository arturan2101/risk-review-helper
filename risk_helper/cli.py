import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from risk_helper import __version__
from risk_helper.completeness import (
    check_completeness,
    compute_risk_score,
    expert_consensus,
    risk_level,
)
from risk_helper.llm import LLM, FakeLLM, HuggingFaceLLM
from risk_helper.models import DecisionPackage, Risk
from risk_helper.summarizer import summarize

# Canned answer for --fake-llm: runs the full pipeline in demos and CI without a model.
FAKE_RESPONSE = json.dumps(
    {
        "summary": "Demo summary from FakeLLM; no real model was used.",
        "draft_recommendation": "mitigate",
        "rationale": "Demo rationale. Run without --fake-llm to use the real model.",
    }
)


def build_package(risk: Risk, llm: LLM | None) -> DecisionPackage:
    completeness = check_completeness(risk)
    score = compute_risk_score(risk)
    draft, status = summarize(risk, completeness, score, llm)
    return DecisionPackage(
        risk_id=risk.id,
        completeness=completeness,
        risk_score=score,
        risk_level=risk_level(score),
        expert_consensus=expert_consensus(risk.expert_opinions),
        llm_status=status,
        **draft,
    )


def load_risk(path: Path) -> Risk:
    # utf-8-sig also accepts files saved with a BOM (e.g. by Windows Notepad).
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return Risk.model_validate(data)


def _format_validation_error(exc: ValidationError) -> str:
    lines = [f"  {'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()]
    return "invalid risk description:\n" + "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="risk_helper",
        description="Prepare a decision package for a human risk reviewer.",
    )
    parser.add_argument("path", type=Path, help="risk description (JSON file)")
    llm_mode = parser.add_mutually_exclusive_group()
    llm_mode.add_argument("--no-llm", action="store_true", help="skip the LLM step (rules only)")
    llm_mode.add_argument(
        "--fake-llm", action="store_true", help="use a canned LLM answer (demo / CI)"
    )
    parser.add_argument("--output", type=Path, help="write JSON to this file instead of stdout")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Validate input before loading the model: loading takes seconds, a typo shouldn't.
    try:
        risk = load_risk(args.path)
    except (OSError, UnicodeDecodeError) as exc:
        return _error(f"cannot read {args.path}: {exc}")
    except json.JSONDecodeError as exc:
        return _error(f"{args.path} is not valid JSON: {exc}")
    except ValidationError as exc:
        return _error(_format_validation_error(exc))

    llm: LLM | None
    if args.no_llm:
        llm = None
    elif args.fake_llm:
        llm = FakeLLM(FAKE_RESPONSE)
    else:
        try:
            llm = HuggingFaceLLM()
        # OSError: e.g. no internet on the first run, before the model is cached.
        except (ImportError, OSError) as exc:
            return _error(f"{exc}\n(or run with --no-llm / --fake-llm)")

    output = build_package(risk, llm).model_dump_json(indent=2)

    if args.output is None:
        print(output)
        return 0
    try:
        args.output.write_text(output + "\n", encoding="utf-8")
    except OSError as exc:
        return _error(f"cannot write {args.output}: {exc}")
    print(f"Decision package written to {args.output}", file=sys.stderr)
    return 0
