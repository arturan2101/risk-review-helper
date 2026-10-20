import json
import subprocess
import sys
from pathlib import Path

import pytest

from risk_helper.cli import build_package, load_risk, main
from risk_helper.models import DecisionPackage

ROOT = Path(__file__).resolve().parent.parent
COMPLETE = ROOT / "examples" / "complete_risk.json"
INCOMPLETE = ROOT / "examples" / "incomplete_risk.json"


def _run(capsys, *args: str) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


@pytest.mark.parametrize("example", [COMPLETE, INCOMPLETE], ids=["complete", "incomplete"])
def test_no_llm_produces_valid_package(capsys, example):
    code, out, _ = _run(capsys, example, "--no-llm")
    assert code == 0
    package = DecisionPackage.model_validate_json(out)
    assert package.requires_human_decision is True
    assert package.llm_status == "skipped"
    assert package.draft_recommendation == "undecided"


def test_fake_llm_status_ok(capsys):
    code, out, _ = _run(capsys, COMPLETE, "--fake-llm")
    assert code == 0
    package = DecisionPackage.model_validate_json(out)
    assert package.llm_status == "ok"
    assert package.draft_recommendation == "mitigate"


def test_incomplete_example_package_contents():
    package = build_package(load_risk(INCOMPLETE), llm=None)
    assert package.completeness.missing_fields == ["owner", "mitigation", "expert_opinions"]
    assert package.completeness.score == 0.5
    assert package.risk_score == 20
    assert package.risk_level == "high"
    assert package.expert_consensus == "none"
    assert len(package.completeness.warnings) == 1


def test_output_flag_writes_file(capsys, tmp_path):
    target = tmp_path / "package.json"
    code, out, err = _run(capsys, COMPLETE, "--no-llm", "--output", target)
    assert code == 0
    assert out == ""
    assert str(target) in err
    assert DecisionPackage.model_validate_json(target.read_text(encoding="utf-8"))


def test_missing_file_exits_1(capsys, tmp_path):
    code, out, err = _run(capsys, tmp_path / "nope.json", "--no-llm")
    assert code == 1
    assert out == ""
    assert "cannot read" in err


def test_invalid_json_exits_1(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code, _, err = _run(capsys, bad, "--no-llm")
    assert code == 1
    assert "not valid JSON" in err


def test_validation_error_exits_1_and_names_field(capsys, tmp_path):
    data = json.loads(COMPLETE.read_text(encoding="utf-8"))
    data["likelihood"] = 9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    code, _, err = _run(capsys, bad, "--no-llm")
    assert code == 1
    assert "likelihood" in err


def test_unwritable_output_exits_1(capsys, tmp_path):
    target = tmp_path / "missing-dir" / "package.json"
    code, _, err = _run(capsys, COMPLETE, "--no-llm", "--output", target)
    assert code == 1
    assert "cannot write" in err


def test_file_with_bom_is_accepted(capsys, tmp_path):
    bom = tmp_path / "bom.json"
    bom.write_text(COMPLETE.read_text(encoding="utf-8"), encoding="utf-8-sig")
    code, _, _ = _run(capsys, bom, "--no-llm")
    assert code == 0


def test_llm_flags_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(COMPLETE), "--no-llm", "--fake-llm"])
    assert exc.value.code == 2


def test_real_llm_without_dependencies_exits_1(capsys, monkeypatch):
    # Blocks the transformers import, so this never downloads a model even if it's installed.
    monkeypatch.setitem(sys.modules, "transformers", None)
    code, _, err = _run(capsys, COMPLETE)
    assert code == 1
    assert "requirements-llm.txt" in err


def test_model_download_failure_exits_1(capsys, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr("risk_helper.cli.HuggingFaceLLM", offline)
    code, _, err = _run(capsys, COMPLETE)
    assert code == 1
    assert "couldn't connect" in err
    assert "--no-llm" in err


def test_python_dash_m_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "risk_helper", str(COMPLETE), "--fake-llm"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert DecisionPackage.model_validate_json(result.stdout).llm_status == "ok"
