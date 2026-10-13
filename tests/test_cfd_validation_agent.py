from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src.cfd_validation_agent.case_discovery import inventory_case, select_cases
from src.cfd_validation_agent.cli import parse_args
from src.cfd_validation_agent.config import AgentConfig, ForceThresholds, ResidualThresholds
from src.cfd_validation_agent.environment import load_project_environment
from src.cfd_validation_agent.force_coeffs_parser import analyze_force_coeffs, parse_force_coeffs
from src.cfd_validation_agent.llm_agent import BudgetExhausted, RunBudget, run_agent
from src.cfd_validation_agent.log_parser import parse_log_file
from src.cfd_validation_agent.pipeline import ValidationPipeline
from src.cfd_validation_agent.residuals_parser import analyze_residuals, parse_residual_table, parse_solver_log
from src.cfd_validation_agent.schemas import AgentFinding, CaseResult, FinalStatus, Severity


def _case(tmp_path: Path, name: str = "case_0001_test") -> Path:
    path = tmp_path / name
    (path / "logs").mkdir(parents=True)
    return path


def _force_file(path: Path, values: list[tuple[float, float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Force coefficients\n# Time Cm Cd Cl\n"
    text += "".join(f"{i} {cm:.9g} {cd:.9g} {cl:.9g}\n" for i, (cm, cd, cl) in enumerate(values))
    path.write_text(text, encoding="utf-8")
    return path


def _residual_file(path: Path, values: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Residuals\n# Time p Ux Uy k omega\n0 N/A N/A N/A N/A\n"
    text += "".join(f"{i + 1} {v} {v} {v} {v} {v}\n" for i, v in enumerate(values))
    path.write_text(text, encoding="utf-8")
    return path


def test_log_parser_handles_ansi_warning_fatal_and_failed_mesh(tmp_path: Path) -> None:
    case = _case(tmp_path)
    log = case / "logs" / "02_checkMesh.log"
    log.write_text("\x1b[31m--> FOAM Warning : harmless\x1b[0m\nFailed 1 mesh checks.\nFOAM FATAL ERROR: bad mesh\nEnd\n")
    findings = parse_log_file(log, case)
    assert {item.category for item in findings} >= {"warning_review", "mesh_failure_ambiguous", "fatal_error"}
    assert next(item for item in findings if item.category == "fatal_error").line_number == 3


def _check_mesh_result(tmp_path: Path, body: str) -> CaseResult:
    case = _case(tmp_path)
    (case / "logs" / "02_checkMesh.log").write_text(body + "\nEnd\n", encoding="utf-8")
    return ValidationPipeline().deterministic(case)


def test_expected_2d_aspect_ratio_is_informational(tmp_path: Path) -> None:
    result = _check_mesh_result(tmp_path, " *** High   aspect ratio cells found , Max aspect ratio: 2000\nFailed 1 mesh checks.")
    expected = next(item for item in result.evidence if item.category == "expected_2d_mesh_check")
    assert expected.severity == Severity.INFO and expected.message.startswith("*** High")
    assert result.deterministic_validation["mesh"] == FinalStatus.PASS
    assert not any(issue.type == "mesh_failed" for issue in result.issues)


@pytest.mark.parametrize("body", [
    "***Illegal face found\nFailed 1 mesh checks.",
    "***Illegal face found\n***Bad topology found\nFailed 2 mesh checks.",
])
def test_unexpected_failed_mesh_checks_are_suspicious(tmp_path: Path, body: str) -> None:
    result = _check_mesh_result(tmp_path, body)
    assert result.deterministic_validation["mesh"] == FinalStatus.REVIEW
    assert any(issue.type == "mesh_failed" for issue in result.issues)


def test_expected_plus_unexpected_mesh_failure_is_not_accepted(tmp_path: Path) -> None:
    result = _check_mesh_result(tmp_path, "***High aspect ratio cells found, Max aspect ratio: 2000\n***Illegal face found\nFailed 2 mesh checks.")
    assert result.deterministic_validation["mesh"] == FinalStatus.REVIEW
    assert {item.category for item in result.evidence} >= {"expected_2d_mesh_check", "mesh_failed"}


def test_zero_and_ambiguous_failed_mesh_checks(tmp_path: Path) -> None:
    zero = _check_mesh_result(tmp_path / "zero", "Mesh OK.\nFailed 0 mesh checks.")
    assert zero.deterministic_validation["mesh"] == FinalStatus.PASS
    assert not any(item.category.startswith("mesh_") for item in zero.evidence)
    ambiguous = _check_mesh_result(tmp_path / "ambiguous", "High aspect ratio cells found\nFailed 1 mesh checks.")
    assert ambiguous.deterministic_validation["mesh"] == FinalStatus.REVIEW
    assert any(issue.type == "mesh_failure_ambiguous" for issue in ambiguous.issues)


def test_fatal_error_remains_effective_with_expected_mesh_finding(tmp_path: Path) -> None:
    result = _check_mesh_result(tmp_path, "***High aspect ratio cells found, Max aspect ratio: 2000\nFailed 1 mesh checks.\nFOAM FATAL ERROR: bad mesh")
    assert result.status == FinalStatus.FAIL
    assert any(issue.type == "fatal_error" and issue.deterministic_hard_failure for issue in result.issues)


def test_missing_completion_is_heuristic(tmp_path: Path) -> None:
    case = _case(tmp_path); log = case / "logs" / "04_simpleFoam.log"
    log.write_text("Time = 10\nSolving for Ux, Initial residual = 1e-3, Final residual = 1e-5, No Iterations 2\n")
    finding = next(item for item in parse_log_file(log, case) if item.category == "missing_completion")
    assert finding.explicit is False


def test_residual_table_handles_na_and_malformed_rows(tmp_path: Path) -> None:
    path = _residual_file(tmp_path / "residuals.dat", [1.0, 0.1])
    times, fields = parse_residual_table(path)
    assert len(times) == 3 and np.isnan(fields["p"][0]) and np.isnan(fields["omega"][0]) and fields["Ux"][-1] == 0.1
    path.write_text("# Time p Ux\n1 0.1\n")
    with pytest.raises(ValueError, match="expected 3 columns"):
        parse_residual_table(path)


def test_solver_log_keeps_repeated_pressure_solves(tmp_path: Path) -> None:
    path = tmp_path / "simpleFoam.log"
    path.write_text("Time = 7\nGAMG: Solving for p, Initial residual = 0.1, Final residual = 0.01, No Iterations 2\n"
                    "GAMG: Solving for p, Initial residual = 0.02, Final residual = 0.001, No Iterations 2\n")
    times, initial, final = parse_solver_log(path)
    assert times == [7.0, 7.0] and initial["p"] == [0.1, 0.02] and final["p"] == [0.01, 0.001]


def test_residual_convergence_oscillation_and_divergence(tmp_path: Path) -> None:
    case = _case(tmp_path)
    converged = _residual_file(case / "postProcessing/residuals/0/residuals.dat", np.geomspace(1, 1e-5, 100).tolist())
    assert analyze_residuals(case, [converged], [], ResidualThresholds()).status == FinalStatus.PASS
    oscillating = _residual_file(case / "oscillating.dat", ([1e-3, 1e-1] * 50))
    assert analyze_residuals(case, [oscillating], [], ResidualThresholds()).status == FinalStatus.REVIEW
    diverging = _residual_file(case / "diverging.dat", np.geomspace(1e-5, 1, 100).tolist())
    assert analyze_residuals(case, [diverging], [], ResidualThresholds()).status == FinalStatus.FAIL


def test_low_and_high_stagnant_residual_plateaus(tmp_path: Path) -> None:
    case = _case(tmp_path)
    low = _residual_file(case / "low.dat", [1e-6] * 100)
    result = analyze_residuals(case, [low], [], ResidualThresholds())
    assert result.status == FinalStatus.PASS
    assert any("acceptable low residual plateau" in note and "threshold=" in note for note in result.notes)

    high = _residual_file(case / "high.dat", [1e-2] * 100)
    result = analyze_residuals(case, [high], [], ResidualThresholds())
    assert result.status == FinalStatus.REVIEW
    assert any("high residual plateau" in note and "log-slope=" in note for note in result.notes)


def test_significant_residual_spike_is_detected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    values = [1e-5] * 100
    values[75] = 1e-2
    result = analyze_residuals(case, [_residual_file(case / "spike.dat", values)], [], ResidualThresholds())
    assert result.status == FinalStatus.REVIEW
    assert any("significant tail spike" in note for note in result.notes)


def test_stable_forces_do_not_hide_bad_residuals(tmp_path: Path) -> None:
    case = _case(tmp_path)
    residual = _residual_file(case / "high.dat", [1e-2] * 100)
    force = _force_file(case / "forceCoeffs.dat", [(0.0, 0.01, 0.2)] * 100)
    assert analyze_residuals(case, [residual], [], ResidualThresholds()).status == FinalStatus.REVIEW
    assert analyze_force_coeffs(case, [force], ForceThresholds()).status == FinalStatus.PASS


@pytest.mark.parametrize(("line", "category"), [
    ("Mesh non-orthogonality Max: 80 average: 30", "mesh_non_orthogonality_metric"),
    ("Mesh non-orthogonality Max: 86 average: 30", "mesh_non_orthogonality_exceeded"),
    ("Max skewness = 3.5 OK.", "mesh_skewness_metric"),
    ("Max skewness = 4.1", "mesh_skewness_exceeded"),
])
def test_mesh_metrics_use_values_and_thresholds(tmp_path: Path, line: str, category: str) -> None:
    case = _case(tmp_path); log = case / "logs" / "02_checkMesh.log"
    log.write_text(f"{line}\nMesh OK.\nEnd\n", encoding="utf-8")
    findings = parse_log_file(log, case)
    item = next(item for item in findings if item.category == category)
    assert "limit" in item.message


def test_expected_preprocessing_warning_is_informational(tmp_path: Path) -> None:
    case = _case(tmp_path); log = case / "logs" / "01_blockMesh.log"
    log.write_text("Warning: The Windows Subsystem for Linux has been disabled and compilation may not work!\nEnd\n")
    item = next(item for item in parse_log_file(log, case) if item.category == "expected_warning")
    assert item.severity == Severity.INFO


def test_missing_mesh_diagnostics_is_explicit(tmp_path: Path) -> None:
    case = _case(tmp_path); (case / "logs" / "04_simpleFoam.log").write_text("End\n")
    result = ValidationPipeline().deterministic(case)
    assert result.deterministic_validation["mesh"] == FinalStatus.REVIEW
    assert any(item.category == "missing_mesh_diagnostics" for item in result.evidence)


def test_force_coefficients_stable_oscillating_and_near_zero(tmp_path: Path) -> None:
    case = _case(tmp_path); cfg = ForceThresholds()
    stable = _force_file(case / "postProcessing/force/0/forceCoeffs.dat",
                         [(0.01 + i * 1e-8, 0.02, -0.3) for i in range(120)])
    assert analyze_force_coeffs(case, [stable], cfg).status == FinalStatus.PASS
    oscillating = _force_file(case / "osc.dat", [(0.0, 0.01 + (-1) ** i * 0.05, 0.2) for i in range(120)])
    assert analyze_force_coeffs(case, [oscillating], cfg).status == FinalStatus.REVIEW
    near_zero = _force_file(case / "zero.dat", [(1e-8, (-1) ** i * 1e-7, 1e-8) for i in range(120)])
    assert analyze_force_coeffs(case, [near_zero], cfg).status == FinalStatus.PASS


def test_force_restarts_and_missing_evidence(tmp_path: Path) -> None:
    first = _force_file(tmp_path / "0/forceCoeffs.dat", [(0, 1, 2), (0, 1, 2)])
    second = _force_file(tmp_path / "100/forceCoeffs.dat", [(0, 3, 4)])
    _, fields = parse_force_coeffs(first)
    assert fields["Cd"] == [1.0, 1.0]
    result = analyze_force_coeffs(tmp_path, [], ForceThresholds())
    assert result.status == FinalStatus.REVIEW and "No forceCoeffs" in result.notes[0]


def test_incomplete_case_is_review_and_hard_failure_cannot_be_overridden(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = ValidationPipeline().deterministic(case)
    assert result.status == FinalStatus.REVIEW
    log = case / "logs" / "04_simpleFoam.log"; log.write_text("FOAM FATAL ERROR: solver exploded\nEnd\n")
    result = ValidationPipeline().deterministic(case)
    assert result.status == FinalStatus.FAIL
    assert any(issue.deterministic_hard_failure for issue in result.issues)


def test_pydantic_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        AgentFinding.model_validate({"summary": "x", "observed_facts": [], "hypotheses": [],
                                     "recommendations": [], "additional_issues": [{"type": "x", "severity": "impossible"}]})


def test_case_selection_and_cli_safety(tmp_path: Path) -> None:
    for index in range(7): _case(tmp_path, f"case_{index:04d}")
    assert [p.name for p in select_cases(tmp_path, None, 5)] == [f"case_{i:04d}" for i in range(5)]
    assert [p.name for p in select_cases(tmp_path, ["case_0003", "case_0001"], None)] == ["case_0003", "case_0001"]
    assert parse_args(["--no-llm"]).limit == 5
    assert parse_args(["--no-llm", "--all-cases"]).limit is None
    with pytest.raises(SystemExit, match="cannot be combined"):
        parse_args(["--no-llm", "--all-cases", "--limit", "5"])
    with pytest.raises(SystemExit, match="requires --full-dataset-llm"):
        parse_args([])


class _Responses:
    def __init__(self, responses=None, error: Exception | None = None):
        self.items = list(responses or []); self.error = error; self.calls = 0
    def create(self, **kwargs):
        self.calls += 1
        if self.error: raise self.error
        return self.items.pop(0)


def _response(output_text: str = "", output=None, response_id="r1", in_tokens=10, out_tokens=5):
    return SimpleNamespace(output_text=output_text, output=output or [], id=response_id,
                           usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens))


def _deterministic_result(tmp_path: Path) -> CaseResult:
    return ValidationPipeline().deterministic(_case(tmp_path))


def test_mocked_agent_tool_call_and_structured_result(tmp_path: Path) -> None:
    tool_call = SimpleNamespace(type="function_call", name="get_file_inventory", arguments="{}", call_id="c1")
    finding = {"summary": "Review evidence", "observed_facts": ["No logs"], "hypotheses": [],
               "recommendations": ["Restore logs"], "additional_issues": []}
    responses = _Responses([_response(output=[tool_call]), _response(json.dumps(finding), response_id="r2")])
    client = SimpleNamespace(responses=responses)
    diagnostic, usage = run_agent(_deterministic_result(tmp_path), AgentConfig(max_retries=0),
                                  tmp_path / "cache", RunBudget(1.0), client=client)
    assert diagnostic.performed and diagnostic.summary == "Review evidence"
    assert usage.api_calls == 2 and responses.calls == 2


def test_agent_timeout_retry_and_auth_no_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.cfd_validation_agent.llm_agent.time.sleep", lambda _: None)
    responses = _Responses(error=TimeoutError("timeout")); client = SimpleNamespace(responses=responses)
    diagnostic, usage = run_agent(_deterministic_result(tmp_path), AgentConfig(max_retries=2),
                                  tmp_path / "cache1", RunBudget(1.0), client=client)
    assert "TimeoutError" in (diagnostic.error or "") and responses.calls == 3 and usage.api_calls == 3
    auth = type("AuthenticationError", (Exception,), {"status_code": 401})("bad key")
    responses = _Responses(error=auth); client = SimpleNamespace(responses=responses)
    diagnostic, _ = run_agent(_deterministic_result(tmp_path / "other"), AgentConfig(max_retries=2),
                              tmp_path / "cache2", RunBudget(1.0), client=client)
    assert "AuthenticationError" in (diagnostic.error or "") and responses.calls == 1


def test_budget_exhaustion_and_dry_run_do_not_call_api(tmp_path: Path) -> None:
    result = _deterministic_result(tmp_path)
    with pytest.raises(BudgetExhausted):
        run_agent(result, AgentConfig(), tmp_path / "cache", RunBudget(0.0), client=SimpleNamespace())
    analyzed = ValidationPipeline().analyze(Path(result.inventory.case_path), llm=True, dry_run=True,
                                            cache_dir=tmp_path / "cache", budget=RunBudget(0.0),
                                            client=SimpleNamespace())
    assert not analyzed.llm_analysis.performed and "not sent" in analyzed.llm_analysis.summary


def test_missing_key_only_blocks_live_llm_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.cfd_validation_agent.llm_agent.load_project_environment", lambda: None)
    result = _deterministic_result(tmp_path)
    diagnostic, usage = run_agent(result, AgentConfig(), tmp_path / "cache", RunBudget(1.0))
    assert diagnostic.error == "OPENAI_API_KEY is not set" and usage.api_calls == 0
    assert ValidationPipeline().deterministic(Path(result.inventory.case_path)).status == result.status


def test_dotenv_loading_precedence_and_credential_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    dotenv_key = "sk-synthetic-dotenv-key"
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={dotenv_key}\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_project_environment(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == dotenv_key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-explicit-key")
    load_project_environment(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == "sk-synthetic-explicit-key"

    secret = os.environ["OPENAI_API_KEY"]
    responses = _Responses(error=RuntimeError(f"request rejected for {secret}"))
    diagnostic, _ = run_agent(_deterministic_result(tmp_path / "case"), AgentConfig(max_retries=0),
                              tmp_path / "cache", RunBudget(1.0), client=SimpleNamespace(responses=responses))
    serialized = diagnostic.model_dump_json()
    assert secret not in serialized and "[REDACTED]" in serialized
