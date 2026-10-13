from __future__ import annotations

from pathlib import Path

from .case_discovery import inventory_case
from .config import ValidationConfig
from .force_coeffs_parser import analyze_force_coeffs
from .llm_agent import BudgetExhausted, RunBudget, prepare_evidence_payload, run_agent
from .log_parser import parse_all_logs
from .residuals_parser import analyze_residuals
from .schemas import CaseResult, FinalStatus, LLMDiagnostic, LLMUsage, Severity, ValidationIssue
from .validation_rules import evaluate_log_findings, final_status


class ValidationPipeline:
    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def deterministic(self, case_dir: Path) -> CaseResult:
        inventory = inventory_case(case_dir)
        logs = [case_dir / path for path in inventory.log_files]
        residual_files = [case_dir / path for path in inventory.residual_files]
        force_files = [case_dir / path for path in inventory.force_coeff_files]
        evidence = parse_all_logs(case_dir, logs, self.config.excerpt_context_lines,
                                  self.config.max_evidence_excerpt_chars, self.config.mesh)
        log_status, issues = evaluate_log_findings(evidence)
        residuals = analyze_residuals(case_dir, residual_files, logs, self.config.residuals)
        force = analyze_force_coeffs(case_dir, force_files, self.config.force_coeffs)
        if inventory.empty_files:
            issues.append(ValidationIssue(
                type="empty_files", severity=Severity.HIGH, source=inventory.empty_files[0],
                explanation=f"{len(inventory.empty_files)} required evidence file(s) are empty.",
                recommendation="Restore or regenerate the missing function-object/log output.",
            ))
            if log_status == FinalStatus.PASS: log_status = FinalStatus.REVIEW
        components = {"logs": log_status, "mesh": _mesh_status(evidence),
                      "residuals": residuals.status, "force_coeffs": force.status}
        return CaseResult(
            case_id=case_dir.name, status=final_status(components, issues),
            deterministic_validation=components, inventory=inventory, evidence=evidence,
            residuals=residuals, force_coeffs=force, issues=issues,
        )

    def analyze(self, case_dir: Path, *, llm: bool, dry_run: bool, cache_dir: Path,
                budget: RunBudget, client=None) -> CaseResult:
        result = self.deterministic(case_dir)
        if dry_run:
            result.llm_analysis = LLMDiagnostic(model=self.config.agent.model,
                                                summary=("Dry run: bounded LLM request prepared but not sent "
                                                         f"({len(prepare_evidence_payload(result, self.config.agent))} characters)."))
        elif llm:
            try:
                diagnostic, usage = run_agent(result, self.config.agent, cache_dir, budget, client=client)
                result.llm_analysis = diagnostic; result.usage = usage
                valid_ids = {item.evidence_id for item in result.evidence}
                for issue in diagnostic.additional_issues:
                    if issue.evidence_ids and set(issue.evidence_ids).issubset(valid_ids):
                        result.issues.append(issue.model_copy(update={"deterministic_hard_failure": False}))
            except BudgetExhausted as exc:
                result.llm_analysis = LLMDiagnostic(model=self.config.agent.model, error=str(exc))
        result.status = final_status(result.deterministic_validation, result.issues)
        return result


def _mesh_status(evidence) -> FinalStatus:
    categories = {item.category for item in evidence}
    if "negative_volume" in categories: return FinalStatus.FAIL
    if categories & {"mesh_failed", "mesh_failure_ambiguous", "mesh_non_orthogonality_exceeded",
                     "mesh_skewness_exceeded", "missing_mesh_diagnostics"}: return FinalStatus.REVIEW
    return FinalStatus.PASS
