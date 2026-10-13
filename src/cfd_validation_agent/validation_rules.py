from __future__ import annotations

from collections import Counter

from .schemas import EvidenceItem, FinalStatus, Severity, ValidationIssue

HARD_FAILURE_CATEGORIES = {"fatal_error", "non_finite", "negative_volume", "solver_failure"}


def evaluate_log_findings(evidence: list[EvidenceItem]) -> tuple[FinalStatus, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    grouped: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        grouped.setdefault(item.category, []).append(item)
    for category, items in grouped.items():
        hard = category in HARD_FAILURE_CATEGORIES
        if category in {"continuity", "expected_2d_mesh_check", "expected_warning",
                        "mesh_non_orthogonality_metric", "mesh_skewness_metric"}:
            continue
        severity = max((item.severity for item in items), key=lambda value: list(Severity).index(value))
        issues.append(ValidationIssue(
            type=category, severity=severity, source=items[0].source,
            evidence_ids=[item.evidence_id for item in items[:10]],
            explanation=(
                "checkMesh reported failed checks, but their identities could not be reconciled "
                "reliably with the reported count."
                if category == "mesh_failure_ambiguous"
                else f"{len(items)} {category.replace('_', ' ')} finding(s) detected."
            ),
            recommendation="Inspect the cited bounded excerpts and the originating OpenFOAM file.",
            deterministic_hard_failure=hard,
        ))
    if any(issue.deterministic_hard_failure for issue in issues):
        return FinalStatus.FAIL, issues
    if any(issue.severity in {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL} for issue in issues):
        return FinalStatus.REVIEW, issues
    return FinalStatus.PASS, issues


def final_status(components: dict[str, FinalStatus], issues: list[ValidationIssue]) -> FinalStatus:
    if any(issue.deterministic_hard_failure for issue in issues) or FinalStatus.FAIL in components.values():
        return FinalStatus.FAIL
    if FinalStatus.REVIEW in components.values():
        return FinalStatus.REVIEW
    if any(issue.severity in {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL} for issue in issues):
        return FinalStatus.REVIEW
    return FinalStatus.PASS


def category_counts(results: list) -> dict[str, int]:
    return dict(Counter(issue.type for result in results for issue in result.issues))
