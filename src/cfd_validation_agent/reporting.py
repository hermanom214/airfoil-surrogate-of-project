from __future__ import annotations

import csv
import json
from pathlib import Path

from .schemas import CaseResult, FinalStatus, LLMUsage, RunSummary
from .validation_rules import category_counts


def build_summary(results: list[CaseResult]) -> RunSummary:
    usage = LLMUsage(
        input_tokens=sum(result.usage.input_tokens for result in results),
        output_tokens=sum(result.usage.output_tokens for result in results),
        estimated_cost_usd=sum(result.usage.estimated_cost_usd for result in results),
        request_latency_seconds=sum(result.usage.request_latency_seconds for result in results),
        api_calls=sum(result.usage.api_calls for result in results),
    )
    return RunSummary(
        total_cases=len(results),
        pass_count=sum(result.status == FinalStatus.PASS for result in results),
        fail_count=sum(result.status == FinalStatus.FAIL for result in results),
        review_count=sum(result.status == FinalStatus.REVIEW for result in results),
        failure_categories=category_counts(results), case_ids=[result.case_id for result in results],
        total_usage=usage,
    )


def write_reports(results: list[CaseResult], output_dir: Path) -> RunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"; cases_dir.mkdir(exist_ok=True)
    for result in results:
        (cases_dir / f"{result.case_id}.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    summary = build_summary(results)
    aggregate = {"summary": summary.model_dump(mode="json"),
                 "cases": [result.model_dump(mode="json") for result in results]}
    (output_dir / "results.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case_id", "status", "logs", "residuals", "force_coeffs", "issue_count",
            "llm_performed", "input_tokens", "output_tokens", "estimated_cost_usd",
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "case_id": result.case_id, "status": result.status.value,
                "logs": result.deterministic_validation["logs"].value,
                "residuals": result.residuals.status.value, "force_coeffs": result.force_coeffs.status.value,
                "issue_count": len(result.issues), "llm_performed": result.llm_analysis.performed,
                "input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
                "estimated_cost_usd": f"{result.usage.estimated_cost_usd:.6f}",
            })
    lines = ["# OpenFOAM CFD validation report", "", f"Cases: {summary.total_cases}",
             f"PASS: {summary.pass_count} | FAIL: {summary.fail_count} | REVIEW: {summary.review_count}", "",
             "Numerical convergence is not proof of physical correctness.", "", "## Cases", ""]
    for result in results:
        lines.extend([
            f"### {result.case_id} — {result.status.value}", "",
            f"- Logs: {result.deterministic_validation['logs'].value}",
            f"- Mesh: {result.deterministic_validation['mesh'].value}",
            f"- Residuals: {result.residuals.status.value}",
            f"- Force coefficients: {result.force_coeffs.status.value}",
            f"- LLM analysis: {'yes' if result.llm_analysis.performed else 'no'}", "",
        ])
        for issue in result.issues:
            lines.append(f"- `{issue.type}` ({issue.severity.value}): {issue.explanation}")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
