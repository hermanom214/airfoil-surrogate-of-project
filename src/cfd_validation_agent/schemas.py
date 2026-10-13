from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FinalStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    category: str
    severity: Severity
    source: str
    line_number: int | None = None
    message: str
    context: str = ""
    explicit: bool = True


class SeriesStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: int
    finite_samples: int
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    tail_mean: float | None = None
    tail_std: float | None = None
    tail_peak_to_peak: float | None = None
    log_slope: float | None = None
    linear_slope: float | None = None
    window_shift: float | None = None
    oscillation_score: float | None = None
    spike_count: int = 0


class ResidualAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: FinalStatus
    source_files: list[str] = Field(default_factory=list)
    pseudo_time: bool = True
    fields: dict[str, SeriesStatistics] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ForceAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: FinalStatus
    source_files: list[str] = Field(default_factory=list)
    coefficients: dict[str, SeriesStatistics] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class FileInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_path: str
    log_files: list[str] = Field(default_factory=list)
    residual_files: list[str] = Field(default_factory=list)
    force_coeff_files: list[str] = Field(default_factory=list)
    empty_files: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    severity: Severity
    source: str
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str
    recommendation: str = ""
    deterministic_hard_failure: bool = False


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    request_latency_seconds: float = 0.0
    api_calls: int = 0


class LLMDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    performed: bool = False
    model: str | None = None
    summary: str = ""
    observed_facts: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    additional_issues: list[ValidationIssue] = Field(default_factory=list)
    error: str | None = None
    cached: bool = False


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    status: FinalStatus
    deterministic_validation: dict[str, FinalStatus]
    inventory: FileInventory
    evidence: list[EvidenceItem] = Field(default_factory=list)
    residuals: ResidualAnalysis
    force_coeffs: ForceAnalysis
    issues: list[ValidationIssue] = Field(default_factory=list)
    llm_analysis: LLMDiagnostic = Field(default_factory=LLMDiagnostic)
    usage: LLMUsage = Field(default_factory=LLMUsage)


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_cases: int
    pass_count: int
    fail_count: int
    review_count: int
    failure_categories: dict[str, int]
    case_ids: list[str]
    total_usage: LLMUsage


class AgentFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    observed_facts: list[str]
    hypotheses: list[str]
    recommendations: list[str]
    additional_issues: list[ValidationIssue] = Field(default_factory=list)


def relative_source(path: Path, case_dir: Path) -> str:
    try:
        return path.relative_to(case_dir).as_posix()
    except ValueError:
        return str(path)
