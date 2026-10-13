from __future__ import annotations

import re
from pathlib import Path

from .config import MeshThresholds
from .evidence import make_evidence
from .schemas import EvidenceItem, Severity, relative_source

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PATTERNS: list[tuple[str, Severity, re.Pattern[str], bool]] = [
    ("fatal_error", Severity.CRITICAL, re.compile(r"FOAM FATAL|fatal error|floating point exception|segmentation fault", re.I), True),
    ("non_finite", Severity.CRITICAL, re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf)(?![A-Za-z])", re.I), True),
    ("negative_volume", Severity.CRITICAL, re.compile(r"negative (?:cell )?volume|illegal cell volume", re.I), True),
    ("solver_failure", Severity.HIGH, re.compile(r"solver failed|diverg|cannot continue|aborting", re.I), True),
    ("courant", Severity.MEDIUM, re.compile(r"Courant Number.*(?:max|mean)", re.I), False),
    ("continuity", Severity.INFO, re.compile(r"continuity errors", re.I), False),
]

NON_ORTH_RE = re.compile(r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)\s+average:\s*([0-9.eE+-]+)", re.I)
SKEWNESS_RE = re.compile(r"Max skewness\s*=\s*([0-9.eE+-]+)", re.I)
WARNING_RE = re.compile(r"(?:FOAM )?Warning", re.I)
EXPECTED_WARNING_CONTEXTS = (
    re.compile(r"Windows Subsystem for Linux has been disabled", re.I),
    re.compile(r"Selecting compressed binary is inefficient.*resetting to uncompressed binary", re.I | re.S),
)

MESH_FAILURE_SUMMARY_RE = re.compile(r"^\s*Failed\s+(\d+)\s+mesh\s+checks?\.?\s*$", re.I)
MESH_FAILED_CHECK_RE = re.compile(r"^\s*\*{3}\s*(\S.*?)\s*$")
EXPECTED_2D_ASPECT_RATIO_RE = re.compile(
    r"^High\s+aspect\s+ratio\s+cells\s+found(?:\s*,\s*.*)?$", re.I
)


def _context(lines: list[str], index: int, radius: int, limit: int) -> str:
    excerpt = "\n".join(lines[max(0, index - radius): min(len(lines), index + radius + 1)])
    return excerpt[:limit]


def _parse_mesh_check_failures(
    lines: list[str], source: str, context_lines: int, excerpt_limit: int,
) -> list[EvidenceItem]:
    """Classify checkMesh's failed-check markers against its final count."""
    summaries = [
        (index, int(match.group(1)))
        for index, line in enumerate(lines)
        if (match := MESH_FAILURE_SUMMARY_RE.match(line))
    ]
    if not summaries:
        return []

    summary_index, failed_count = summaries[-1]
    if failed_count == 0:
        return []

    marked = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines[:summary_index])
        if (match := MESH_FAILED_CHECK_RE.match(line))
    ]
    findings: list[EvidenceItem] = []
    for index, diagnostic in marked:
        expected = EXPECTED_2D_ASPECT_RATIO_RE.fullmatch(diagnostic) is not None
        findings.append(make_evidence(
            source=source,
            line_number=index + 1,
            category="expected_2d_mesh_check" if expected else "mesh_failed",
            severity=Severity.INFO if expected else Severity.HIGH,
            message=lines[index].strip()[:500],
            context=_context(lines, index, context_lines, excerpt_limit),
            explicit=True,
        ))

    if len(marked) != failed_count:
        findings.append(make_evidence(
            source=source,
            line_number=summary_index + 1,
            category="mesh_failure_ambiguous",
            severity=Severity.HIGH,
            message=lines[summary_index].strip()[:500],
            context=_context(lines, summary_index, context_lines, excerpt_limit),
            explicit=True,
        ))
    return findings


def parse_log_file(path: Path, case_dir: Path, context_lines: int = 2, excerpt_limit: int = 1200,
                   mesh_cfg: MeshThresholds | None = None) -> list[EvidenceItem]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [make_evidence(source=relative_source(path, case_dir), category="unreadable_file",
                              severity=Severity.HIGH, message=str(exc))]
    lines = [ANSI_RE.sub("", line) for line in raw.splitlines()]
    source = relative_source(path, case_dir)
    evidence: list[EvidenceItem] = []
    mesh_cfg = mesh_cfg or MeshThresholds()
    seen: set[tuple[int, str]] = set()
    for index, line in enumerate(lines):
        for category, severity, pattern, explicit in PATTERNS:
            if pattern.search(line) and (index, category) not in seen:
                seen.add((index, category))
                evidence.append(make_evidence(
                    source=source, line_number=index + 1, category=category,
                    severity=severity, message=line.strip()[:500],
                    context=_context(lines, index, context_lines, excerpt_limit), explicit=explicit,
                ))
        warning_match = WARNING_RE.search(line)
        if warning_match and (index, "warning") not in seen:
            seen.add((index, "warning"))
            block = "\n".join(lines[index:min(len(lines), index + 8)])
            expected = any(pattern.search(block) for pattern in EXPECTED_WARNING_CONTEXTS)
            evidence.append(make_evidence(
                source=source, line_number=index + 1,
                category="expected_warning" if expected else "warning_review",
                severity=Severity.INFO if expected else Severity.MEDIUM,
                message=line.strip()[:500], context=_context(lines, index, 5, excerpt_limit), explicit=True,
            ))
        non_orth = NON_ORTH_RE.search(line)
        if non_orth:
            maximum, average = map(float, non_orth.groups())
            exceeded = maximum > mesh_cfg.max_non_orthogonality or average > mesh_cfg.max_average_non_orthogonality
            evidence.append(make_evidence(
                source=source, line_number=index + 1,
                category="mesh_non_orthogonality_exceeded" if exceeded else "mesh_non_orthogonality_metric",
                severity=Severity.MEDIUM if exceeded else Severity.INFO,
                message=(f"max non-orthogonality={maximum:g} (limit {mesh_cfg.max_non_orthogonality:g}); "
                         f"average={average:g} (limit {mesh_cfg.max_average_non_orthogonality:g})"),
                context=_context(lines, index, context_lines, excerpt_limit), explicit=True,
            ))
        skewness = SKEWNESS_RE.search(line)
        if skewness:
            maximum = float(skewness.group(1)); exceeded = maximum > mesh_cfg.max_skewness
            evidence.append(make_evidence(
                source=source, line_number=index + 1,
                category="mesh_skewness_exceeded" if exceeded else "mesh_skewness_metric",
                severity=Severity.MEDIUM if exceeded else Severity.INFO,
                message=f"max skewness={maximum:g} (limit {mesh_cfg.max_skewness:g})",
                context=_context(lines, index, context_lines, excerpt_limit), explicit=True,
            ))
    evidence.extend(_parse_mesh_check_failures(lines, source, context_lines, excerpt_limit))
    lower_name = path.name.lower()
    expects_end = any(token in lower_name for token in ("simplefoam", "checkmesh", "blockmesh", "reconstruct", "foamtovtk"))
    if expects_end and raw.strip() and not re.search(r"(?m)^End\s*$", ANSI_RE.sub("", raw)):
        evidence.append(make_evidence(
            source=source, category="missing_completion", severity=Severity.MEDIUM,
            message="Expected OpenFOAM completion marker 'End' was not found.", explicit=False,
            context="\n".join(lines[-5:])[:excerpt_limit],
        ))
    return evidence


def parse_all_logs(case_dir: Path, log_paths: list[Path], context_lines: int = 2, excerpt_limit: int = 1200,
                   mesh_cfg: MeshThresholds | None = None) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    if not log_paths:
        return [make_evidence(source="logs", category="missing_logs", severity=Severity.MEDIUM,
                              message="No .log files were discovered.", explicit=True)]
    for path in log_paths:
        if path.stat().st_size == 0:
            evidence.append(make_evidence(source=relative_source(path, case_dir), category="empty_log",
                                          severity=Severity.HIGH, message="Log file is empty."))
        else:
            evidence.extend(parse_log_file(path, case_dir, context_lines, excerpt_limit, mesh_cfg))
    if not any("checkmesh" in path.name.lower() for path in log_paths):
        evidence.append(make_evidence(source="logs", category="missing_mesh_diagnostics", severity=Severity.MEDIUM,
                                      message="No checkMesh log was discovered; mesh quality is unverified."))
    return evidence
