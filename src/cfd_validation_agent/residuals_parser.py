from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .config import ResidualThresholds
from .schemas import FinalStatus, ResidualAnalysis, SeriesStatistics, relative_source

SUPPORTED_FIELDS = {"Ux", "Uy", "Uz", "p", "k", "omega", "epsilon"}
SOLVER_RE = re.compile(
    r"Solving for\s+(?P<field>Ux|Uy|Uz|p|k|omega|epsilon),\s*"
    r"Initial residual\s*=\s*(?P<initial>[^,]+),\s*Final residual\s*=\s*(?P<final>[^,]+)", re.I,
)


def _number(value: str) -> float:
    if value.strip().lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "n/a"}:
        return float("nan")
    return float(value)


def parse_residual_table(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    times: list[float] = []
    fields: list[str] | None = None
    values: dict[str, list[float]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").split()
            if tokens and tokens[0].lower() == "time":
                fields = tokens[1:]
                values = {field: [] for field in fields if field in SUPPORTED_FIELDS}
            continue
        if fields is None:
            continue
        tokens = stripped.split()
        if len(tokens) < len(fields) + 1 and tokens[1:] and all(
            token.lower() == "n/a" for token in tokens[1:]
        ):
            tokens.extend(["N/A"] * (len(fields) + 1 - len(tokens)))
        if len(tokens) != len(fields) + 1:
            raise ValueError(f"{path}:{line_number}: expected {len(fields) + 1} columns, got {len(tokens)}")
        times.append(_number(tokens[0]))
        for field, token in zip(fields, tokens[1:]):
            if field in values:
                values[field].append(_number(token))
    if fields is None:
        raise ValueError(f"No '# Time ...' residual header in {path}")
    return times, values


def parse_solver_log(path: Path) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]]]:
    times: list[float] = []
    initial: dict[str, list[float]] = {}
    final: dict[str, list[float]] = {}
    current_time = float(len(times))
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        time_match = re.match(r"\s*Time\s*=\s*([^\s]+)", raw)
        if time_match:
            try:
                current_time = float(time_match.group(1))
            except ValueError:
                current_time = float(len(times))
        match = SOLVER_RE.search(raw)
        if match:
            field = match.group("field")
            initial.setdefault(field, []).append(_number(match.group("initial")))
            final.setdefault(field, []).append(_number(match.group("final")))
            times.append(current_time)
    return times, initial, final


def series_statistics(values: list[float], tail_window: int, *, logarithmic: bool,
                      spike_mad_z: float = 8.0, spike_min_ratio: float = 1.0) -> SeriesStatistics:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    result = SeriesStatistics(samples=int(array.size), finite_samples=int(finite.size))
    if finite.size == 0:
        return result
    tail = finite[-min(tail_window, finite.size):]
    x = np.arange(tail.size, dtype=float)
    result.mean = float(np.mean(finite)); result.std = float(np.std(finite))
    result.minimum = float(np.min(finite)); result.maximum = float(np.max(finite))
    result.tail_mean = float(np.mean(tail)); result.tail_std = float(np.std(tail))
    result.tail_peak_to_peak = float(np.ptp(tail))
    result.linear_slope = float(np.polyfit(x, tail, 1)[0]) if tail.size > 1 else 0.0
    positive = np.maximum(np.abs(tail), np.finfo(float).tiny)
    result.log_slope = float(np.polyfit(x, np.log10(positive), 1)[0]) if logarithmic and tail.size > 1 else None
    half = max(1, tail.size // 2)
    result.window_shift = float(abs(np.mean(tail[-half:]) - np.mean(tail[:half])))
    centered = tail - np.mean(tail)
    signs = np.sign(centered)
    result.oscillation_score = float(np.mean(signs[1:] != signs[:-1])) if tail.size > 1 else 0.0
    spike_values = np.log10(positive) if logarithmic else tail
    if spike_values.size > 1:
        trend = np.polyval(np.polyfit(x, spike_values, 1), x)
        deviations = spike_values - trend
    else:
        deviations = spike_values - spike_values[0]
    median = float(np.median(deviations)); mad = float(np.median(np.abs(deviations - median)))
    threshold = spike_mad_z * max(mad, np.finfo(float).eps)
    if logarithmic:
        threshold = max(threshold, math.log10(spike_min_ratio))
    result.spike_count = int(np.count_nonzero(np.abs(deviations - median) > threshold))
    return result


def analyze_residuals(case_dir: Path, residual_paths: list[Path], log_paths: list[Path], cfg: ResidualThresholds) -> ResidualAnalysis:
    combined: dict[str, list[float]] = {}
    sources: list[str] = []
    notes: list[str] = []
    for path in residual_paths:
        try:
            _, fields = parse_residual_table(path)
            for field, values in fields.items():
                combined.setdefault(field, []).extend(values)
            sources.append(relative_source(path, case_dir))
        except (OSError, ValueError) as exc:
            notes.append(str(exc))
    if not combined:
        for path in log_paths:
            if "simplefoam" not in path.name.lower():
                continue
            try:
                _, initial, _ = parse_solver_log(path)
                for field, values in initial.items():
                    combined.setdefault(field, []).extend(values)
                sources.append(relative_source(path, case_dir))
            except OSError as exc:
                notes.append(str(exc))
    if not combined:
        return ResidualAnalysis(status=FinalStatus.REVIEW, source_files=sources,
                                notes=[*notes, "No parseable residual history was found."])
    stats = {
        field: series_statistics(values, cfg.tail_window, logarithmic=True,
                                 spike_mad_z=cfg.spike_mad_z, spike_min_ratio=cfg.spike_min_ratio)
        for field, values in combined.items()
    }
    status = FinalStatus.PASS
    for field, item in stats.items():
        raw = np.asarray(combined[field], dtype=float)
        finite_positions = np.flatnonzero(np.isfinite(raw))
        has_internal_non_finite = bool(
            finite_positions.size and np.any(~np.isfinite(raw[finite_positions[0]:]))
        )
        if has_internal_non_finite:
            status = FinalStatus.FAIL; notes.append(f"{field}: non-finite residual values detected")
        elif item.finite_samples < cfg.min_samples:
            status = FinalStatus.REVIEW; notes.append(f"{field}: only {item.finite_samples} finite samples")
        elif item.log_slope is not None and item.log_slope > 0.002:
            status = FinalStatus.FAIL; notes.append(f"{field}: positive tail log-slope {item.log_slope:.3g}")
        elif item.log_slope is not None and abs(item.log_slope) < cfg.stagnation_log_slope_abs:
            threshold = cfg.acceptable_initial_residuals.get(field)
            tail_magnitude = abs(item.tail_mean or 0.0)
            if threshold is None or tail_magnitude > threshold:
                if status == FinalStatus.PASS: status = FinalStatus.REVIEW
                notes.append(
                    f"{field}: high residual plateau; tail mean={tail_magnitude:.3g}, "
                    f"threshold={threshold if threshold is not None else 'not configured'}, "
                    f"log-slope={item.log_slope:.3g}"
                )
            else:
                notes.append(
                    f"{field}: acceptable low residual plateau; tail mean={tail_magnitude:.3g} "
                    f"<= threshold={threshold:.3g}, log-slope={item.log_slope:.3g}"
                )
        relative_tail_std = (item.tail_std or 0.0) / max(abs(item.tail_mean or 0.0), np.finfo(float).eps)
        if item.oscillation_score is not None and item.oscillation_score > cfg.oscillation_cv and relative_tail_std > 0.1:
            if status == FinalStatus.PASS: status = FinalStatus.REVIEW
            notes.append(f"{field}: oscillatory tail (sign-change score={item.oscillation_score:.3g})")
        if item.spike_count and relative_tail_std > 0.1:
            if status == FinalStatus.PASS: status = FinalStatus.REVIEW
            notes.append(f"{field}: {item.spike_count} significant tail spike(s) detected")
    return ResidualAnalysis(status=status, source_files=sources, fields=stats, notes=notes)
