from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ForceThresholds
from .residuals_parser import series_statistics
from .schemas import FinalStatus, ForceAnalysis, relative_source

SUPPORTED = {"Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"}


def parse_force_coeffs(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    columns: list[str] | None = None
    times: list[float] = []
    values: dict[str, list[float]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = stripped.lstrip("#").split()
            if tokens and tokens[0].lower() == "time":
                columns = tokens[1:]
                values = {name: [] for name in columns if name in SUPPORTED}
            continue
        if columns is None:
            continue
        tokens = stripped.split()
        if len(tokens) != len(columns) + 1:
            raise ValueError(f"{path}:{line_number}: malformed forceCoeffs row")
        try:
            times.append(float(tokens[0]))
            row = [float(token) for token in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric forceCoeffs row") from exc
        for name, value in zip(columns, row):
            if name in values:
                values[name].append(value)
    if columns is None:
        raise ValueError(f"No '# Time ...' forceCoeffs header in {path}")
    return times, values


def _merge_restarts(paths: list[Path]) -> tuple[dict[str, list[float]], list[str]]:
    rows: dict[float, dict[str, float]] = {}
    notes: list[str] = []
    for path in sorted(paths):
        try:
            times, fields = parse_force_coeffs(path)
        except (OSError, ValueError) as exc:
            notes.append(str(exc)); continue
        for index, time_value in enumerate(times):
            rows[time_value] = {field: values[index] for field, values in fields.items()}
    merged: dict[str, list[float]] = {}
    for time_value in sorted(rows):
        for field, value in rows[time_value].items():
            merged.setdefault(field, []).append(value)
    return merged, notes


def analyze_force_coeffs(case_dir: Path, paths: list[Path], cfg: ForceThresholds) -> ForceAnalysis:
    if not paths:
        return ForceAnalysis(status=FinalStatus.REVIEW, notes=["No forceCoeffs data were found."])
    merged, notes = _merge_restarts(paths)
    sources = [relative_source(path, case_dir) for path in paths]
    if not merged:
        return ForceAnalysis(status=FinalStatus.REVIEW, source_files=sources,
                             notes=[*notes, "No parseable force coefficients were found."])
    stats = {name: series_statistics(values, cfg.tail_window, logarithmic=False) for name, values in merged.items()}
    status = FinalStatus.PASS
    for name in ("Cd", "Cl", "Cm"):
        item = stats.get(name)
        if item is None:
            if status == FinalStatus.PASS: status = FinalStatus.REVIEW
            notes.append(f"{name}: coefficient not present"); continue
        if item.finite_samples < item.samples:
            status = FinalStatus.FAIL; notes.append(f"{name}: non-finite values detected"); continue
        if item.finite_samples < cfg.min_samples:
            if status == FinalStatus.PASS: status = FinalStatus.REVIEW
            notes.append(f"{name}: only {item.finite_samples} samples"); continue
        scale = max(abs(item.tail_mean or 0.0), cfg.scale_floor)
        scaled_std = (item.tail_std or 0.0) / scale
        scaled_slope = abs(item.linear_slope or 0.0) / scale
        window_shift = (item.window_shift or 0.0) / scale
        unstable = ((item.tail_std or 0.0) > cfg.absolute_std_limit and scaled_std > cfg.scaled_std_limit)
        unstable |= (item.tail_peak_to_peak or 0.0) > cfg.peak_to_peak_limit
        unstable |= scaled_slope > cfg.scaled_slope_limit
        unstable |= window_shift > cfg.window_shift_limit
        if unstable:
            if status == FinalStatus.PASS: status = FinalStatus.REVIEW
            notes.append(f"{name}: tail is not demonstrably stable (scaled std={scaled_std:.3g}, slope={scaled_slope:.3g})")
    return ForceAnalysis(status=status, source_files=sources, coefficients=stats, notes=notes)
