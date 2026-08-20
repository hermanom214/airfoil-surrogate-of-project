from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SearchConfiguration:
    config_id: int
    hyperparameters: dict[str, Any]


def generate_search_configurations(
    strategy: str, base_config: Mapping[str, Any], *,
    search_space: Mapping[str, Sequence[Any]] | None = None,
    manual_coarse_configs: Sequence[Mapping[str, Any]] | None = None,
    n_iter: int = 1, seed: int = 42,
) -> list[SearchConfiguration]:
    """Create deterministic candidates without coupling PyTorch to sklearn."""
    strategy = strategy.lower().strip()
    base = dict(base_config)
    if strategy == "none":
        return [SearchConfiguration(0, base)]
    if strategy == "manual_coarse":
        configs = list(manual_coarse_configs or [])
        if not configs:
            raise ValueError("manual_coarse requires manual_coarse_configs")
        return [SearchConfiguration(i, base | dict(v)) for i, v in enumerate(configs)]
    if strategy != "randomized":
        raise ValueError("strategy must be none, randomized, or manual_coarse")
    space = dict(search_space or {})
    if not space:
        raise ValueError("randomized search requires a non-empty search_space")
    keys = sorted(space)
    values = [list(space[key]) for key in keys]
    if any(not choices for choices in values):
        raise ValueError("search_space values must be non-empty lists")
    combinations = [dict(zip(keys, choice)) for choice in itertools.product(*values)]
    rng = random.Random(seed)
    rng.shuffle(combinations)
    selected = combinations[:min(int(n_iter), len(combinations))]
    return [SearchConfiguration(i, base | config) for i, config in enumerate(selected)]


def summarize_cv_results(fold_metrics: Sequence[float]) -> tuple[float, float]:
    if not fold_metrics:
        raise ValueError("fold_metrics cannot be empty")
    mean = sum(float(v) for v in fold_metrics) / len(fold_metrics)
    variance = sum((float(v) - mean) ** 2 for v in fold_metrics) / len(fold_metrics)
    return mean, variance ** 0.5


def serialize_hyperparameters(values: Mapping[str, Any]) -> str:
    return json.dumps(dict(values), sort_keys=True, separators=(",", ":"))
