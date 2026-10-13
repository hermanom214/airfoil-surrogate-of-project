from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResidualThresholds:
    min_samples: int = 30
    tail_window: int = 100
    divergence_ratio: float = 10.0
    stagnation_log_slope_abs: float = 1e-4
    oscillation_cv: float = 0.5
    spike_mad_z: float = 8.0
    spike_min_ratio: float = 2.0
    acceptable_initial_residuals: dict[str, float] = field(default_factory=lambda: {
        "p": 1e-3, "Ux": 1e-4, "Uy": 1e-4, "Uz": 1e-4,
        "k": 1e-4, "omega": 1e-5, "epsilon": 1e-5,
    })


@dataclass(frozen=True)
class MeshThresholds:
    max_non_orthogonality: float = 85.0
    max_average_non_orthogonality: float = 35.0
    max_skewness: float = 4.0


@dataclass(frozen=True)
class ForceThresholds:
    min_samples: int = 50
    tail_window: int = 100
    absolute_std_limit: float = 5e-3
    scaled_std_limit: float = 2e-2
    peak_to_peak_limit: float = 2e-2
    scaled_slope_limit: float = 2e-4
    window_shift_limit: float = 2e-2
    scale_floor: float = 5e-2


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    request_timeout_seconds: float = 45.0
    max_retries: int = 2
    max_prompt_chars: int = 30_000
    max_output_tokens: int = 1_200
    max_steps: int = 4
    max_api_calls_per_case: int = 4
    max_tool_calls_per_case: int = 8
    input_cost_per_million: float = 0.40
    output_cost_per_million: float = 1.60
    prompt_version: str = "cfd-validation-v1"


@dataclass(frozen=True)
class ValidationConfig:
    residuals: ResidualThresholds = field(default_factory=ResidualThresholds)
    mesh: MeshThresholds = field(default_factory=MeshThresholds)
    force_coeffs: ForceThresholds = field(default_factory=ForceThresholds)
    agent: AgentConfig = field(default_factory=AgentConfig)
    excerpt_context_lines: int = 2
    max_evidence_excerpt_chars: int = 1_200
