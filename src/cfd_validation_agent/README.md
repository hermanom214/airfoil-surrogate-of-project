# OpenFOAM CFD Validation Agent

This package performs read-only engineering validation of archived OpenFOAM cases. Deterministic parsers and rules remain authoritative; an optional OpenAI diagnostic pass can investigate bounded evidence and suggest likely causes or corrective actions. Numerical convergence is not proof of physical correctness.

## Architecture and data flow

`case_discovery` inventories cases and dynamically locates logs and function-object output. `log_parser`, `residuals_parser`, and `force_coeffs_parser` convert evidence into structured metrics. `validation_rules` produces PASS, FAIL, or REVIEW without LLM involvement. `llm_agent` optionally submits a bounded structured evidence packet. `reporting` writes per-case JSON, aggregate JSON, Markdown, and CSV outside the simulation dataset.

The tool never executes OpenFOAM, writes into cases, or exposes arbitrary file/command tools to the model. Log content is treated as untrusted text.

## Supported evidence

- Every `*.log` below a case, independent of filename.
- OpenFOAM solver lines for `Ux`, `Uy`, `Uz`, `p`, `k`, `omega`, and `epsilon`.
- Dynamically discovered `postProcessing/**/residuals.dat` tables.
- Dynamically discovered `postProcessing/**/forceCoeffs.dat` files, including multiple restart directories. Duplicate time values use the later sorted source.
- `checkMesh` failures, fatal errors, non-finite values, abnormal termination, mesh warnings, continuity messages, and bounded source excerpts.

For steady `simpleFoam`, `Time` is explicitly treated as iteration/pseudo-time.

## Validation criteria

Hard failures include explicit OpenFOAM fatal errors, floating-point/segmentation failures, non-finite solver values, illegal/negative volumes, and explicit solver failure. The LLM cannot override them.

Residual checks use finite-sample completeness, tail log-linear trend, robust spike detection, and sign-change oscillation. A stagnant tail is acceptable only when its mean initial residual is below the configured equation-specific level (`p`: `1e-3`; velocity, `k`: `1e-4`; `omega`, `epsilon`: `1e-5`). These are screening levels for application-level initial residual histories, not linear-solver final-residual tolerances. A plateau above its level remains REVIEW, and divergence, non-finite values, oscillation, or significant spikes remain effective regardless of force stability.

`checkMesh` metrics are parsed numerically rather than matched as keywords. The project defaults are maximum/average non-orthogonality of 85/35 degrees and maximum skewness of 4.0. The skewness limit follows the standard OpenFOAM internal-cell quality limit; the non-orthogonality limits document the archived quasi-2D mesh envelope and should be tightened if the mesh workflow improves. The precise starred `High aspect ratio cells found` failed check is retained as an informational quasi-2D exception; no exemption is inferred merely from a count of one failed check. Missing `checkMesh` evidence produces REVIEW.

Known Windows-port environment notices and the explicit compressed-binary fallback are retained as informational evidence. Other OpenFOAM warning blocks remain REVIEW, while fatal errors and execution failures remain hard failures.

Force convergence uses the final 100 samples by default and combines absolute standard deviation, scale-aware standard deviation, peak-to-peak amplitude, scale-aware slope, and consecutive half-window shift. The coefficient scale floor avoids misleading relative errors near zero. Missing or short histories produce REVIEW rather than fabricated convergence.

Defaults are conservative screening thresholds, not universal aerodynamic tolerances. They are defined in `config.py` and should be calibrated against independently reviewed cases for each solver/setup.

## Safe five-case workflow

Deterministic only:

```powershell
python -m src.cfd_validation_agent.cli --limit 5 --no-llm
```

Prepare the exact bounded evidence without an API call:

```powershell
python -m src.cfd_validation_agent.cli --limit 5 --dry-run
```

Exactly five manual cases:

```powershell
python -m src.cfd_validation_agent.cli --case-ids case_0001_naca0008_aoam4p0_u15p0 case_0002_naca0008_aoam4p0_u17p5 case_0003_naca0008_aoam4p0_u20p0 case_0004_naca0008_aoam4p0_u22p5 case_0005_naca0008_aoam4p0_u25p0 --dry-run
```

## OpenAI setup and cost controls

Set `OPENAI_API_KEY`; it is never logged. The agent loads a project-root `.env` with `python-dotenv`, while an explicit process environment variable takes precedence. Copy `.env.example` to `.env` and edit it locally, or configure the current PowerShell session directly:

```powershell
Copy-Item .env.example .env
# Edit .env and replace the placeholder, without committing the file.

# Alternatively, for this PowerShell process only:
$env:OPENAI_API_KEY = "your-key-entered-locally"
```

Verify presence without displaying the credential:

```powershell
if ($env:OPENAI_API_KEY) { "OPENAI_API_KEY is configured" } else { "OPENAI_API_KEY is not configured in this process; the agent may still load .env" }
python -c "from src.cfd_validation_agent.environment import load_project_environment; import os; load_project_environment(); print('OPENAI_API_KEY is configured' if os.getenv('OPENAI_API_KEY') else 'OPENAI_API_KEY is missing')"
```

The default model is `gpt-4.1-mini` and can be changed with `--model`. Requests have explicit timeout/retry limits, bounded input/output, structured Pydantic output, usage/latency/cost logging, a run budget (`--max-cost-usd`), and evidence/model/prompt-version caching.

First live five-case test:

```powershell
python -m src.cfd_validation_agent.cli --limit 5 --max-cost-usd 0.10
```

Live LLM processing of the unrestricted dataset is rejected unless `--full-dataset-llm` is supplied. A deterministic unrestricted run uses `--no-llm --all-cases`; prefer an explicit `--limit` during development.

## Outputs and limitations

The default output is `outputs/cfd_validation_agent/`, separate from simulation data. Physical setup correctness, turbulence-model suitability, discretization independence, boundary-condition correctness, and experimental agreement cannot be established from convergence histories alone. Mesh warnings are setup-dependent and generally trigger REVIEW unless they prove invalid geometry.
