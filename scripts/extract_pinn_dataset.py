from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pinn_dataset import (  # noqa: E402
    FIELD_UNITS,
    GRID_SHAPE,
    REQUIRED_FIELDS,
    CaseResult,
    airfoil_sdf_from_geometry,
    build_physics_valid_mask,
    case_result_dict,
    parse_openfoam_uniform_nu,
    read_openfoam_binary_internal_field,
    save_case_npz,
    summary_for_arrays,
    validate_arrays,
)
from src.ml_quality_filter import load_inspection_membership  # noqa: E402


SOURCE_ROOT = PROJECT_ROOT / "data" / "flow_fields"
CASE_CONFIG_ROOT = PROJECT_ROOT / "run" / "airfoil_surrogate_cases" / "blockmesh_cases"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "pinn_dataset"
SOURCE_TIME = "1000"
STENCIL_RADIUS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the separate physics-ready RANS dataset.")
    parser.add_argument("--case-id", action="append", default=[], help="Extract only this case (repeatable).")
    parser.add_argument("--validate-only", action="store_true", help="Validate source cases without writing NPZ files.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true", help="Resume an explicitly selected existing output directory.")
    parser.add_argument(
        "--clean-rebuild", action="store_true",
        help="Delete and rebuild only the default generated PINN dataset directory.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Independent extraction processes (default: 1).")
    return parser.parse_args()


def _load_mesh_and_attach_fields(case_dir: Path, field_values: dict[str, np.ndarray]):
    import pyvista as pv

    vtk_files = sorted((case_dir / "VTK").glob("*.vtk"))
    if not vtk_files:
        raise FileNotFoundError(f"No VTK mesh found in {case_dir / 'VTK'}")
    mesh = pv.read(vtk_files[-1])
    for name, values in field_values.items():
        if len(values) != mesh.n_cells:
            raise ValueError(f"{name}: {len(values)} values but VTK mesh has {mesh.n_cells} cells")
        mesh.cell_data[name] = values
    return mesh


def _sample_turbulence(case_dir: Path, xy: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import pyvista as pv

    time_dir = case_dir / SOURCE_TIME
    fields = {name: read_openfoam_binary_internal_field(time_dir / name) for name in ("k", "omega", "nut")}
    mesh = _load_mesh_and_attach_fields(case_dir, fields)
    point_mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    target_points = np.column_stack(
        [xy[..., 0].ravel(), xy[..., 1].ravel(), np.zeros(xy.shape[:2]).ravel()]
    )
    sampled = pv.PolyData(target_points).sample(point_mesh, pass_cell_data=False, pass_point_data=True)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=np.uint8).reshape(xy.shape[:2])
    invalid_flat = valid.ravel() == 0
    nearest_indices = None
    if np.any(invalid_flat):
        # Match the established U-Net convention: retain a finite nearest-neighbour
        # value outside linear interpolation support, while keeping those cells
        # explicitly invalid for future PDE stencils.
        tree = cKDTree(np.asarray(point_mesh.points)[:, :2])
        nearest_indices = tree.query(target_points[invalid_flat, :2], workers=-1)[1]
    output: dict[str, np.ndarray] = {}
    for name in fields:
        values = np.asarray(sampled.point_data[name], dtype=np.float64).copy()
        if nearest_indices is not None:
            values[invalid_flat] = np.asarray(point_mesh.point_data[name])[nearest_indices]
        output[name] = values.reshape(xy.shape[:2]).astype(np.float32)
    return output, valid


def discover_archived_case_ids(source_root: Path = SOURCE_ROOT) -> list[str]:
    return sorted(
        path.name for path in source_root.glob("case_*")
        if path.is_dir() and (path / "VTK").is_dir() and (path / "system").is_dir()
    )


def inspection_candidates(source_root: Path = SOURCE_ROOT):
    archived_ids = discover_archived_case_ids(source_root)
    membership = load_inspection_membership(source_root)
    archived = set(archived_ids)
    inspected = set(membership.case_statuses)
    if archived != inspected:
        raise RuntimeError(
            "Inspection membership does not exactly cover archived CFD cases; "
            f"missing inspection records={sorted(archived - inspected)}, "
            f"unknown inspection records={sorted(inspected - archived)}"
        )
    return archived_ids, membership


def clean_generated_output(output_root: Path, expected_root: Path = OUTPUT_ROOT) -> None:
    resolved = output_root.resolve()
    expected = expected_root.resolve()
    if resolved != expected:
        raise ValueError(
            f"--clean-rebuild may only target the generated PINN dataset {expected}; got {resolved}"
        )
    if resolved == SOURCE_ROOT.resolve() or SOURCE_ROOT.resolve() in resolved.parents:
        raise ValueError("Refusing to clean CFD/U-Net source data")
    if resolved.exists():
        import shutil
        shutil.rmtree(resolved)


def verify_final_artifacts(
    output_root: Path, candidate_ids: list[str], results: list[CaseResult]
) -> dict[str, object]:
    successful = {result.case_id for result in results if result.status == "ok"}
    rejected = {result.case_id for result in results if result.status != "ok"}
    candidates = set(candidate_ids)
    artifact_ids = {
        path.parent.name for path in (output_root / "cases").glob("*/*.npz")
    }
    if successful | rejected != candidates or successful & rejected:
        raise RuntimeError("PINN manifest does not partition all inspection-OK candidates")
    if artifact_ids != successful:
        raise RuntimeError(
            "PINN artifacts do not match successful manifest membership; "
            f"stale_or_unknown={sorted(artifact_ids - successful)}, "
            f"missing={sorted(successful - artifact_ids)}"
        )
    return {
        "all_candidates_accounted_for": True,
        "artifact_membership_matches_manifest": True,
        "stale_artifact_count": 0,
    }


def inspect_case_sources(case_id: str) -> tuple[Path, Path, dict[str, object], float]:
    case_dir = SOURCE_ROOT / case_id
    config_dir = CASE_CONFIG_ROOT / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Archived CFD case missing: {case_dir}")
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Generated source case configuration missing: {config_dir}")
    for field in REQUIRED_FIELDS:
        path = case_dir / SOURCE_TIME / field
        if not path.is_file():
            raise FileNotFoundError(f"Required final-time field missing: {path}")
    params = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
    nu_path = config_dir / "constant" / "transportProperties"
    nu = parse_openfoam_uniform_nu(nu_path)
    return case_dir, config_dir, params, nu


def extract_case(
    case_id: str, output_root: Path, validate_only: bool = False, allow_existing: bool = False
) -> tuple[CaseResult, dict]:
    try:
        output_file = output_root / "cases" / case_id / f"{case_id}.npz"
        if output_file.exists():
            if not allow_existing:
                raise FileExistsError(f"Existing output requires --resume: {output_file}")
            with np.load(output_file, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in (
                    "xy", "fluid_mask", "airfoil_sdf", "interpolation_valid_mask",
                    "physics_valid_mask", "p", "U", "k", "omega", "nut",
                )}
                metadata = json.loads(str(saved["metadata"].item()))
            issues = validate_arrays(arrays)
            if issues:
                return CaseResult(case_id, "invalid", message="; ".join(issues)), {}
            stats = summary_for_arrays(arrays)
            for key in ("nu", "Re"):
                value = float(metadata[key])
                stats[key] = {"min": value, "max": value, "mean": value}
            return CaseResult(case_id, "ok", str(output_file), "resumed existing artifact"), {
                "metadata": metadata, "stats": stats,
            }
        case_dir, config_dir, params, nu = inspect_case_sources(case_id)
        existing_npz = case_dir / f"{case_id}.npz"
        with np.load(existing_npz, allow_pickle=False) as source:
            xy = source["xy"].astype(np.float32)
            p = source["p"].astype(np.float32)
            U = source["U"].astype(np.float32)
            fluid_mask = source["fluid_mask"].astype(np.uint8)
        if xy.shape[:2] != GRID_SHAPE:
            raise ValueError(f"Existing U-Net grid is {xy.shape[:2]}, expected {GRID_SHAPE}")
        turbulence, interpolation_valid = _sample_turbulence(case_dir, xy)
        sdf = airfoil_sdf_from_geometry(
            xy, str(params["naca_code"]), float(params["chord"]), float(params["aoa_deg"])
        )
        # Source interpolation may bridge the airfoil; never consider solid cells valid.
        interpolation_valid = ((interpolation_valid > 0) & (fluid_mask > 0)).astype(np.uint8)
        physics_valid = build_physics_valid_mask(fluid_mask, interpolation_valid, STENCIL_RADIUS)
        arrays = {
            "xy": xy,
            "fluid_mask": fluid_mask,
            "airfoil_sdf": sdf,
            "interpolation_valid_mask": interpolation_valid,
            "physics_valid_mask": physics_valid,
            "p": p,
            "U": U,
            **turbulence,
        }
        issues = validate_arrays(arrays)
        metadata = {
            "case_id": case_id,
            "source_case_id": case_id,
            "source_time": SOURCE_TIME,
            "extractor_version": 2,
            "naca_code": str(params["naca_code"]),
            "camber_percent": float(params["camber_percent"]),
            "camber_position_tenths": float(params["camber_position_tenths"]),
            "thickness_percent": float(params["thickness_percent"]),
            "aoa_deg": float(params["aoa_deg"]),
            "U_inf": float(params["inlet_velocity"]),
            "chord": float(params["chord"]),
            "nu": nu,
            "Re": float(params["inlet_velocity"]) * float(params["chord"]) / nu,
            "nu_source": str((config_dir / "constant" / "transportProperties").relative_to(PROJECT_ROOT)),
            "sdf_sign_convention": "positive in fluid; negative in solid; zero on analytic NACA surface",
            "physics_stencil_radius_pixels": STENCIL_RADIUS,
            "units": FIELD_UNITS,
            "validation_issues": issues,
        }
        stats = summary_for_arrays(arrays)
        stats["nu"] = {"min": nu, "max": nu, "mean": nu}
        stats["Re"] = {"min": metadata["Re"], "max": metadata["Re"], "mean": metadata["Re"]}
        if issues:
            return CaseResult(case_id, "invalid", message="; ".join(issues)), {"metadata": metadata, "stats": stats}
        if not validate_only:
            save_case_npz(output_file, arrays, metadata)
        return CaseResult(case_id, "validated" if validate_only else "ok", str(output_file)), {
            "metadata": metadata,
            "stats": stats,
        }
    except Exception as exc:
        return CaseResult(case_id, "error", message=str(exc)), {}


def _aggregate_stats(details: list[dict]) -> dict[str, dict[str, float]]:
    keys = sorted({key for detail in details for key in detail.get("stats", {})})
    output = {}
    for key in keys:
        rows = [detail["stats"][key] for detail in details if key in detail.get("stats", {})]
        output[key] = {
            "min": min(float(row["min"]) for row in rows),
            "max": max(float(row["max"]) for row in rows),
            "mean_of_case_means": float(np.mean([float(row["mean"]) for row in rows])),
        }
    return output


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root == SOURCE_ROOT.resolve() or SOURCE_ROOT.resolve() in output_root.parents:
        raise ValueError("PINN dataset output must not target the existing U-Net flow_fields directory")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.clean_rebuild:
        if args.resume or args.validate_only or args.case_id:
            raise ValueError("--clean-rebuild cannot be combined with --resume, --validate-only, or --case-id")
        clean_generated_output(output_root)
    if output_root.exists() and not args.validate_only and not args.resume:
        raise FileExistsError(f"Refusing to reuse existing output directory without --resume: {output_root}")
    if not args.validate_only:
        output_root.mkdir(parents=True, exist_ok=args.resume)
    archived_ids, membership = inspection_candidates(SOURCE_ROOT)
    case_ids = args.case_id or membership.ok_case_ids
    unknown_or_excluded = sorted(set(case_ids) - set(membership.ok_case_ids))
    if unknown_or_excluded:
        raise ValueError(f"Requested cases are not inspection-OK: {unknown_or_excluded}")
    results: list[CaseResult] = []
    details: list[dict] = []
    if args.workers == 1:
        completed = (
            (case_id, extract_case(case_id, output_root, args.validate_only, args.resume))
            for case_id in case_ids
        )
        for index, (case_id, (result, detail)) in enumerate(completed, start=1):
            results.append(result)
            if detail and result.status in {"ok", "validated"}:
                details.append(detail)
            print(f"[{index}/{len(case_ids)}] {case_id}: {result.status} {result.message}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(extract_case, case_id, output_root, args.validate_only, args.resume): case_id
                for case_id in case_ids
            }
            for index, future in enumerate(as_completed(futures), start=1):
                case_id = futures[future]
                result, detail = future.result()
                results.append(result)
                if detail and result.status in {"ok", "validated"}:
                    details.append(detail)
                print(f"[{index}/{len(case_ids)}] {case_id}: {result.status} {result.message}")
    if args.validate_only:
        failed = [result for result in results if result.status not in {"validated", "ok"}]
        print(json.dumps({"cases": len(results), "failed": len(failed), "stats": _aggregate_stats(details)}, indent=2))
        raise SystemExit(1 if failed else 0)
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "status", "output_file", "message"])
        writer.writeheader()
        writer.writerows(case_result_dict(result) for result in results)
    successful_ids = [result.case_id for result in results if result.status == "ok"]
    invariants = verify_final_artifacts(output_root, case_ids, results)
    inspection_exclusions = [
        {"case_id": case_id, "reason": membership.exclusion_reasons[case_id]}
        for case_id in membership.nok_case_ids
    ]
    payload = {
        "source_time": SOURCE_TIME,
        "grid_shape": list(GRID_SHAPE),
        "stencil_radius_pixels": STENCIL_RADIUS,
        "archived_source_cases": len(archived_ids),
        "inspection_ok_cases": len(membership.ok_case_ids),
        "inspection_nok_cases": len(membership.nok_case_ids),
        "inspection_exclusions": inspection_exclusions,
        "inspection_artifact": str(membership.csv_path),
        "inspection_artifact_sha256": membership.artifact_sha256,
        "inspection_membership_fingerprint": membership.membership_fingerprint,
        "requested_cases": len(case_ids),
        "successful_cases": len(successful_ids),
        "pinn_rejected_cases": len(case_ids) - len(successful_ids),
        "invalid_or_failed_cases": [case_result_dict(result) for result in results if result.status != "ok"],
        "post_rebuild_invariants": invariants,
        "aggregate_statistics": _aggregate_stats(details),
        "units": FIELD_UNITS,
    }
    (output_root / "validation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if any(result.status == "error" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
