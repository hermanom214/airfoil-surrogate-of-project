"""Build the complete accepted 480x272 body-frame TE/wake dataset.

The authoritative SimpleUNet fixed split defines membership.  Generation is
transactional at directory level: an existing final or staging directory is a
hard error, and the staging directory is renamed only after full validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.highres_wake_patch_feasibility import (
    NX, NY, XMAX_C, XMIN_C, YMAX_C, YMIN_C, accepted_case_ids,
    lab_to_body, make_interpolator,
)
from src.flow_extractor import create_fluid_mask, find_vtk_file, read_vtk_with_pyvista


SOURCE_ROOT = ROOT / "data" / "flow_fields"
OUTPUT_ROOT = ROOT / "data" / "flow_fields_highres_wake"
STAGING_ROOT = ROOT / "data" / "flow_fields_highres_wake.building"
SPLIT_PATH = ROOT / "data" / "models" / "simple_unet" / "fixed_test_split.json"


def _export_case(case_id: str) -> dict[str, object]:
    case_dir = SOURCE_ROOT / case_id
    params = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
    chord = float(params["chord"])
    aoa = float(params["aoa_deg"])
    naca = str(params["naca_code"])
    vtk_file = find_vtk_file(case_dir)
    points, pressure, velocity = read_vtk_with_pyvista(vtk_file)
    xb, yb = lab_to_body(points[:, 0], points[:, 1], aoa, chord)
    source_xy_c = np.c_[xb / chord, yb / chord]

    xs_c = np.linspace(XMIN_C, XMAX_C, NX)
    ys_c = np.linspace(YMIN_C, YMAX_C, NY)
    gx_c, gy_c = np.meshgrid(xs_c, ys_c)
    queries = np.c_[gx_c.ravel(), gy_c.ravel()]
    sampled = make_interpolator(source_xy_c, np.c_[pressure, velocity[:, :2]])(queries).reshape(NY, NX, 3)
    xy = (np.stack([gx_c, gy_c], axis=-1) * chord).astype(np.float32)
    p = sampled[..., 0].astype(np.float32)
    u = sampled[..., 1:].astype(np.float32)
    fluid_mask = create_fluid_mask(xy, naca, chord, 0.0).astype(np.float32)
    dx_c = (XMAX_C - XMIN_C) / (NX - 1)
    dy_c = (YMAX_C - YMIN_C) / (NY - 1)
    metadata = {
        **params,
        "coordinate_frame": "body",
        "velocity_component_frame": "laboratory/global",
        "patch_bounds_c": [XMIN_C, XMAX_C, YMIN_C, YMAX_C],
        "nx": NX, "ny": NY, "dx_c": dx_c, "dy_c": dy_c,
        "source_vtk": str(vtk_file),
        "interpolation": "linear scipy Delaunay; nearest only outside convex hull",
        "p_semantics": "OpenFOAM kinematic pressure; outlet p=0 reference",
    }
    case_output = STAGING_ROOT / case_id
    case_output.mkdir()
    output_file = case_output / f"{case_id}.npz"
    np.savez_compressed(
        output_file, xy=xy, p=p, U=u, fluid_mask=fluid_mask, case_id=case_id,
        naca_code=naca, aoa_deg=aoa, U_inf=float(params["inlet_velocity"]), chord=chord,
        patch_bounds_c=np.array(metadata["patch_bounds_c"]), dx_c=dx_c, dy_c=dy_c,
        source_vtk=str(vtk_file), params=json.dumps(params), metadata=json.dumps(metadata),
    )
    return {"case_id": case_id, "status": "ok", "file_bytes": output_file.stat().st_size,
            "fluid_fraction": float(np.mean(fluid_mask > 0.5))}


def _validate(ids: list[str]) -> None:
    found = sorted(p.parent.name for p in STAGING_ROOT.glob("*/*.npz"))
    if found != ids:
        raise RuntimeError(f"Dataset membership mismatch: missing={sorted(set(ids)-set(found))}, extra={sorted(set(found)-set(ids))}")
    expected_bounds = np.array([XMIN_C, XMAX_C, YMIN_C, YMAX_C])
    for case_id in ids:
        with np.load(STAGING_ROOT / case_id / f"{case_id}.npz", allow_pickle=False) as z:
            if z["xy"].shape != (NY, NX, 2) or z["p"].shape != (NY, NX) or z["U"].shape != (NY, NX, 2):
                raise RuntimeError(f"Shape mismatch: {case_id}")
            if not np.array_equal(z["patch_bounds_c"], expected_bounds):
                raise RuntimeError(f"Bounds mismatch: {case_id}")
            if not np.isclose(float(z["dx_c"]), (XMAX_C-XMIN_C)/(NX-1), rtol=0, atol=1e-12):
                raise RuntimeError(f"dx mismatch: {case_id}")
            if not np.isclose(float(z["dy_c"]), (YMAX_C-YMIN_C)/(NY-1), rtol=0, atol=1e-12):
                raise RuntimeError(f"dy mismatch: {case_id}")
            if not np.isfinite(z["p"]).all() or not np.isfinite(z["U"]).all():
                raise RuntimeError(f"Non-finite field: {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() or STAGING_ROOT.exists():
        raise FileExistsError(f"Refusing output collision: final={OUTPUT_ROOT.exists()}, staging={STAGING_ROOT.exists()}")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    development = list(split["development_case_ids"])
    test = list(split["test_case_ids"])
    if len(development) != 465 or len(test) != 116 or set(development) & set(test):
        raise RuntimeError("Authoritative split must contain disjoint 465/116 populations")
    ids = sorted(accepted_case_ids())
    if ids != sorted(development + test):
        raise RuntimeError("Accepted membership differs from the persisted split")
    STAGING_ROOT.mkdir()
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_export_case, case_id): case_id for case_id in ids}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                rows.append(result)
                print(f"[{i:03d}/{len(ids)}] {result['case_id']} {int(result['file_bytes'])/2**20:.3f} MiB", flush=True)
        _validate(ids)
        rows.sort(key=lambda row: str(row["case_id"]))
        with (STAGING_ROOT / "manifest.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary = {
            "case_count": len(ids), "development_count": len(development), "test_count": len(test),
            "split_sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
            "shape_ny_nx": [NY, NX], "bounds_c": [XMIN_C, XMAX_C, YMIN_C, YMAX_C],
            "dx_c": (XMAX_C-XMIN_C)/(NX-1), "dy_c": (YMAX_C-YMIN_C)/(NY-1),
            "total_bytes": sum(int(row["file_bytes"]) for row in rows),
            "runtime_seconds": time.perf_counter()-started,
        }
        (STAGING_ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        STAGING_ROOT.rename(OUTPUT_ROOT)
        print(json.dumps(summary, indent=2))
    except BaseException:
        print(f"Generation failed; incomplete staging data retained at {STAGING_ROOT}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
