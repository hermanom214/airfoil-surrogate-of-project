from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.evaluate_clcd_plotting import (
    SUMMARY_FIGURE_DPI,
    SUMMARY_FIGURE_SIZE,
    evaluate_generate_clcd_plots,
)
from src.evaluate_metrics import (
    evaluate_finalize_global_aggregator,
    evaluate_init_global_aggregator,
    evaluate_update_global_aggregator,
)
from src.ml_dataset import AirfoilFlowDataset


def test_clcd_summary_uses_evaluated_values_metrics_and_separate_scales(
    tmp_path, monkeypatch
) -> None:
    metrics_df = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "Cl_true": [-0.5, 0.25, 1.0],
            "Cl_pred": [-0.48, 0.27, 0.96],
            "Cl_abs_error": [0.02, 0.02, 0.04],
            "Cd_true": [0.010, 0.020, 0.030],
            "Cd_pred": [0.012, 0.018, 0.035],
            "Cd_abs_error": [0.002, 0.002, 0.005],
        }
    )
    global_metrics = {
        "Cl": {"R2": 0.987654321, "MAE": 0.026666666},
        "Cd": {"R2": 0.75, "MAE": 0.003},
    }
    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def capture_summary(self, fname, *args, **kwargs):
        if str(fname).endswith("clcd_summary_scatter.png"):
            captured["figure"] = self
            captured["dpi"] = kwargs.get("dpi")
        return original_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_summary)
    evaluate_generate_clcd_plots(metrics_df, tmp_path, global_metrics)

    assert (tmp_path / "clcd_summary_scatter.png").is_file()
    assert captured["dpi"] == SUMMARY_FIGURE_DPI
    figure = captured["figure"]
    assert tuple(figure.get_size_inches()) == SUMMARY_FIGURE_SIZE
    assert figure._suptitle is None

    cl_axis, cd_axis = figure.axes
    np.testing.assert_allclose(
        cl_axis.collections[0].get_offsets(), metrics_df[["Cl_true", "Cl_pred"]]
    )
    np.testing.assert_allclose(
        cd_axis.collections[0].get_offsets(), metrics_df[["Cd_true", "Cd_pred"]]
    )
    assert cl_axis.get_xlim() == cl_axis.get_ylim()
    assert cd_axis.get_xlim() == cd_axis.get_ylim()
    assert cl_axis.get_xlim() != cd_axis.get_xlim()
    assert "R² = 0.987654" in cl_axis.texts[0].get_text()
    assert "MAE = 0.003" in cd_axis.texts[0].get_text()


def test_global_aggregator_uses_intersection_of_valid_cells() -> None:
    fluid_mask = np.ones((2, 2), dtype=bool)

    p_pred = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float64)
    p_true = np.zeros((2, 2), dtype=np.float64)

    ux_pred = np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float64)
    ux_true = np.zeros((2, 2), dtype=np.float64)

    uy_pred = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)
    uy_true = np.zeros((2, 2), dtype=np.float64)

    speed_pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    speed_true = np.zeros((2, 2), dtype=np.float64)

    velocity_vector_error = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    agg = evaluate_init_global_aggregator()
    evaluate_update_global_aggregator(
        agg=agg,
        p_pred=p_pred,
        p_true=p_true,
        ux_pred=ux_pred,
        ux_true=ux_true,
        uy_pred=uy_pred,
        uy_true=uy_true,
        speed_pred=speed_pred,
        speed_true=speed_true,
        velocity_vector_error=velocity_vector_error,
        fluid_mask=fluid_mask,
    )

    # Valid intersection contains only cells (0,1) and (1,0).
    assert int(agg["fluid_cells"]) == 2
    assert np.isclose(float(agg["p_abs_sum"]), 5.0)
    assert np.isclose(float(agg["ux_abs_sum"]), 50.0)
    assert np.isclose(float(agg["uy_abs_sum"]), 500.0)
    assert np.isclose(float(agg["speed_abs_sum"]), 5.0)
    assert np.isclose(float(agg["vec_abs_sum"]), 0.5)

    finalized = evaluate_finalize_global_aggregator(agg)
    assert finalized["global_fluid_cell_count"] == 2
    assert np.isclose(float(finalized["global_pixel_weighted_p_mae"]), 2.5)


def test_dataset_returns_float32_tensors(tmp_path) -> None:
    h, w = 4, 5
    x = np.linspace(-0.5, 1.0, w, dtype=np.float32)
    y = np.linspace(-0.2, 0.2, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xy = np.stack([xx, yy], axis=-1).astype(np.float32)

    p = np.ones((h, w), dtype=np.float32)
    u = np.zeros((h, w, 2), dtype=np.float32)
    fluid_mask = np.ones((h, w), dtype=np.float32)

    case_dir = tmp_path / "case_0001_naca0012_aoa0p0_u15p0"
    case_dir.mkdir()
    sample_path = case_dir / "case_0001_naca0012_aoa0p0_u15p0_flow.npz"
    np.savez_compressed(sample_path, xy=xy, p=p, U=u, fluid_mask=fluid_mask)
    inspection_path = tmp_path.parent / "pictures_inspect_flow" / "inspect_plausibility.csv"
    inspection_path.parent.mkdir(exist_ok=True)
    with inspection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_name", "overall_status", "overall_reason"]
        )
        writer.writeheader()
        writer.writerow({
            "case_name": case_dir.name, "overall_status": "OK", "overall_reason": ""
        })

    dataset = AirfoilFlowDataset(tmp_path)
    inp, target, mask = dataset[0]

    assert inp.dtype == torch.float32
    assert target.dtype == torch.float32
    assert mask.dtype == torch.float32
