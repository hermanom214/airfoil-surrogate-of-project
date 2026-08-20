from __future__ import annotations

import numpy as np
import torch

from src.evaluate_metrics import (
    evaluate_finalize_global_aggregator,
    evaluate_init_global_aggregator,
    evaluate_update_global_aggregator,
)
from src.ml_dataset import AirfoilFlowDataset


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

    dataset = AirfoilFlowDataset(tmp_path)
    inp, target, mask = dataset[0]

    assert inp.dtype == torch.float32
    assert target.dtype == torch.float32
    assert mask.dtype == torch.float32
