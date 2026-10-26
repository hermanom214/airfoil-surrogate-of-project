from __future__ import annotations

import numpy as np
import pytest

from src.unet_pinn_field_evaluation import (
    PRESSURE_UNIT, assert_matching_normalization, denormalize_prediction,
    masked_metrics, region_masks, select_representative_cases,
    shared_error_limit, shared_value_limits,
    decode_checkpoint_prediction, summarize_field_metrics, field_interpretation,
)


def norm():
    return dict(aoa_mean=0,aoa_std=1,inlet_u_mean=20,inlet_u_std=2,p_mean=10,p_std=2,ux_mean=20,ux_std=4,uy_mean=0,uy_std=.5)


def test_checkpoint_normalization_and_physical_conversion():
    assert assert_matching_normalization(norm(),dict(norm())) == norm()
    bad=dict(norm()); bad["p_mean"]=11
    with pytest.raises(ValueError,match="different"): assert_matching_normalization(norm(),bad)
    physical=denormalize_prediction(np.ones((3,2,2)),norm())
    assert np.all(physical["p"]==12) and np.all(physical["Ux"]==24) and np.all(physical["Uy"]==.5)
    assert "kinematic pressure" in PRESSURE_UNIT and "Pa" not in PRESSURE_UNIT


def test_metrics_respect_mask_and_aggregate_exactly():
    ref=np.zeros((2,2)); pred=np.array([[1.,100.],[3.,100.]])
    mask=np.array([[1,0],[1,0]],dtype=bool)
    mae,rmse=masked_metrics(pred,ref,mask)
    assert mae==2 and rmse==pytest.approx(np.sqrt(5))


def test_rois_are_physical_and_aoa_aligned():
    x=np.linspace(-.5,2,101); y=np.linspace(-1,1,81); xx,yy=np.meshgrid(x,y); xy=np.stack((xx,yy),-1)
    masks=region_masks(xy,np.ones(xx.shape,bool),1.,4.)
    assert masks["wake"].shape==xx.shape and masks["wake"].sum()>0 and masks["near_airfoil"].sum()>masks["wake"].sum()
    assert not np.any(masks["outer_flow"] & masks["near_airfoil"])


def test_shared_plot_limits_are_common_and_error_symmetric():
    ref=np.arange(100.).reshape(10,10); mask=np.ones_like(ref,dtype=bool)
    lo,hi=shared_value_limits(ref,mask); assert lo<hi
    e=shared_error_limit(np.ones_like(ref),-2*np.ones_like(ref),mask)
    assert e==2


def test_selection_is_deterministic_metadata_only_and_eligible():
    cases=[]
    for i in range(12):
        cases.append(dict(case_id=f"case_{i:02d}",camber_percent=i%3,thickness_percent=8+2*(i%5),aoa_deg=(-4,-2,0,2,4)[i%5],U_inf=(15,17.5,20,22.5,25)[(i*2)%5],inspection_ok=True,pinn_valid=True,error=100-i))
    a=select_representative_cases(cases,8); b=select_representative_cases(list(reversed(cases)),8)
    assert [x["case_id"] for x in a]==[x["case_id"] for x in b]
    assert len({x["case_id"] for x in a})==8
    assert {np.sign(x["aoa_deg"]) for x in a} == {-1,0,1}
    assert {x["camber_percent"] == 0 for x in a} == {False,True}


@pytest.mark.parametrize("speed", [8., 25.])
def test_improved_field_decoding_uses_checkpoint_and_case_speed(speed):
    from src.ml_supervised import SupervisedLossConfig, physical_to_targets
    config = SupervisedLossConfig(target_representation="nondimensional_perturbation", p_inf=13.)
    checkpoint = dict(norm(), supervised_loss_config=config.metadata())
    for field in ("p", "ux", "uy"):
        checkpoint[field+"_mean"], checkpoint[field+"_std"] = 0., 1.
    x, y = np.meshgrid(np.arange(4.), np.arange(3.))
    p, u, v = 20+x-y, 3+2*x+y, -4+x+3*y
    prediction = physical_to_targets(p,u,v,speed,config.target_representation,config.p_inf)
    decoded = decode_checkpoint_prediction(prediction, checkpoint, speed)
    for key, expected in zip(("p", "Ux", "Uy"), (p,u,v)):
        np.testing.assert_allclose(decoded[key], expected, atol=2e-6)
    raw = decode_checkpoint_prediction(np.ones((3,2,2)), norm(), speed)
    assert raw["p"][0,0] == 12 and raw["Ux"][0,0] == 24
    with pytest.raises(ValueError, match="target/preprocessing"):
        assert_matching_normalization(checkpoint, {**checkpoint, "supervised_loss_config": {}})


@pytest.mark.parametrize("count", [6,8])
@pytest.mark.parametrize("identical", [False,True])
def test_report_interpretation_uses_actual_metrics(count, identical):
    rows = []
    deltas = [0.] * count if identical else [-1., 0., 2.] * (count//3) + [-1.,2.][:count%3]
    for region in ("full_domain", "near_airfoil", "wake"):
        for delta in deltas:
            rows.append(dict(region=region, variable="p", supervised_MAE=3., physics_MAE=3.+delta,
                supervised_RMSE=4., physics_RMSE=4.-delta))
    summary = summarize_field_metrics(rows)
    for s in summary:
        actual = np.array(deltas) * (1 if s["metric"] == "MAE" else -1)
        assert s["case_count"] == count
        assert s["physics_improved_cases"] == (actual < 0).sum()
        assert s["physics_worsened_cases"] == (actual > 0).sum()
        assert s["ties"] == (actual == 0).sum()
        assert s["mean_B_minus_A"] == pytest.approx(actual.mean())
        assert s["median_B_minus_A"] == pytest.approx(np.median(actual))
    text = "\n".join(field_interpretation(summary))
    assert f"of {count} selected cases" in text
    assert "wake p MAE" in text
    if identical:
        assert f"tied in {count}" in text


def test_field_comparison_selects_improved_pair_and_isolates_output():
    from scripts.compare_unet_pinn_fields import parse_args, DEFAULT_OUTPUT
    args = parse_args(["--improved", "--pilot"])
    assert args.improved and args.pilot
    assert args.output_dir != DEFAULT_OUTPUT
    assert args.output_dir.parent.name == "unet_improved_pinn_fields"

