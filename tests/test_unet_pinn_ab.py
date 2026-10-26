from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.train_unet_pinn_ab import resolve_experiment
from src.config import load_ml_models_config
from src.ml_data_split import build_holdout_fold
from src.ml_models import build_model
from src.ml_supervised import SupervisedLossConfig
from src.ml_training import supervised_loss_components
from src.unet_pinn import unet_pinn_losses
from src.unet_pinn_ab import (AB_EXPERIMENTS, IMPROVED_AB_EXPERIMENTS, compare_metadata, configure_determinism,
    cuda_peak_memory, membership_sha256, state_dict_sha256, validate_finite_gradients)


CFG = Path("configs/ml_models_config.yaml")


def profiles():
    cfg = load_ml_models_config(CFG)
    return cfg, [resolve_experiment(cfg, name, False, None) for name in AB_EXPERIMENTS]


def improved_profiles():
    cfg = load_ml_models_config(CFG)
    return cfg, [resolve_experiment(cfg, name, False, None) for name in IMPROVED_AB_EXPERIMENTS]


def test_improved_profiles_differ_only_in_physics_weights_and_output_identity():
    _, (left, right) = improved_profiles()
    ignored = {"experiment_name", "output_name", "loss_weights"}
    assert {k: v for k, v in left.items() if k not in ignored} == {
        k: v for k, v in right.items() if k not in ignored
    }
    assert left["selection_metric"] == "improved_supervised_loss"
    assert left["supervised_loss_config"] == right["supervised_loss_config"]
    assert left["loss_weights"]["continuity_weight"] == 0.0
    assert right["loss_weights"]["continuity_weight"] == 0.1
    assert left["output_name"] == "unet_improved_supervised"
    assert right["output_name"] == "unet_improved_pinn"


def test_improved_pinn_loss_reuses_original_supervised_implementation_exactly():
    config = SupervisedLossConfig(target_representation="nondimensional_perturbation",
        wall_weight=3., wake_weight=3., gradient_loss_weight=.1)
    pred = torch.randn(2, 3, 9, 11, requires_grad=True)
    target = torch.randn_like(pred)
    fluid = torch.ones(2, 1, 9, 11)
    fluid[:, :, 3:5, 4:7] = 0
    regions = torch.zeros(2, 3, 9, 11)
    regions[:, 0, 2:7, 2:9] = fluid[:, 0, 2:7, 2:9]
    regions[:, 1, 4:6, 7:10] = fluid[:, 0, 4:6, 7:10]
    regions[:, 2] = fluid[:, 0] * (~regions[:, 0].bool() & ~regions[:, 1].bool())
    x, y = torch.meshgrid(torch.linspace(0, 1, 11), torch.linspace(0, 1, 9), indexing="xy")
    xy = torch.stack((x, y), -1)[None].repeat(2, 1, 1, 1)
    original = supervised_loss_components(pred, target, fluid, regions, config, .1, .125)
    combined = unet_pinn_losses(pred, target, fluid, fluid, xy, fluid, fluid*.01,
        torch.full((2,), 1e-5), torch.full((2,), 10.), torch.ones(2),
        (0.,1.,0.,1.,0.,1.), 1., 0., 0., compute_physics=False,
        supervised_config=config, regions=regions)
    torch.testing.assert_close(combined["supervised_loss"], original["total_loss"])
    torch.testing.assert_close(combined["supervised_weighted_data_loss"], original["data_loss"])
    torch.testing.assert_close(combined["supervised_gradient_loss"], original["gradient_loss"])
    combined["loss_total"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_improved_initial_state_and_training_settings_are_identical():
    _, resolved = improved_profiles()
    assert resolved[0]["model_config"] == resolved[1]["model_config"]
    assert resolved[0]["training_config"] == resolved[1]["training_config"]
    hashes = []
    for item in resolved:
        configure_determinism(item["training_config"]["seed"])
        hashes.append(state_dict_sha256(build_model("unet_pinn", **item["model_config"]).state_dict()))
    assert hashes[0] == hashes[1]


def test_ab_profiles_differ_only_in_physics_weights_and_output_identity():
    _, (left, right) = profiles()
    ignored = {"experiment_name", "output_name", "loss_weights"}
    assert {k: v for k, v in left.items() if k not in ignored} == {k: v for k, v in right.items() if k not in ignored}
    assert left["loss_weights"] == {"supervised_weight": 1.0, "continuity_weight": 0.0, "momentum_weight": 0.0}
    assert right["loss_weights"] == {"supervised_weight": 1.0, "continuity_weight": 0.1, "momentum_weight": 0.1}
    assert left["output_name"] != right["output_name"]


def test_identical_initial_state_is_verified_not_assumed():
    cfg, resolved = profiles()
    hashes = []
    for item in resolved:
        configure_determinism(int(item["training_config"]["seed"]))
        model = build_model("unet_pinn", **item["model_config"])
        hashes.append(state_dict_sha256(model.state_dict()))
    assert hashes[0] == hashes[1]


def test_holdout_membership_is_identical_and_disjoint():
    development = [0, 1, 2, 3, 4, 5, 6, 7]
    ids = [f"case-{i}" for i in range(10)]
    a = build_holdout_fold(development, ids, .25, 42)
    b = build_holdout_fold(development, ids, .25, 42)
    assert a.train_case_ids == b.train_case_ids and a.val_case_ids == b.val_case_ids
    assert set(a.train_case_ids).isdisjoint(a.val_case_ids)
    assert set(a.train_case_ids) | set(a.val_case_ids) == set(ids[:8])


def test_disabled_physics_has_no_training_gradient_and_raw_metrics_exist():
    pred = torch.randn(1, 3, 9, 11, requires_grad=True)
    target = torch.randn_like(pred)
    mask = torch.ones(1, 1, 9, 11)
    x, y = torch.meshgrid(torch.linspace(0, 1, 11), torch.linspace(0, 1, 9), indexing="xy")
    xy = torch.stack((x, y), -1)[None]
    values = unet_pinn_losses(pred, target, mask, mask, xy, mask, mask * .01,
        torch.tensor([1e-5]), torch.tensor([10.]), torch.tensor([1.]), (0., 1., 0., 1., 0., 1.),
        1., 0., 0., compute_physics=False)
    supervised_grad = torch.autograd.grad(values["supervised_loss"], pred, retain_graph=True)[0]
    total_grad = torch.autograd.grad(values["loss_total"], pred)[0]
    assert torch.equal(supervised_grad, total_grad)
    assert {"raw_continuity_loss", "raw_momentum_x_loss", "raw_momentum_y_loss", "raw_momentum_loss"} <= values.keys()


def test_comparison_rejects_incompatible_split_or_dataset():
    base = {"dataset_fingerprint": "d", "fixed_test_fingerprint": "t", "train_fingerprint": "r",
        "validation_fingerprint": "v", "initial_state_hash": "i", "model_config": {"x": 1},
        "training_config": {"epochs": 2}, "selection_metric": "supervised_loss",
        "loss_weights": {"supervised_weight": 1.0}}
    assert compare_metadata(base, copy.deepcopy(base)) == []
    changed = copy.deepcopy(base); changed["dataset_fingerprint"] = "other"
    assert "dataset_fingerprint differs" in compare_metadata(base, changed)


def test_checkpoint_output_collision_guard_is_explicit_in_runner_source():
    source = Path("scripts/train_unet_pinn_ab.py").read_text(encoding="utf-8")
    assert "if output.exists()" in source
    assert "full_run_with_only_preserved_pilot" in source
    assert "Refusing to overwrite experiment output" in source
    assert "output.mkdir(parents=True, exist_ok=True)" in source


def test_finite_gradients_allow_step_and_grad_none_is_ignored():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Linear(1, 1))
    model[1].requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    before = model[0].weight.detach().clone()
    assert validate_finite_gradients(model) == 2
    optimizer.step()
    assert not torch.equal(before, model[0].weight)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_gradient_stops_before_optimizer_step(bad):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.weight.grad = torch.tensor([[bad]])
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="weight"):
        validate_finite_gradients(model)
    assert torch.equal(before, model.weight)
    assert optimizer.state == {}


def test_nonfinite_total_loss_is_detected_before_backward():
    source = Path("scripts/train_unet_pinn_ab.py").read_text(encoding="utf-8")
    finite_check = source.index('if not torch.isfinite(losses["loss_total"])')
    backward = source.index('losses["loss_total"].backward()')
    assert finite_check < backward


def test_cpu_peak_memory_is_explicitly_unavailable():
    metrics = cuda_peak_memory(torch.device("cpu"))
    assert metrics == {
        "peak_cuda_memory_allocated_bytes": None,
        "peak_cuda_memory_allocated_mib": None,
        "peak_cuda_memory_reserved_bytes": None,
        "peak_cuda_memory_reserved_mib": None,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_peak_allocated_and_reserved_are_recorded():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    tensor = torch.ones(1024, device=device)
    metrics = cuda_peak_memory(device)
    assert tensor.is_cuda
    assert metrics["peak_cuda_memory_allocated_bytes"] >= tensor.numel() * tensor.element_size()
    assert metrics["peak_cuda_memory_reserved_bytes"] >= metrics["peak_cuda_memory_allocated_bytes"]
    assert metrics["peak_cuda_memory_allocated_mib"] is not None
    assert metrics["peak_cuda_memory_reserved_mib"] is not None


def test_comparator_preserves_existing_outputs_and_reports_new_memory_fields():
    source = Path("scripts/compare_unet_pinn_ab.py").read_text(encoding="utf-8")
    assert "peak_cuda_memory_allocated_mib" in source
    assert "peak_cuda_memory_reserved_mib" in source
    assert 'parser.add_argument("--supervised-dir"' in source
    assert 'parser.add_argument("--physics-dir"' in source
