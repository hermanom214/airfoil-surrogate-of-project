from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.train_unet_pinn_ab import resolve_experiment
from src.config import load_ml_models_config
from src.ml_data_split import build_holdout_fold
from src.ml_models import build_model
from src.unet_pinn import unet_pinn_losses
from src.unet_pinn_ab import AB_EXPERIMENTS, compare_metadata, configure_determinism, membership_sha256, state_dict_sha256


CFG = Path("configs/ml_models_config.yaml")


def profiles():
    cfg = load_ml_models_config(CFG)
    return cfg, [resolve_experiment(cfg, name, False, None) for name in AB_EXPERIMENTS]


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
    assert "Refusing to overwrite experiment output" in source
