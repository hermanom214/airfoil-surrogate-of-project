"""Validate and compare completed controlled Phase-A A/B outputs."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.unet_pinn_ab import compare_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-dir", type=Path, default=Path("data/models/unet_pinn_supervised"))
    parser.add_argument("--physics-dir", type=Path, default=Path("data/models/unet_pinn_physics"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    def load(root, name):
        path = root / name
        if not path.exists(): raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))
    left_meta, right_meta = load(args.supervised_dir, "experiment_metadata.json"), load(args.physics_dir, "experiment_metadata.json")
    problems = compare_metadata(left_meta, right_meta)
    report = {"controlled_comparison": not problems, "compatibility_problems": problems}
    if not problems:
        lm, rm = load(args.supervised_dir, "validation_metrics.json"), load(args.physics_dir, "validation_metrics.json")
        report["metrics"] = {"supervised": lm, "physics_informed": rm,
            "delta_physics_minus_supervised": {key: rm[key] - lm[key] for key in (
                "supervised_loss", "loss_p", "loss_ux", "loss_uy", "raw_continuity_loss",
                "raw_momentum_x_loss", "raw_momentum_y_loss")}}
        report["physics_rms"] = {label: {
            "continuity_rms": math.sqrt(metrics["raw_continuity_loss"]),
            "momentum_x_rms": math.sqrt(metrics["raw_momentum_x_loss"]),
            "momentum_y_rms": math.sqrt(metrics["raw_momentum_y_loss"]),
        } for label, metrics in (("supervised", lm), ("physics_informed", rm))}
        left_history = load(args.supervised_dir, "training_history.json")["epochs"]
        right_history = load(args.physics_dir, "training_history.json")["epochs"]
        report["convergence"] = {"supervised": left_history, "physics_informed": right_history}
        report["training"] = {label: {key: meta.get(key) for key in (
            "best_validation_epoch", "training_duration_seconds", "peak_cuda_memory_bytes")}
            for label, meta in (("supervised", left_meta), ("physics_informed", right_meta))}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if problems:
        raise SystemExit(2)


if __name__ == "__main__": main()
