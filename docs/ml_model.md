# ML Model and Training

## Active training script

- [scripts/train_ml_model.py](../scripts/train_ml_model.py)

## Supported model families

- `simple_unet`
- `rans_pinn`

Defined in [src/ml_models.py](../src/ml_models.py).

## Config-driven ML setup

All ML-related hardcoded parameters were moved to:
- [configs/ml_models_config.yaml](../configs/ml_models_config.yaml)

This includes:
- selected model name
- model architecture parameters
- training hyperparameters
- physics-loss parameters
- output naming/path template

Loader and dataclasses:
- [src/config.py](../src/config.py) (`load_ml_models_config`)

## Dataset interface

- dataset class: [src/ml_dataset.py](../src/ml_dataset.py)
- expected sample format: `*_flow.npz`
- target channels: pressure + velocity components (`p`, `Ux`, `Uy`)

## Training loop components

- masked data loss and evaluation: [src/ml_training.py](../src/ml_training.py)
- physics-informed residual loss: in model implementation ([src/ml_models.py](../src/ml_models.py))
- loss curve plotting: [src/ml_validation.py](../src/ml_validation.py)

## Typical execution

```bash
python -m scripts.train_ml_model
```

Model weights and plots are written according to `output.*` in [configs/ml_models_config.yaml](../configs/ml_models_config.yaml).