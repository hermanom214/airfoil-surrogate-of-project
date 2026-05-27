# ML Model and Training

## Active training script

- [scripts/train_ml_model.py](../scripts/train_ml_model.py)

## Supported model families

- `simple_unet`
- `rans_pinn`

Defined in [src/ml_models.py](../src/ml_models.py).

## Architecture snapshot

Both models use the same dataset interface:
- input tensor shape: `(B, 5, H, W)`
- output tensor shape: `(B, 3, H, W)`
- output channels represent `(p, Ux, Uy)` in normalized form.

### 1) `simple_unet` (encoder-decoder with skip connections)

Implementation: [src/ml_models.py](../src/ml_models.py), class `SimpleUNet`

Main structure:
- encoder: three convolutional stages (`enc1`, `enc2`, `enc3`) with `Conv(3x3)+ReLU` blocks
- downsampling: `MaxPool2d(2)` between encoder stages
- bottleneck: deeper convolutional block at lowest spatial resolution
- decoder: transposed-convolution upsampling (`up3`, `up2`, `up1`)
- skip connections: concatenation with encoder features at matching scale
- head: final `1x1` convolution to 3 output channels

Configurable parameters (from [configs/ml_models_config.yaml](../configs/ml_models_config.yaml)):
- `in_channels`, `out_channels`
- `encoder_channels` (default `[32, 64, 128]`)
- `bottleneck_channels` (default `256`)

Use case:
- strong baseline for dense field regression when data fidelity is primary.

### 2) `rans_pinn` (physics-informed convolutional network)

Implementation: [src/ml_models.py](../src/ml_models.py), class `PhysicsInformedCNN`

Main structure:
- plain fully-convolutional stack (no pooling, no skip paths)
- first layer: `Conv(3x3)` from input channels to `hidden_channels`
- hidden trunk: repeated `Conv(3x3)+Tanh` blocks (`depth` controls count)
- output layer: final `Conv(3x3)` to 3 channels

Physics-informed component:
- model provides `rans_residual_loss(...)` for continuity + momentum residuals
- spatial derivatives are computed with finite-difference-like operators (`_ddx`, `_ddy`, `_laplacian`)
- residual loss is masked to fluid cells and mixed with data loss during training

Configurable parameters:
- `in_channels`, `out_channels`
- `hidden_channels` (default `128`)
- `depth` (default `6`)

Use case:
- preferred when enforcing PDE consistency together with supervised data fit.

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