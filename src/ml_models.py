from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


AVAILABLE_MODELS = ("simple_unet", "rans_pinn", "clcd_mlp")


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        encoder_channels: tuple[int, int, int] | list[int] = (32, 64, 128),
        bottleneck_channels: int = 256,
    ):
        super().__init__()

        if len(encoder_channels) != 3:
            raise ValueError("encoder_channels must contain exactly 3 values")

        c1, c2, c3 = int(encoder_channels[0]), int(encoder_channels[1]), int(encoder_channels[2])
        cb = int(bottleneck_channels)

        if min(c1, c2, c3, cb) <= 0:
            raise ValueError("All channel counts must be > 0")

        self.enc1 = self.block(in_channels, c1)
        self.enc2 = self.block(c1, c2)
        self.enc3 = self.block(c2, c3)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.block(c3, cb)

        self.up3 = nn.ConvTranspose2d(cb, c3, 2, stride=2)
        self.dec3 = self.block(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = self.block(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = self.block(c1 + c1, c1)

        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass (U-Net).
        # Model output channel order is [p, Ux, Uy], consistent with AirfoilFlowDataset.
#
# Input:
# x: (B, 5, H, W)
#
# Encoder (reduces resolution, increases channels – feature extraction)
# --------------------------------------------------------------------
# e1 = enc1(x)
# → (B, 5, H, W) → (B, 32, H, W)
#   2× Conv + ReLU, keeps resolution, increases channels
#
# e2 = enc2(pool(e1))
# → pool: (B, 32, H, W) → (B, 32, H/2, W/2)
# → enc2: (B, 32, H/2, W/2) → (B, 64, H/2, W/2)
#   downsampling + more feature channels
#
# e3 = enc3(pool(e2))
# → pool: (B, 64, H/2, W/2) → (B, 64, H/4, W/4)
# → enc3: (B, 64, H/4, W/4) → (B, 128, H/4, W/4)
#
# Bottleneck (smallest resolution, highest channel count – global context)
# -----------------------------------------------------------------------
# b = bottleneck(pool(e3))
# → pool: (B, 128, H/4, W/4) → (B, 128, H/8, W/8)
# → bottleneck: (B, 128, H/8, W/8) → (B, 256, H/8, W/8)
#
# Decoder (increases resolution, combines with encoder features)
# --------------------------------------------------------------
# d3 = up3(b)
# → (B, 256, H/8, W/8) → (B, 128, H/4, W/4)
#   upsampling (ConvTranspose)
#
# d3 = cat(d3, e3)
# → (B, 128, H/4, W/4) + (B, 128, H/4, W/4)
# → (B, 256, H/4, W/4)
#   skip connection (restores spatial detail)
#
# d3 = dec3(d3)
# → (B, 256, H/4, W/4) → (B, 128, H/4, W/4)
#
# d2 = up2(d3)
# → (B, 128, H/4, W/4) → (B, 64, H/2, W/2)
#
# d2 = cat(d2, e2)
# → (B, 64, H/2, W/2) + (B, 64, H/2, W/2)
# → (B, 128, H/2, W/2)
#
# d2 = dec2(d2)
# → (B, 128, H/2, W/2) → (B, 64, H/2, W/2)
#
# d1 = up1(d2)
# → (B, 64, H/2, W/2) → (B, 32, H, W)
#
# d1 = cat(d1, e1)
# → (B, 32, H, W) + (B, 32, H, W)
# → (B, 64, H, W)
#
# d1 = dec1(d1)
# → (B, 64, H, W) → (B, 32, H, W)
#
# Output
# ------
# out = self.out(d1)
# → (B, 32, H, W) → (B, 3, H, W)
#   1×1 convolution – maps feature maps to physical quantities in order [p, Ux, Uy]
#
# Summary:
# encoder → compresses spatial resolution, learns context
# bottleneck → global representation
# decoder → restores resolution + recovers details via skip connections

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class PhysicsInformedCNN(nn.Module):
    """
    Physics-informed CNN surrogate for 2D steady incompressible airfoil flow.

    Model output channel order is [p, Ux, Uy], consistent with AirfoilFlowDataset.
    Current version predicts [p, u, v] from per-cell input features and uses
    RANS residual terms (continuity + momentum equations) as a physics loss.

    Project note for SST k-omega turbulence:
    for our SST k-omega setup, this model must be extended with additional
    predicted fields k, omega, and nut, and with turbulence transport equations
    included in the physics residual formulation.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        hidden_channels: int = 128,
        depth: int = 6,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("depth must be >= 2 for PhysicsInformedCNN")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        ]

        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.Tanh(),
                ]
            )

        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _ddx(field: torch.Tensor, dx: float) -> torch.Tensor:
        padded = F.pad(field, (1, 1, 0, 0), mode="replicate")
        return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * dx)

    @staticmethod
    def _ddy(field: torch.Tensor, dy: float) -> torch.Tensor:
        padded = F.pad(field, (0, 0, 1, 1), mode="replicate")
        return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * dy)

    @staticmethod
    def _laplacian(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        ddx = PhysicsInformedCNN._ddx(PhysicsInformedCNN._ddx(field, dx), dx)
        ddy = PhysicsInformedCNN._ddy(PhysicsInformedCNN._ddy(field, dy), dy)
        return ddx + ddy

    @staticmethod
    def _erode_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
        if pixels <= 0:
            return (mask > 0.5).to(mask.dtype)

        binary_mask = (mask > 0.5).to(mask.dtype)
        inv_mask = 1.0 - binary_mask
        kernel = 2 * pixels + 1
        dilated_inv = F.max_pool2d(inv_mask, kernel_size=kernel, stride=1, padding=pixels)
        eroded = 1.0 - dilated_inv
        return eroded.clamp(0.0, 1.0)

    @staticmethod
    def _masked_mse(residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sq = residual.pow(2) * mask
        denom = mask.sum() + 1e-8
        return sq.sum() / denom

    def rans_residual_loss(
        self,
        pred: torch.Tensor,
        fluid_mask: torch.Tensor,
        dx: float,
        dy: float,
        nu: float,
        p_mean: float,
        p_std: float,
        ux_mean: float,
        ux_std: float,
        uy_mean: float,
        uy_std: float,
        nu_t: torch.Tensor | None = None,
        pressure_is_kinematic: bool = True,
        mask_erode_pixels: int = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Returns masked loss terms of steady incompressible 2D RANS residuals.
        Expects prediction channels in order [p_norm, ux_norm, uy_norm].
        """

        # Model output channel order is [p, Ux, Uy], consistent with AirfoilFlowDataset.
        p_norm = pred[:, 0:1, :, :]
        u_norm = pred[:, 1:2, :, :]
        v_norm = pred[:, 2:3, :, :]

        p = p_norm * p_std + p_mean
        u = u_norm * ux_std + ux_mean
        v = v_norm * uy_std + uy_mean

        mask = self._erode_mask(fluid_mask, mask_erode_pixels)
        if mask.shape[1] != 1:
            raise ValueError("fluid_mask must have shape [B, 1, H, W]")
        if mask.shape[0] != pred.shape[0] or mask.shape[2:] != pred.shape[2:]:
            raise ValueError("fluid_mask must match pred spatial dimensions and batch size")

        du_dx = self._ddx(u, dx)
        du_dy = self._ddy(u, dy)
        dv_dx = self._ddx(v, dx)
        dv_dy = self._ddy(v, dy)
        dp_dx = self._ddx(p, dx)
        dp_dy = self._ddy(p, dy)

        if nu_t is None:
            nu_eff = torch.full_like(u, fill_value=nu)
        else:
            nu_eff = nu + nu_t

        continuity = du_dx + dv_dy
        if pressure_is_kinematic:
            pressure_x = dp_dx
            pressure_y = dp_dy
        else:
            rho = 1.0
            pressure_x = (1.0 / rho) * dp_dx
            pressure_y = (1.0 / rho) * dp_dy

        mom_x = u * du_dx + v * du_dy + pressure_x - nu_eff * self._laplacian(u, dx, dy)
        mom_y = u * dv_dx + v * dv_dy + pressure_y - nu_eff * self._laplacian(v, dx, dy)

        continuity_loss = self._masked_mse(continuity, mask)
        mom_x_loss = self._masked_mse(mom_x, mask)
        mom_y_loss = self._masked_mse(mom_y, mask)
        total_loss = continuity_loss + mom_x_loss + mom_y_loss

        return {
            "physics_loss": total_loss,
            "continuity_loss": continuity_loss,
            "momentum_x_loss": mom_x_loss,
            "momentum_y_loss": mom_y_loss,
        }


class ClCdMLP(nn.Module):
    """MLP regressor for aerodynamic coefficients [Cl, Cd]."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        output_dim: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")
        if any(int(v) <= 0 for v in hidden_dims):
            raise ValueError("All hidden layer sizes must be > 0")
        if not (0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout must be in range [0, 1)")

        dims = [int(input_dim), *[int(v) for v in hidden_dims], int(output_dim)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(model_name: str, **kwargs) -> nn.Module:
    """Factory for selecting a model architecture from AVAILABLE_MODELS."""

    key = model_name.lower().strip()
    if key == "simple_unet":
        return SimpleUNet(**kwargs)
    if key == "rans_pinn":
        return PhysicsInformedCNN(**kwargs)
    if key == "clcd_mlp":
        return ClCdMLP(**kwargs)

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: {', '.join(AVAILABLE_MODELS)}"
    )