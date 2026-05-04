## U-Net Forward Pass Overview

| Step | Operation | Shape Change | Description |
|------|----------|-------------|-------------|
| Input | x | (B, 5, H, W) | Input tensor (e.g. geometry + conditions) |
| e1 | enc1(x) | (B, 5, H, W) → (B, 32, H, W) | Convolution block (Conv + ReLU), increases channels, keeps resolution |
| e2 | enc2(pool(e1)) | (B, 32, H, W) → (B, 64, H/2, W/2) | Downsampling (MaxPool) → lower resolution, then Conv block |
| e3 | enc3(pool(e2)) | (B, 64, H/2, W/2) → (B, 128, H/4, W/4) | Downsampling (MaxPool) → lower resolution, then Conv block |
| b | bottleneck(pool(e3)) | (B, 128, H/4, W/4) → (B, 256, H/8, W/8) | Downsampling (MaxPool) → lower resolution, then Conv block |
| d3 up | up3(b) | (B, 256, H/8, W/8) → (B, 128, H/4, W/4) | Upsampling (ConvTranspose2d) → higher resolution |
| d3 cat | cat(d3, e3) | (B, 128, H/4, W/4) → (B, 256, H/4, W/4) | Concatenation along channel dimension (skip connection) |
| d3 | dec3(d3) | (B, 256, H/4, W/4) → (B, 128, H/4, W/4) | Convolution block (Conv + ReLU), feature processing |
| d2 up | up2(d3) | (B, 128, H/4, W/4) → (B, 64, H/2, W/2) | Upsampling (ConvTranspose2d) → higher resolution |
| d2 cat | cat(d2, e2) | (B, 64, H/2, W/2) → (B, 128, H/2, W/2) | Concatenation along channel dimension (skip connection) |
| d2 | dec2(d2) | (B, 128, H/2, W/2) → (B, 64, H/2, W/2) | Convolution block (Conv + ReLU), feature processing |
| d1 up | up1(d2) | (B, 64, H/2, W/2) → (B, 32, H, W) | Upsampling (ConvTranspose2d) → higher resolution |
| d1 cat | cat(d1, e1) | (B, 32, H, W) → (B, 64, H, W) | Concatenation along channel dimension (skip connection) |
| d1 | dec1(d1) | (B, 64, H, W) → (B, 32, H, W) | Convolution block (Conv + ReLU), feature processing |
| Output | out(d1) | (B, 32, H, W) → (B, 3, H, W) | 1×1 convolution, maps features to output channels |

---

## Summary

- **Downsampling (MaxPool2d)**: reduces spatial resolution (H, W ↓)  
- **Upsampling (ConvTranspose2d)**: increases spatial resolution (H, W ↑)  
- **Conv block**: applies convolution + nonlinearity to transform features  
- **Concatenation (skip connection)**: merges encoder and decoder features along channels  