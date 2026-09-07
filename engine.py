#!/usr/bin/env python3
"""
Classical DIP & Deep Learning Engine (engine.py)
Provides surface map estimation algorithms, PyTorch U-Net deep learning model trained with Cosine Angular Loss,
2D DFT roughness estimation, 2D Fourier Poisson height integration, and quantitative evaluation metrics.
"""

import os
import io
import base64
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from skimage.metrics import structural_similarity as ssim_func

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "unet_normal.pth"


# =====================================================================
# 1. Classical DIP Estimators & Preprocessing
# =====================================================================

def preprocess_image(rgb_image: np.ndarray, gamma: float = 2.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess RGB image:
    1. Apply gamma correction (sRGB -> Linear space).
    2. Convert to Grayscale using standard Luma weights.
    3. Apply 1st and 99th percentile contrast stretching.
    Returns: (preprocessed_rgb [0,1], preprocessed_gray [0,1])
    """
    if rgb_image.dtype == np.uint8:
        img_float = rgb_image.astype(np.float32) / 255.0
    else:
        img_float = rgb_image.copy()

    # Gamma correction
    linear_img = np.power(np.clip(img_float, 0.0, 1.0), gamma)

    # Grayscale conversion (Luma weights)
    if len(linear_img.shape) == 3 and linear_img.shape[2] == 3:
        gray = 0.2989 * linear_img[:, :, 0] + 0.5870 * linear_img[:, :, 1] + 0.1140 * linear_img[:, :, 2]
    else:
        gray = linear_img.copy()

    # Percentile-based contrast stretching
    p1, p99 = np.percentile(gray, (1, 99))
    if p99 > p1:
        gray_stretched = np.clip((gray - p1) / (p99 - p1), 0.0, 1.0)
    else:
        gray_stretched = gray

    return linear_img, gray_stretched


def estimate_normal_map_dip(gray_img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Classical DIP Surface Normal Estimation:
    1. Sobel gradient extraction: Ix = dI/dx, Iy = dI/dy
    2. Normalized to unit vector N = (-Ix * scale, -Iy * scale, 1.0) / ||N||
    3. Remapped to OpenGL RGB [0, 255].
    """
    Ix = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    Nx = -Ix * scale
    Ny = -Iy * scale
    Nz = np.ones_like(gray_img, dtype=np.float64)

    magnitude = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    magnitude = np.maximum(magnitude, 1e-8)

    Nx_unit = Nx / magnitude
    Ny_unit = Ny / magnitude
    Nz_unit = Nz / magnitude

    R = ((Nx_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    G = ((Ny_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    B = ((Nz_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)


def estimate_roughness_map_dft(gray_img: np.ndarray, cutoff_radius: float = 30.0) -> np.ndarray:
    """
    Classical Roughness Estimation:
    1. 2D Discrete Fourier Transform (DFT) & shift DC component to center.
    2. 2D Gaussian High-Pass Frequency Filter Mask: H_hp(u,v) = 1 - exp(-D^2 / (2 * D0^2))
    3. Inverse DFT to isolate micro-texture spatial components.
    4. Local variance / standard deviation calculation in sliding window.
    """
    H, W = gray_img.shape

    # 1. 2D DFT
    F = fftshift(fft2(gray_img))

    # 2. Gaussian High-Pass Frequency Filter Mask
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D_sq = U**2 + V**2

    D0_sq = max(cutoff_radius**2, 1.0)
    H_hp = 1.0 - np.exp(-D_sq / (2.0 * D0_sq))

    # 3. Filter in frequency domain & Inverse DFT
    F_filtered = F * H_hp
    high_freq_spatial = np.abs(ifft2(ifftshift(F_filtered)))

    # 4. Local variance in sliding window
    local_mean = cv2.blur(high_freq_spatial, (9, 9))
    local_sq_mean = cv2.blur(high_freq_spatial**2, (9, 9))
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    local_std = np.sqrt(local_var)

    # 5. Composite high-frequency magnitude & local stddev
    roughness = high_freq_spatial * 0.5 + local_std * 0.5

    p5, p95 = np.percentile(roughness, (5, 95))
    if p95 > p5:
        roughness_norm = np.clip((roughness - p5) / (p95 - p5), 0.0, 1.0)
    else:
        roughness_norm = np.clip(roughness, 0.0, 1.0)

    return (roughness_norm * 255.0).astype(np.uint8)


def estimate_height_map(normal_map_rgb: np.ndarray) -> np.ndarray:
    """
    Classical Height Estimation:
    2D Fourier Poisson Depth Integration (Frankot-Chellappa algorithm).
    Solves Poisson Equation: laplacian(Z) = d(p)/dx + d(q)/dy
    where p = -Nx / Nz, q = -Ny / Nz
    """
    norm_float = normal_map_rgb.astype(np.float64) / 255.0 * 2.0 - 1.0
    Nx = norm_float[:, :, 0]
    Ny = norm_float[:, :, 1]
    Nz = norm_float[:, :, 2]
    Nz = np.maximum(Nz, 1e-5)

    p = -Nx / Nz
    q = -Ny / Nz

    dp_dx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    dq_dy = cv2.Sobel(q, cv2.CV_64F, 0, 1, ksize=3)
    divergence = dp_dx + dq_dy

    H, W = divergence.shape
    div_fft = fft2(divergence)

    u = fftfreq(H).reshape(-1, 1)
    v = fftfreq(W).reshape(1, -1)

    denom = -4.0 * (np.pi ** 2) * (u**2 + v**2)
    denom[0, 0] = 1.0

    Z_fft = div_fft / denom
    Z_fft[0, 0] = 0.0

    Z = np.real(ifft2(Z_fft))

    z_min, z_max = np.percentile(Z, (1, 99))
    if z_max > z_min:
        Z_norm = np.clip((Z - z_min) / (z_max - z_min), 0.0, 1.0)
    else:
        Z_norm = np.zeros_like(Z)

    return (Z_norm * 255.0).astype(np.uint8)


# =====================================================================
# 2. PyTorch U-Net Surface Normal Estimator & Cosine Angular Loss
# =====================================================================

class CosineAngularLoss(nn.Module):
    """
    Cosine Angular Loss for Normal Map Estimation:
    L_angular = 1 - (1/N) * sum(pred_normal . gt_normal)
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # L2 Normalize predicted normal vectors along channel axis
        pred_norm = F.normalize(pred, p=2, dim=1, eps=self.eps)
        target_norm = F.normalize(target, p=2, dim=1, eps=self.eps)

        # Dot product along channel dimension (B, 1, H, W)
        dot_product = torch.sum(pred_norm * target_norm, dim=1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)

        loss = 1.0 - dot_product

        if mask is not None:
            mask = mask.float()
            loss = torch.sum(loss * mask) / (torch.sum(mask) * pred.shape[1] + self.eps)
        else:
            loss = torch.mean(loss)

        return loss


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNetNormalEstimator(nn.Module):
    """
    Lightweight Encoder-Decoder PyTorch U-Net Architecture with Skip Connections
    Input: RGB Image (B, 3, H, W)
    Output: Surface Normal Map Vectors (B, 3, H, W) in range [-1, 1]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)

        # Output projection layer
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_bot = self.bottleneck(x4)

        x = self.up1(x_bot)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        out = self.outc(x)
        # Normalize to unit 3D vectors
        out = F.normalize(out, p=2, dim=1)
        return out


_global_model = None

def get_unet_model(model_path: str = None) -> UNetNormalEstimator:
    """Instantiate or load cached PyTorch U-Net model."""
    global _global_model
    if _global_model is not None:
        return _global_model

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = UNetNormalEstimator(in_channels=3, out_channels=3)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"[Engine] Loaded trained U-Net model weights from '{model_path}'")
        except Exception as e:
            print(f"[Engine] Failed to load U-Net weights ({e}), using initialized weights.")
    else:
        print("[Engine] U-Net model initialized with PyTorch default weights.")

    model.eval()
    _global_model = model
    return _global_model


def estimate_normal_map_unet(rgb_img: np.ndarray, model_path: str = None) -> np.ndarray:
    """
    Predict surface normal map using PyTorch U-Net model.
    Accepts RGB uint8 image (H, W, 3), returns OpenGL normal map uint8 (H, W, 3).
    """
    model = get_unet_model(model_path)

    h_orig, w_orig = rgb_img.shape[:2]

    # Resize to multiple of 16 for U-Net architecture
    h_pad = ((h_orig + 15) // 16) * 16
    w_pad = ((w_orig + 15) // 16) * 16

    img_resized = cv2.resize(rgb_img, (w_pad, h_pad), interpolation=cv2.INTER_AREA)

    # Convert to Tensor (B, 3, H, W) normalized to [0, 1]
    tensor_in = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        pred_normal_tensor = model(tensor_in)  # (1, 3, H, W) in [-1, 1]

    pred_normal_np = pred_normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    if (h_pad, w_pad) != (h_orig, w_orig):
        pred_normal_np = cv2.resize(pred_normal_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Normalize vectors
    norm = np.linalg.norm(pred_normal_np, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    unit_normals = pred_normal_np / norm

    # Remap OpenGL vectors [-1, 1] -> [0, 255]
    R = ((unit_normals[:, :, 0] * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    G = ((unit_normals[:, :, 1] * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    B = ((unit_normals[:, :, 2] * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)


# =====================================================================
# 3. Quantitative Evaluation Metrics
# =====================================================================

def evaluate_normal_angular_metrics(pred_normal_rgb: np.ndarray, gt_normal: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Calculate extended surface normal angular metrics:
    - MAE: Mean Angular Error in degrees
    - % < 11.25°: Percentage of valid pixels with angular error < 11.25 deg
    - % < 22.5°: Percentage of valid pixels with angular error < 22.5 deg
    """
    if gt_normal.dtype == np.uint8:
        n2 = gt_normal.astype(np.float64) / 255.0 * 2.0 - 1.0
    else:
        n2 = gt_normal.astype(np.float64)

    n1 = pred_normal_rgb.astype(np.float64) / 255.0 * 2.0 - 1.0

    if n1.shape[:2] != n2.shape[:2]:
        n2 = cv2.resize(n2, (n1.shape[1], n1.shape[0]))
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (n1.shape[1], n1.shape[0]), interpolation=cv2.INTER_NEAREST)

    n1_norm = n1 / np.maximum(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-8)
    n2_norm = n2 / np.maximum(np.linalg.norm(n2, axis=-1, keepdims=True), 1e-8)

    dot_prod = np.sum(n1_norm * n2_norm, axis=-1)
    dot_prod_clipped = np.clip(dot_prod, -1.0, 1.0)

    angular_error_deg = np.arccos(dot_prod_clipped) * (180.0 / np.pi)

    if mask is not None:
        valid_indices = mask > 0
        valid_errors = angular_error_deg[valid_indices] if np.any(valid_indices) else angular_error_deg.ravel()
    else:
        valid_errors = angular_error_deg.ravel()

    mae = float(np.mean(valid_errors))
    pct_11_25 = float(np.mean(valid_errors < 11.25) * 100.0)
    pct_22_5 = float(np.mean(valid_errors < 22.5) * 100.0)

    return {
        "mae": round(mae, 2),
        "pct_11_25": round(pct_11_25, 2),
        "pct_22_5": round(pct_22_5, 2)
    }


def evaluate_map_metrics(pred_map: np.ndarray, gt_map: np.ndarray, mask: np.ndarray = None) -> dict:
    """Calculate MSE, PSNR (dB), and SSIM between predicted map and ground truth."""
    if pred_map.shape != gt_map.shape:
        gt_map = cv2.resize(gt_map, (pred_map.shape[1], pred_map.shape[0]))
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (pred_map.shape[1], pred_map.shape[0]), interpolation=cv2.INTER_NEAREST)

    diff_sq = (pred_map.astype(np.float64) - gt_map.astype(np.float64)) ** 2

    if mask is not None:
        if len(pred_map.shape) == 3 and len(mask.shape) == 2:
            mask_3d = np.stack([mask] * pred_map.shape[2], axis=-1)
        else:
            mask_3d = mask
        valid_diffs = diff_sq[mask_3d > 0]
        mse = float(np.mean(valid_diffs)) if len(valid_diffs) > 0 else float(np.mean(diff_sq))
    else:
        mse = float(np.mean(diff_sq))

    psnr = 100.0 if mse < 1e-10 else float(10.0 * np.log10((255.0 ** 2) / mse))

    try:
        if len(pred_map.shape) == 3 and pred_map.shape[2] == 3:
            ssim_val = float(ssim_func(pred_map, gt_map, channel_axis=2, data_range=255))
        else:
            ssim_val = float(ssim_func(pred_map, gt_map, data_range=255))
    except Exception:
        ssim_val = 0.0

    return {
        "mse": round(mse, 4),
        "psnr": round(psnr, 2),
        "ssim": round(ssim_val, 4)
    }


def image_to_base64(img_array: np.ndarray, fmt: str = "PNG") -> str:
    """Convert numpy uint8 image array to Base64 data URL string."""
    pil_img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    pil_img.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_str}"


# =====================================================================
# 4. Unified Surface Map Processing Pipeline
# =====================================================================

def process_surface_maps(rgb_image: np.ndarray, method: str = "both",
                         gt_normal: np.ndarray = None, gt_roughness: np.ndarray = None,
                         gt_height: np.ndarray = None, gt_mask: np.ndarray = None) -> dict:
    """
    Run surface map estimation (DIP and/or PyTorch U-Net), calculate evaluation metrics against GT,
    and return Base64 image maps and metric scorecard JSON.
    Methods: 'dip', 'unet', or 'both'
    """
    _, gray_img = preprocess_image(rgb_image)

    # 1. Normal Estimation
    dip_normal = estimate_normal_map_dip(gray_img, scale=2.0)
    unet_normal = estimate_normal_map_unet(rgb_image)

    # Active normal map to present by default in 2D/3D
    primary_normal = unet_normal if method == "unet" else dip_normal

    # 2. Roughness Map Estimation (2D DFT)
    pred_roughness = estimate_roughness_map_dft(gray_img)

    # 3. Height Map Estimation (Poisson Integration)
    pred_height = estimate_height_map(primary_normal)

    # 4. Metric Scorecard Calculation
    metrics_dip = {"mae": None, "pct_11_25": None, "pct_22_5": None, "mse": None, "psnr": None, "ssim": None}
    metrics_unet = {"mae": None, "pct_11_25": None, "pct_22_5": None, "mse": None, "psnr": None, "ssim": None}

    if gt_normal is not None:
        gt_norm_rgb = gt_normal if gt_normal.dtype == np.uint8 else diode_npy_to_rgb(gt_normal, gt_mask)

        # Evaluate DIP Normal Map
        ang_dip = evaluate_normal_angular_metrics(dip_normal, gt_normal, mask=gt_mask)
        map_dip = evaluate_map_metrics(dip_normal, gt_norm_rgb, mask=gt_mask)
        metrics_dip.update(ang_dip)
        metrics_dip.update(map_dip)

        # Evaluate PyTorch U-Net Normal Map
        ang_unet = evaluate_normal_angular_metrics(unet_normal, gt_normal, mask=gt_mask)
        map_unet = evaluate_map_metrics(unet_normal, gt_norm_rgb, mask=gt_mask)
        metrics_unet.update(ang_unet)
        metrics_unet.update(map_unet)

    roughness_metrics = evaluate_map_metrics(pred_roughness, gt_roughness, mask=gt_mask) if gt_roughness is not None else {"mse": None, "psnr": None, "ssim": None}
    height_metrics = evaluate_map_metrics(pred_height, gt_height, mask=gt_mask) if gt_height is not None else {"mse": None, "psnr": None, "ssim": None}

    maps_b64 = {
        "input_rgb": image_to_base64(rgb_image, fmt="JPEG"),
        "normal": image_to_base64(primary_normal, fmt="PNG"),
        "normal_dip": image_to_base64(dip_normal, fmt="PNG"),
        "normal_unet": image_to_base64(unet_normal, fmt="PNG"),
        "roughness": image_to_base64(pred_roughness, fmt="PNG"),
        "height": image_to_base64(pred_height, fmt="PNG")
    }

    return {
        "maps": maps_b64,
        "metrics": {
            "normal_dip": metrics_dip,
            "normal_unet": metrics_unet,
            "normal": metrics_unet if method == "unet" else metrics_dip,
            "roughness": roughness_metrics,
            "height": height_metrics
        }
    }


def diode_npy_to_rgb(normal_npy: np.ndarray, mask_npy: np.ndarray = None) -> np.ndarray:
    """Helper to render float32 DIODE normal array [-1, 1] as RGB OpenGL image."""
    if normal_npy.dtype != np.float32 and normal_npy.dtype != np.float64:
        normal_npy = normal_npy.astype(np.float64)
    norm = np.linalg.norm(normal_npy, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    unit_normals = normal_npy / norm
    normal_rgb = ((unit_normals * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    if mask_npy is not None:
        mask_3d = np.stack([mask_npy] * 3, axis=-1) if len(mask_npy.shape) == 2 else mask_npy
        neutral = np.array([128, 128, 255], dtype=np.uint8)
        normal_rgb = np.where(mask_3d > 0, normal_rgb, neutral)
    return normal_rgb


if __name__ == "__main__":
    print("[Engine Check] Verifying PyTorch U-Net and Classical DIP Estimator...")
    dummy_input = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = process_surface_maps(dummy_input, method="both")
    print(f"[Engine Check] Success! Generated maps: {list(res['maps'].keys())}")
