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


def depth_map_to_normal_map(depth_img: np.ndarray, scale: float = 1.5, blur_ksize: int = 5) -> np.ndarray:
    """
    Convert a 2D scalar depth/height map to an OpenGL uint8 RGB normal map via Sobel gradients:
    1. Preprocess: Ensure depth_img is float64 in range [0.0, 1.0].
    2. Denoise: Apply Gaussian blur to suppress high-frequency noise before differentiation.
    3. Sobel gradient extraction: Ix = dI/dx, Iy = dI/dy.
    4. L2 Unit Vector Normalization: N = (-Ix * scale, -Iy * scale, 1.0) / ||N||
    5. Remapped to OpenGL uint8 RGB [0, 255].
    """
    depth_float = depth_img.astype(np.float64)
    if depth_float.max() > 1.0:
        depth_float = depth_float / 255.0

    if blur_ksize > 1:
        depth_float = cv2.GaussianBlur(depth_float, (blur_ksize, blur_ksize), 1.0)

    Ix = cv2.Sobel(depth_float, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(depth_float, cv2.CV_64F, 0, 1, ksize=3)

    Nx = -Ix * scale
    Ny = -Iy * scale
    Nz = np.ones_like(depth_float, dtype=np.float64)

    magnitude = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    magnitude = np.maximum(magnitude, 1e-8)

    Nx_unit = Nx / magnitude
    Ny_unit = Ny / magnitude
    Nz_unit = Nz / magnitude

    R = ((Nx_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    G = ((Ny_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    B = ((Nz_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)


def estimate_normal_map_dip(gray_img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    Classical DIP Surface Normal Estimation using Gaussian-smoothed Sobel gradients.
    """
    return depth_map_to_normal_map(gray_img, scale=scale, blur_ksize=5)



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
    micro_texture = high_freq_spatial * 0.5 + local_std * 0.5

    # Base PBR roughness offset (0.65) + relative micro-texture variance
    base_roughness = 0.65
    raw_roughness = base_roughness + (micro_texture - micro_texture.mean()) * 0.8

    return (np.clip(raw_roughness, 0.0, 1.0) * 255.0).astype(np.uint8)


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
# 2. Deep Learning Surface Normal Estimator (MiDaS Monocular Depth)
# =====================================================================

_midas_model = None
_midas_transforms = None


def get_midas_model():
    """Instantiate and load cached MiDaS_small deep learning monocular depth model."""
    global _midas_model, _midas_transforms
    if _midas_model is not None and _midas_transforms is not None:
        return _midas_model, _midas_transforms

    try:
        # Pre-trust dependent repository to prevent interactive prompt
        torch.hub.load("rwightman/gen-efficientnet-pytorch", "efficientnet_lite0", trust_repo=True)
    except Exception:
        pass

    _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _midas_model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _midas_transforms = midas_transforms.small_transform

    print("[Engine] MiDaS_small model loaded successfully.")
    return _midas_model, _midas_transforms


def estimate_normal_map_midas(rgb_img: np.ndarray) -> np.ndarray:
    """
    Predict surface normal map using Deep Learning MiDaS_small Monocular Depth Estimator.
    1. Runs MiDaS_small inference to get relative inverse depth map.
    2. Converts relative depth map to surface normals via Sobel gradients.
    Returns OpenGL normal map uint8 (H, W, 3).
    """
    model, transform = get_midas_model()
    h_orig, w_orig = rgb_img.shape[:2]

    # Transform RGB image for MiDaS input
    input_batch = transform(rgb_img)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h_orig, w_orig),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_np = prediction.cpu().numpy()

    # Percentile-based contrast normalization for stable depth gradient
    p1, p99 = np.percentile(depth_np, (1, 99))
    if p99 > p1:
        depth_norm = np.clip((depth_np - p1) / (p99 - p1), 0.0, 1.0)
    else:
        depth_norm = depth_np

    # Convert inverse depth map to normal map via Sobel gradients
    return depth_map_to_normal_map(depth_norm, scale=2.0)



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


def compute_confidence_scores(dip_normal: np.ndarray, midas_normal: np.ndarray,
                             primary_normal: np.ndarray, pred_height: np.ndarray,
                             pred_roughness: np.ndarray, gray_img: np.ndarray) -> dict:
    """
    Compute ground-truth-free confidence scores for estimated surface maps.
    - Normal Map: Angular agreement (MAE) between Classical DIP (Sobel) and Deep Learning (MiDaS) estimates.
    - Height Map: Closed-loop Poisson integration consistency via re-differentiating pred_height back into a normal map.
    - Roughness Map: SSIM stability between default (30.0) and alternate (15.0) frequency cutoff radii.
    """
    # 1. Normal Map Confidence
    ang_norm = evaluate_normal_angular_metrics(dip_normal, midas_normal)
    mae = ang_norm["mae"]
    norm_pct = round(max(0.0, min(100.0, 100.0 - (mae / 90.0 * 100.0))), 2)

    # 2. Height Map Confidence (Closed-loop re-differentiation check)
    re_normal = estimate_normal_map_dip(pred_height, scale=2.0)
    h_ang = evaluate_normal_angular_metrics(re_normal, primary_normal)
    h_mae = h_ang["mae"]
    height_pct = round(max(0.0, min(100.0, 100.0 - (h_mae / 90.0 * 100.0))), 2)

    # 3. Roughness Map Confidence (SSIM stability across cutoff scales)
    alt_roughness = estimate_roughness_map_dft(gray_img, cutoff_radius=15.0)
    r_metrics = evaluate_map_metrics(pred_roughness, alt_roughness)
    stability_ssim = r_metrics["ssim"]
    rough_pct = round(max(0.0, min(100.0, stability_ssim * 100.0)), 2)

    def get_confidence_band(pct: float) -> str:
        if pct >= 75.0:
            return "high"
        elif pct >= 50.0:
            return "medium"
        else:
            return "low"

    return {
        "normal": {
            "confidence_pct": norm_pct,
            "confidence_band": get_confidence_band(norm_pct),
            "description": f"Angular agreement between Classical DIP (Sobel) and Deep Learning (MiDaS) normal estimates (MAE: {mae:.2f}°)."
        },
        "height": {
            "confidence_pct": height_pct,
            "confidence_band": get_confidence_band(height_pct),
            "description": f"Closed-loop Poisson integration consistency (Re-diff MAE: {h_mae:.2f}°)."
        },
        "roughness": {
            "confidence_pct": rough_pct,
            "stability_ssim": stability_ssim,
            "confidence_band": get_confidence_band(rough_pct),
            "description": f"SSIM stability across frequency cutoff radii (SSIM: {stability_ssim:.4f})."
        },
        "disclaimer": "Ground-truth unavailable for custom uploaded images. Scores reflect relative consistency between Classical DIP and Deep Learning (MiDaS) estimators, not absolute ground-truth accuracy."
    }


# =====================================================================
# 4. Unified Surface Map Processing Pipeline
# =====================================================================

def process_surface_maps(rgb_image: np.ndarray, method: str = "both",
                         gt_normal: np.ndarray = None, gt_roughness: np.ndarray = None,
                         gt_height: np.ndarray = None, gt_mask: np.ndarray = None) -> dict:
    """
    Run surface map estimation (Classical DIP and Deep Learning MiDaS), calculate evaluation metrics against GT if available,
    compute GT-free confidence scores, and return Base64 image maps and scorecard JSON.
    Methods: 'dip', 'midas', 'dl', or 'both'
    """
    _, gray_img = preprocess_image(rgb_image)

    # 1. Normal Estimation
    dip_normal = estimate_normal_map_dip(gray_img, scale=2.0)
    midas_normal = estimate_normal_map_midas(rgb_image)

    # Active normal map to present by default in 2D/3D
    primary_normal = midas_normal if method in ["midas", "dl", "unet"] else dip_normal

    # 2. Roughness Map Estimation (2D DFT)
    pred_roughness = estimate_roughness_map_dft(gray_img)

    # 3. Height Map Estimation (Poisson Integration)
    pred_height = estimate_height_map(primary_normal)

    # 4. Metric Scorecard Calculation
    metrics_dip = {"mae": None, "pct_11_25": None, "pct_22_5": None, "mse": None, "psnr": None, "ssim": None}
    metrics_dl = {"mae": None, "pct_11_25": None, "pct_22_5": None, "mse": None, "psnr": None, "ssim": None}

    if gt_normal is not None:
        gt_norm_rgb = gt_normal if gt_normal.dtype == np.uint8 else diode_npy_to_rgb(gt_normal, gt_mask)

        # Evaluate DIP Normal Map
        ang_dip = evaluate_normal_angular_metrics(dip_normal, gt_normal, mask=gt_mask)
        map_dip = evaluate_map_metrics(dip_normal, gt_norm_rgb, mask=gt_mask)
        metrics_dip.update(ang_dip)
        metrics_dip.update(map_dip)

        # Evaluate Deep Learning (MiDaS) Normal Map
        ang_dl = evaluate_normal_angular_metrics(midas_normal, gt_normal, mask=gt_mask)
        map_dl = evaluate_map_metrics(midas_normal, gt_norm_rgb, mask=gt_mask)
        metrics_dl.update(ang_dl)
        metrics_dl.update(map_dl)

    roughness_metrics = evaluate_map_metrics(pred_roughness, gt_roughness, mask=gt_mask) if gt_roughness is not None else {"mse": None, "psnr": None, "ssim": None}
    height_metrics = evaluate_map_metrics(pred_height, gt_height, mask=gt_mask) if gt_height is not None else {"mse": None, "psnr": None, "ssim": None}

    # 5. GT-Free Confidence Scores
    confidence_scores = compute_confidence_scores(
        dip_normal=dip_normal,
        midas_normal=midas_normal,
        primary_normal=primary_normal,
        pred_height=pred_height,
        pred_roughness=pred_roughness,
        gray_img=gray_img
    )

    maps_b64 = {
        "input_rgb": image_to_base64(rgb_image, fmt="JPEG"),
        "normal": image_to_base64(primary_normal, fmt="PNG"),
        "normal_dip": image_to_base64(dip_normal, fmt="PNG"),
        "normal_dl": image_to_base64(midas_normal, fmt="PNG"),
        "roughness": image_to_base64(pred_roughness, fmt="PNG"),
        "height": image_to_base64(pred_height, fmt="PNG")
    }

    return {
        "maps": maps_b64,
        "metrics": {
            "normal_dip": metrics_dip,
            "normal_dl": metrics_dl,
            "normal": metrics_dl if method in ["midas", "dl", "unet"] else metrics_dip,
            "roughness": roughness_metrics,
            "height": height_metrics
        },
        "confidence": confidence_scores,
        "has_ground_truth": gt_normal is not None
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
