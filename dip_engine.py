#!/usr/bin/env python3
"""
Classical DIP (Digital Image Processing) Baseline Engine
Implements classical physics & signal-processing algorithms for surface map estimation:
1. Image Preprocessing (Gamma correction, contrast stretching)
2. Normal Map Estimation (Sobel gradient operators & DIODE .npy converter)
3. Roughness Map Estimation (2D Discrete Fourier Transform (DFT) High-Pass Filtering + Local Variance)
4. Height Map Estimation (Fourier Poisson depth integration / Frankot-Chellappa algorithm)
5. Extended Evaluation Metrics (MAE, % < 11.25°, % < 22.5°, MSE, PSNR, SSIM with mask support)
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from skimage.metrics import structural_similarity as ssim_func
from PIL import Image
import io
import base64


def preprocess_image(rgb_image: np.ndarray, gamma: float = 2.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess RGB image:
    1. Apply gamma correction (sRGB -> Linear space).
    2. Convert to Grayscale.
    3. Apply percentile-based contrast stretching.
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

    # Contrast stretching via 1st and 99th percentiles
    p1, p99 = np.percentile(gray, (1, 99))
    if p99 > p1:
        gray_stretched = np.clip((gray - p1) / (p99 - p1), 0.0, 1.0)
    else:
        gray_stretched = gray

    return linear_img, gray_stretched


def estimate_normal_map(gray_img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    Estimate OpenGL Normal Map from grayscale intensity image using Gaussian-smoothed Sobel gradient operators.
    Gradient: Ix = dI/dx, Iy = dI/dy
    Surface Normal: N = (-Ix * scale, -Iy * scale, 1.0) / ||N||
    OpenGL Normal Map encoding: R=Nx, G=Ny, B=Nz mapped from [-1, 1] to uint8 [0, 255].
    """
    gray_float = gray_img.astype(np.float64)
    if gray_float.max() > 1.0:
        gray_float = gray_float / 255.0

    gray_smooth = cv2.GaussianBlur(gray_float, (5, 5), 1.0)

    Ix = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_smooth, cv2.CV_64F, 0, 1, ksize=3)

    Nx = -Ix * scale
    Ny = -Iy * scale
    Nz = np.ones_like(gray_smooth, dtype=np.float64)

    magnitude = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    magnitude = np.maximum(magnitude, 1e-8)

    Nx_unit = Nx / magnitude
    Ny_unit = Ny / magnitude
    Nz_unit = Nz / magnitude

    R = ((Nx_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    G = ((Ny_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    B = ((Nz_unit * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)


def diode_npy_to_rgb(normal_npy: np.ndarray, mask_npy: np.ndarray = None) -> np.ndarray:
    """
    Convert DIODE dataset 3D float32 normal array (values in [-1, 1]) to visual uint8 RGB OpenGL normal map.
    Applies optional mask for invalid pixels.
    """
    if normal_npy.dtype != np.float32 and normal_npy.dtype != np.float64:
        normal_npy = normal_npy.astype(np.float64)

    # Normalize vectors
    norm = np.linalg.norm(normal_npy, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    unit_normals = normal_npy / norm

    # Map [-1, 1] to [0, 255]
    normal_rgb = ((unit_normals * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    if mask_npy is not None:
        # Mask out invalid pixels (set to neutral blue [128, 128, 255])
        mask_3d = np.stack([mask_npy] * 3, axis=-1) if len(mask_npy.shape) == 2 else mask_npy
        neutral = np.array([128, 128, 255], dtype=np.uint8)
        normal_rgb = np.where(mask_3d > 0, normal_rgb, neutral)

    return normal_rgb


def estimate_roughness_map_dft(gray_img: np.ndarray, cutoff_radius: float = 30.0) -> np.ndarray:
    """
    Estimate Surface Roughness Map using 2D Discrete Fourier Transform (DFT) High-Pass Filtering.
    High-pass frequency components isolate micro-surface details and high-frequency textures.
    """
    H, W = gray_img.shape

    # 1. 2D DFT & Shift DC component to center
    F = fftshift(fft2(gray_img))

    # 2. Construct 2D Gaussian High-Pass Frequency Filter Mask
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D_sq = U**2 + V**2

    # High-pass filter mask: H_hp(u,v) = 1 - exp(-D^2 / (2 * D0^2))
    D0_sq = max(cutoff_radius**2, 1.0)
    H_hp = 1.0 - np.exp(-D_sq / (2.0 * D0_sq))

    # 3. Filter in frequency domain
    F_filtered = F * H_hp

    # 4. Inverse DFT to get spatial high-frequency micro-texture image
    high_freq_spatial = np.abs(ifft2(ifftshift(F_filtered)))

    # 5. Local variance calculation in sliding window
    local_mean = cv2.blur(high_freq_spatial, (9, 9))
    local_sq_mean = cv2.blur(high_freq_spatial**2, (9, 9))
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    local_std = np.sqrt(local_var)

    # 6. Combine high-pass magnitude with local variance
    micro_texture = high_freq_spatial * 0.5 + local_std * 0.5

    # Base PBR roughness offset (0.65) + relative micro-texture variance
    base_roughness = 0.65
    raw_roughness = base_roughness + (micro_texture - micro_texture.mean()) * 0.8

    return (np.clip(raw_roughness, 0.0, 1.0) * 255.0).astype(np.uint8)


def estimate_height_map(normal_map_rgb: np.ndarray) -> np.ndarray:
    """
    Reconstruct Depth / Height Map from 3D Normal Vectors using 2D Fourier Poisson Integration.
    Solves Poisson Equation: laplacian(Z) = d(p)/dx + d(q)/dy
    where p = -Nx / Nz, q = -Ny / Nz
    """
    norm_float = normal_map_rgb.astype(np.float64) / 255.0 * 2.0 - 1.0
    Nx = norm_float[:, :, 0]
    Ny = norm_float[:, :, 1]
    Nz = norm_float[:, :, 2]
    Nz = np.maximum(Nz, 1e-5)

    p = -Nx / Nz
    q = Ny / Nz

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


def evaluate_normal_angular_metrics(pred_normal_rgb: np.ndarray, gt_normal: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Calculate extended normal angular error metrics:
    - MAE (Mean Angular Error in degrees)
    - % < 11.25°: Percentage of valid pixels with angular error < 11.25 deg
    - % < 22.5°: Percentage of valid pixels with angular error < 22.5 deg
    Supports both RGB normal maps and DIODE float32 .npy arrays.
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
        if not np.any(valid_indices):
            valid_errors = angular_error_deg.ravel()
        else:
            valid_errors = angular_error_deg[valid_indices]
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
    """Calculate MSE, PSNR, and SSIM between predicted map and ground-truth map."""
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
    """Convert numpy image array (RGB or Grayscale uint8) to Base64 data URL string."""
    pil_img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    pil_img.save(buffered, format=fmt)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_str}"


def process_image_pipeline(rgb_image: np.ndarray, gt_normal: np.ndarray = None, 
                           gt_roughness: np.ndarray = None, gt_height: np.ndarray = None,
                           gt_mask: np.ndarray = None) -> dict:
    """
    Full Upgraded DIP Baseline Pipeline:
    1. Preprocessing
    2. Estimate Normal, Roughness (via 2D DFT), and Height maps
    3. Calculate extended evaluation metrics (MAE, % < 11.25°, % < 22.5°, PSNR, SSIM) with mask support
    4. Return Base64 encoded maps and metrics scorecard JSON
    """
    _, gray_img = preprocess_image(rgb_image)

    pred_normal = estimate_normal_map(gray_img, scale=2.0)
    pred_roughness = estimate_roughness_map_dft(gray_img)
    pred_height = estimate_height_map(pred_normal)

    metrics = {
        "normal": {"mae": None, "pct_11_25": None, "pct_22_5": None, "mse": None, "psnr": None, "ssim": None},
        "roughness": {"mse": None, "psnr": None, "ssim": None},
        "height": {"mse": None, "psnr": None, "ssim": None}
    }

    if gt_normal is not None:
        angular_metrics = evaluate_normal_angular_metrics(pred_normal, gt_normal, mask=gt_mask)
        gt_normal_rgb = gt_normal if gt_normal.dtype == np.uint8 else diode_npy_to_rgb(gt_normal, gt_mask)
        normal_metrics = evaluate_map_metrics(pred_normal, gt_normal_rgb, mask=gt_mask)
        
        metrics["normal"].update(angular_metrics)
        metrics["normal"].update(normal_metrics)

    if gt_roughness is not None:
        rough_metrics = evaluate_map_metrics(pred_roughness, gt_roughness, mask=gt_mask)
        metrics["roughness"].update(rough_metrics)

    if gt_height is not None:
        height_metrics = evaluate_map_metrics(pred_height, gt_height, mask=gt_mask)
        metrics["height"].update(height_metrics)

# Function Aliases for Test Suite & Pipeline Compatibility
def estimate_normals_sobel(rgb_img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    _, gray_img = preprocess_image(rgb_img)
    return estimate_normal_map(gray_img, scale=scale)

def estimate_roughness_dft(rgb_img: np.ndarray, cutoff_radius: float = 30.0) -> np.ndarray:
    if len(rgb_img.shape) == 3:
        _, gray_img = preprocess_image(rgb_img)
    else:
        gray_img = rgb_img.astype(np.float32) / 255.0 if rgb_img.dtype == np.uint8 else rgb_img
    return estimate_roughness_map_dft(gray_img, cutoff_radius=cutoff_radius)

def estimate_height_poisson(normal_map_rgb: np.ndarray) -> np.ndarray:
    return estimate_height_map(normal_map_rgb)

def evaluate_normal_mae(pred_normal_rgb: np.ndarray, gt_normal: np.ndarray, mask: np.ndarray = None):
    res = evaluate_normal_angular_metrics(pred_normal_rgb, gt_normal, mask=mask)
    return res["mae"], res["pct_11_25"], res["pct_22_5"]

def evaluate_scalar_metrics(pred_map: np.ndarray, gt_map: np.ndarray, mask: np.ndarray = None):
    return evaluate_map_metrics(pred_map, gt_map, mask=mask)
