#!/usr/bin/env python3
"""
Automated Pipeline Verification & Testing Suite
Iterates through all downloaded dataset samples in data/rgb/, executes Classical DIP & Deep Learning map estimators,
evaluates quantitative metrics against ground-truth pairs in data/ground_truth/, and outputs a formatted terminal summary table.
"""

import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dip_engine import (
    estimate_normals_sobel,
    estimate_roughness_dft,
    estimate_height_poisson,
    evaluate_normal_mae,
    evaluate_scalar_metrics,
    diode_npy_to_rgb
)
from engine import estimate_normal_map_unet, evaluate_normal_angular_metrics

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RGB_DIR = DATA_DIR / "rgb"
GT_NORMALS_DIR = DATA_DIR / "ground_truth" / "normals"
GT_ROUGH_DIR = DATA_DIR / "ground_truth" / "roughness"
GT_HEIGHT_DIR = DATA_DIR / "ground_truth" / "height"


def run_pipeline_verification():
    if not RGB_DIR.exists():
        print("[ERROR] data/rgb directory does not exist! Please run data_downloader.py first.")
        sys.exit(1)

    rgb_files = sorted([p for p in RGB_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not rgb_files:
        print("[ERROR] No RGB image samples found in data/rgb! Generating synthetic fallback dataset...")
        from data_sync import ensure_dataset_ready
        ensure_dataset_ready()
        rgb_files = sorted([p for p in RGB_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    print("\n" + "=" * 105)
    print(f"   SURFACE MAP ESTIMATION & GT EVALUATION SUITE ({len(rgb_files)} DATASET SAMPLES FOUND)")
    print("=" * 105)

    headers = f"{'Sample Asset ID':<24} | {'DIP MAE':<9} | {'%<11.25°':<8} | {'U-Net MAE':<9} | {'Rough PSNR':<10} | {'Rough SSIM':<10} | {'Height SSIM':<11} | {'GT Pair'}"
    print(headers)
    print("-" * 105)

    results = []

    for rgb_path in rgb_files:
        asset_id = rgb_path.stem

        # 1. Read input RGB image
        rgb_img_bgr = cv2.imread(str(rgb_path))
        if rgb_img_bgr is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Estimate Surface Maps
        pred_normal_dip = estimate_normals_sobel(rgb_img)
        pred_normal_unet = estimate_normal_map_unet(rgb_img)
        pred_roughness = estimate_roughness_dft(rgb_img)
        pred_height = estimate_height_poisson(pred_normal_dip)

        # 3. Load Ground-Truth Maps if available
        nor_npy_path = GT_NORMALS_DIR / f"{asset_id}_normal.npy"
        nor_mask_path = GT_NORMALS_DIR / f"{asset_id}_mask.npy"
        nor_png_path = GT_NORMALS_DIR / f"{asset_id}.png"
        rough_path = GT_ROUGH_DIR / f"{asset_id}.png"
        height_path = GT_HEIGHT_DIR / f"{asset_id}.png"

        gt_normal = None
        gt_mask = None
        gt_roughness = None
        gt_height = None

        if nor_npy_path.exists():
            gt_normal = np.load(nor_npy_path)
            if nor_mask_path.exists():
                gt_mask = np.load(nor_mask_path)
        elif nor_png_path.exists():
            nor_bgr = cv2.imread(str(nor_png_path))
            if nor_bgr is not None:
                gt_normal = cv2.cvtColor(nor_bgr, cv2.COLOR_BGR2RGB)

        if rough_path.exists():
            gt_roughness = cv2.imread(str(rough_path), cv2.IMREAD_GRAYSCALE)

        if height_path.exists():
            gt_height = cv2.imread(str(height_path), cv2.IMREAD_GRAYSCALE)

        # 4. Compute Metrics
        has_gt = gt_normal is not None
        dip_mae_str, pct11_str, unet_mae_str = "N/A", "N/A", "N/A"
        rough_psnr_str, rough_ssim_str, height_ssim_str = "N/A", "N/A", "N/A"

        dip_mae_val, pct11_val, unet_mae_val = None, None, None
        rough_psnr_val, rough_ssim_val, height_ssim_val = None, None, None

        if has_gt:
            # Normal MAE for DIP
            gt_norm_rgb = gt_normal if gt_normal.dtype == np.uint8 else diode_npy_to_rgb(gt_normal, gt_mask)
            dip_mae, pct11, pct22 = evaluate_normal_mae(pred_normal_dip, gt_normal, mask=gt_mask)
            dip_mae_val, pct11_val = dip_mae, pct11
            dip_mae_str = f"{dip_mae:.2f}°"
            pct11_str = f"{pct11:.1f}%"

            # Normal MAE for U-Net
            unet_res = evaluate_normal_angular_metrics(pred_normal_unet, gt_normal, mask=gt_mask)
            unet_mae_val = unet_res["mae"]
            unet_mae_str = f"{unet_res['mae']:.2f}°"

            # Roughness Metrics
            if gt_roughness is not None:
                r_metrics = evaluate_scalar_metrics(pred_roughness, gt_roughness, mask=gt_mask)
                rough_psnr_val, rough_ssim_val = r_metrics["psnr"], r_metrics["ssim"]
                rough_psnr_str = f"{r_metrics['psnr']:.2f} dB"
                rough_ssim_str = f"{r_metrics['ssim']:.4f}"

            # Height Metrics
            if gt_height is not None:
                h_metrics = evaluate_scalar_metrics(pred_height, gt_height, mask=gt_mask)
                height_ssim_val = h_metrics["ssim"]
                height_ssim_str = f"{h_metrics['ssim']:.4f}"

        row_str = f"{asset_id[:24]:<24} | {dip_mae_str:<9} | {pct11_str:<8} | {unet_mae_str:<9} | {rough_psnr_str:<10} | {rough_ssim_str:<10} | {height_ssim_str:<11} | {'[YES]' if has_gt else '[NO]'}"
        print(row_str)

        results.append({
            "asset_id": asset_id,
            "has_gt": has_gt,
            "dip_mae": dip_mae_val,
            "pct11": pct11_val,
            "unet_mae": unet_mae_val,
            "rough_psnr": rough_psnr_val,
            "rough_ssim": rough_ssim_val,
            "height_ssim": height_ssim_val
        })

    # Summary Statistics
    valid_dip_maes = [r["dip_mae"] for r in results if r["dip_mae"] is not None]
    valid_unet_maes = [r["unet_mae"] for r in results if r["unet_mae"] is not None]
    valid_pct11s = [r["pct11"] for r in results if r["pct11"] is not None]
    valid_rough_psnrs = [r["rough_psnr"] for r in results if r["rough_psnr"] is not None]
    valid_height_ssims = [r["height_ssim"] for r in results if r["height_ssim"] is not None]

    print("=" * 105)
    if valid_dip_maes:
        avg_dip_mae = np.mean(valid_dip_maes)
        avg_unet_mae = np.mean(valid_unet_maes) if valid_unet_maes else 0.0
        avg_pct11 = np.mean(valid_pct11s)
        avg_rough_psnr = np.mean(valid_rough_psnrs) if valid_rough_psnrs else 0.0
        avg_height_ssim = np.mean(valid_height_ssims) if valid_height_ssims else 0.0

        summary = f"{'AVERAGE OVERALL':<24} | {avg_dip_mae:.2f}°    | {avg_pct11:.1f}%   | {avg_unet_mae:.2f}°    | {avg_rough_psnr:.2f} dB   | {'--':<10} | {avg_height_ssim:.4f}      | {len(valid_dip_maes)} pairs"
        print(summary)
    else:
        print("No ground-truth pairs found for evaluation.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_pipeline_verification()