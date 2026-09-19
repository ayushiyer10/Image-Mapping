#!/usr/bin/env python3
"""
Benchmark Runner (benchmark.py)
Executes process_surface_maps() across data/manifest.json, evaluates Trivial Flat Baseline, Classical DIP,
and Deep Learning (MiDaS) surface normal estimation against ground truth, and exports data/benchmark_results.json and data/benchmark_report.md.
"""

import os
import sys
import time
import json
import random
import argparse
import cv2
import numpy as np
from pathlib import Path

# Force UTF-8 encoding for standard output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_PATH = DATA_DIR / "benchmark_results.json"
REPORT_PATH = DATA_DIR / "benchmark_report.md"

from build_manifest import build_manifest
from engine import process_surface_maps, diode_npy_to_rgb, evaluate_normal_angular_metrics, evaluate_map_metrics


def run_benchmark(max_per_source: int = 150, progress_every: int = 10, seed: int = 42):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    if not MANIFEST_PATH.exists():
        manifest = build_manifest(max_diode_samples=max_per_source)
    else:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Group samples by source
    sources_map = {}
    for sample in manifest:
        src = sample.get("source", "polyhaven")
        sources_map.setdefault(src, []).append(sample)

    sampled_manifest = []
    for src in sorted(sources_map.keys()):
        items = sources_map[src]
        if len(items) > max_per_source:
            sampled_manifest.extend(random.sample(items, max_per_source))
        else:
            sampled_manifest.extend(items)

    print(f"\n{'='*105}")
    print(f"   SURFACE MAP BENCHMARK SUITE — FLAT BASELINE vs CLASSICAL DIP vs DEEP LEARNING (MiDaS)")
    print(f"   Total manifest items: {len(manifest)} | Sampled items: {len(sampled_manifest)} (max {max_per_source}/source)")
    print(f"{'='*105}\n")

    results = []
    source_counts = {}

    for idx, sample in enumerate(sampled_manifest, 1):
        asset_id = sample["asset_id"]
        src = sample["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        src_idx = source_counts[src]
        total_src = len([s for s in sampled_manifest if s["source"] == src])

        rgb_abs = BASE_DIR / sample["rgb_path"]
        if not rgb_abs.exists():
            continue

        rgb_img_bgr = cv2.imread(str(rgb_abs))
        if rgb_img_bgr is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img_bgr, cv2.COLOR_BGR2RGB)

        gt_normal = None
        gt_mask = None
        gt_roughness = None
        gt_height = None

        if sample.get("gt_normal_path"):
            gt_nor_abs = BASE_DIR / sample["gt_normal_path"]
            if gt_nor_abs.suffix == ".npy":
                gt_normal = np.load(gt_nor_abs)
                if sample.get("gt_mask_path"):
                    gt_mask_abs = BASE_DIR / sample["gt_mask_path"]
                    if gt_mask_abs.exists():
                        gt_mask = np.load(gt_mask_abs)
            elif gt_nor_abs.exists():
                nor_bgr = cv2.imread(str(gt_nor_abs))
                if nor_bgr is not None:
                    gt_normal = cv2.cvtColor(nor_bgr, cv2.COLOR_BGR2RGB)

        if sample.get("gt_roughness_path"):
            r_abs = BASE_DIR / sample["gt_roughness_path"]
            if r_abs.exists():
                gt_roughness = cv2.imread(str(r_abs), cv2.IMREAD_GRAYSCALE)

        if sample.get("gt_height_path"):
            h_abs = BASE_DIR / sample["gt_height_path"]
            if h_abs.exists():
                gt_height = cv2.imread(str(h_abs), cv2.IMREAD_GRAYSCALE)

        # 1. Evaluate Trivial Flat Baseline (Nx=0, Ny=0, Nz=1 -> RGB [128, 128, 255])
        flat_m = {"mae": None, "pct_11_25": None, "pct_22_5": None, "mse": None, "psnr": None, "ssim": None}
        if gt_normal is not None:
            h, w = rgb_img.shape[:2]
            flat_normal_rgb = np.full((h, w, 3), [128, 128, 255], dtype=np.uint8)
            gt_norm_rgb = gt_normal if gt_normal.dtype == np.uint8 else diode_npy_to_rgb(gt_normal, gt_mask)

            ang_flat = evaluate_normal_angular_metrics(flat_normal_rgb, gt_normal, mask=gt_mask)
            map_flat = evaluate_map_metrics(flat_normal_rgb, gt_norm_rgb, mask=gt_mask)
            flat_m.update(ang_flat)
            flat_m.update(map_flat)

        # 2. Run Surface Map Generation (both Classical DIP and Deep Learning MiDaS)
        res = process_surface_maps(
            rgb_image=rgb_img,
            method="both",
            gt_normal=gt_normal,
            gt_roughness=gt_roughness,
            gt_height=gt_height,
            gt_mask=gt_mask
        )

        metrics = res["metrics"]
        dip_m = metrics.get("normal_dip", {})
        dl_m = metrics.get("normal_dl", {})
        rough_m = metrics.get("roughness", {})
        height_m = metrics.get("height", {})

        elapsed = time.time() - start_time

        if src_idx % progress_every == 0 or src_idx == total_src:
            flat_mae_str = f"{flat_m['mae']:.2f}°" if flat_m.get('mae') is not None else "N/A"
            dip_mae_str = f"{dip_m['mae']:.2f}°" if dip_m.get('mae') is not None else "N/A"
            dl_mae_str = f"{dl_m['mae']:.2f}°" if dl_m.get('mae') is not None else "N/A"
            print(f"[{src} {src_idx}/{total_src}] flat_mae={flat_mae_str} dip_mae={dip_mae_str} dl_mae={dl_mae_str} elapsed={elapsed:.1f}s", flush=True)

        results.append({
            "asset_id": asset_id,
            "source": src,
            "scene_type": sample.get("scene_type", "outdoor"),
            "has_gt": gt_normal is not None,
            "normal_flat": flat_m,
            "normal_dip": dip_m,
            "normal_dl": dl_m,
            "roughness": rough_m,
            "height": height_m
        })

    total_wall_time = time.time() - start_time
    skipped_count = len(manifest) - len(results)

    # Compute Summary Aggregates
    summary_by_source = {}
    all_sources = sorted(list(set(r["source"] for r in results))) + ["overall"]

    for src in all_sources:
        src_results = results if src == "overall" else [r for r in results if r["source"] == src]
        if not src_results:
            continue

        def calc_stats(metric_name, key_name):
            vals = [r[metric_name][key_name] for r in src_results if r[metric_name].get(key_name) is not None]
            if not vals:
                return None
            return round(float(np.mean(vals)), 2 if "ssim" not in key_name else 4)

        def calc_median(metric_name, key_name):
            vals = [r[metric_name][key_name] for r in src_results if r[metric_name].get(key_name) is not None]
            if not vals:
                return None
            return round(float(np.median(vals)), 2)

        summary_by_source[src] = {
            "count": len(src_results),
            "normal_flat": {
                "mean_mae": calc_stats("normal_flat", "mae"),
                "median_mae": calc_median("normal_flat", "mae"),
                "pct_11_25": calc_stats("normal_flat", "pct_11_25"),
                "pct_22_5": calc_stats("normal_flat", "pct_22_5"),
                "mean_psnr": calc_stats("normal_flat", "psnr"),
                "mean_ssim": calc_stats("normal_flat", "ssim")
            },
            "normal_dip": {
                "mean_mae": calc_stats("normal_dip", "mae"),
                "median_mae": calc_median("normal_dip", "mae"),
                "pct_11_25": calc_stats("normal_dip", "pct_11_25"),
                "pct_22_5": calc_stats("normal_dip", "pct_22_5"),
                "mean_psnr": calc_stats("normal_dip", "psnr"),
                "mean_ssim": calc_stats("normal_dip", "ssim")
            },
            "normal_dl": {
                "mean_mae": calc_stats("normal_dl", "mae"),
                "median_mae": calc_median("normal_dl", "mae"),
                "pct_11_25": calc_stats("normal_dl", "pct_11_25"),
                "pct_22_5": calc_stats("normal_dl", "pct_22_5"),
                "mean_psnr": calc_stats("normal_dl", "psnr"),
                "mean_ssim": calc_stats("normal_dl", "ssim")
            },
            "roughness": {
                "mean_psnr": calc_stats("roughness", "psnr"),
                "mean_ssim": calc_stats("roughness", "ssim")
            },
            "height": {
                "mean_psnr": calc_stats("height", "psnr"),
                "mean_ssim": calc_stats("height", "ssim")
            }
        }

    export_json = {
        "metadata": {
            "total_samples_evaluated": len(results),
            "samples_skipped": skipped_count,
            "wall_clock_seconds": round(total_wall_time, 2),
            "max_per_source": max_per_source
        },
        "summary": summary_by_source,
        "samples": results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(export_json, f, indent=2)

    # Generate Markdown Report
    report_lines = [
        "# Surface Map Estimation Benchmark Report",
        "",
        "**Comparison**: Trivial Flat Baseline vs. Classical DIP (Sobel) vs. Deep Learning (MiDaS_small)",
        f"**Evaluated Samples**: {len(results)} | **Wall-clock Time**: {total_wall_time:.2f} seconds",
        "",
        "## Summary Scorecard (Normal Map Metrics)",
        "",
        "| Source | Estimator | Mean MAE (°) | Median MAE (°) | % < 11.25° | % < 22.5° | PSNR (dB) | SSIM |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for src, data in summary_by_source.items():
        flat_info = data["normal_flat"]
        dip_info = data["normal_dip"]
        dl_info = data["normal_dl"]

        report_lines.append(
            f"| **{src.title()}** | Trivial Flat Baseline | {flat_info['mean_mae']}° | {flat_info['median_mae']}° | {flat_info['pct_11_25']}% | {flat_info['pct_22_5']}% | {flat_info['mean_psnr']} | {flat_info['mean_ssim']} |"
        )
        report_lines.append(
            f"| **{src.title()}** | Classical DIP (Sobel) | {dip_info['mean_mae']}° | {dip_info['median_mae']}° | {dip_info['pct_11_25']}% | {dip_info['pct_22_5']}% | {dip_info['mean_psnr']} | {dip_info['mean_ssim']} |"
        )
        report_lines.append(
            f"| **{src.title()}** | Deep Learning (MiDaS) | {dl_info['mean_mae']}° | {dl_info['median_mae']}° | {dl_info['pct_11_25']}% | {dl_info['pct_22_5']}% | {dl_info['mean_psnr']} | {dl_info['mean_ssim']} |"
        )

    report_lines.extend([
        "",
        "## Roughness & Height Maps (PBR & Synthetic)",
        "",
        "| Source | Roughness PSNR | Roughness SSIM | Height PSNR | Height SSIM |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ])

    for src, data in summary_by_source.items():
        r_info = data["roughness"]
        h_info = data["height"]
        r_psnr = f"{r_info['mean_psnr']} dB" if r_info['mean_psnr'] is not None else "N/A"
        r_ssim = f"{r_info['mean_ssim']}" if r_info['mean_ssim'] is not None else "N/A"
        h_psnr = f"{h_info['mean_psnr']} dB" if h_info['mean_psnr'] is not None else "N/A"
        h_ssim = f"{h_info['mean_ssim']}" if h_info['mean_ssim'] is not None else "N/A"

        report_lines.append(f"| **{src.title()}** | {r_psnr} | {r_ssim} | {h_psnr} | {h_ssim} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n{'='*105}")
    print(f"   BENCHMARK COMPLETE IN {total_wall_time:.2f}s!")
    print(f"   Saved JSON: {RESULTS_PATH}")
    print(f"   Saved Report: {REPORT_PATH}")
    print(f"{'='*105}\n")
    return export_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface Map Estimation Benchmark Runner")
    parser.add_argument("--max_per_source", type=int, default=150, help="Maximum samples to evaluate per dataset source")
    parser.add_argument("--progress_every", type=int, default=10, help="Progress log frequency per source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    run_benchmark(max_per_source=args.max_per_source, progress_every=args.progress_every, seed=args.seed)
