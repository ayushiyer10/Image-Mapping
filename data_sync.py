#!/usr/bin/env python3
"""
Dataset Packager & Auto-Sync Module
Handles packaging paired surface dataset maps into surface_dataset_v1.tar.gz for GitHub Release uploads,
and automatic syncing/downloading (with offline synthetic material generation fallback).
"""

import os
import sys
import tarfile
import argparse
import requests
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"

# Configurable GitHub Release URL
DEFAULT_RELEASE_URL = os.getenv(
    "DATASET_URL",
    "https://github.com/user/repo/releases/download/v1.0.0/surface_dataset_v1.tar.gz"
)


def create_synthetic_material_sample(sample_id: str, sample_name: str, target_dir: Path, size: tuple = (256, 256)):
    """
    Generate synthetic procedural PBR surface material maps (RGB, Normal, Roughness, Height) at 256x256
    for out-of-the-box offline benchmarking.
    """
    w, h = size
    x = np.linspace(-np.pi * 2, np.pi * 2, w)
    y = np.linspace(-np.pi * 2, np.pi * 2, h)
    xx, yy = np.meshgrid(x, y)

    rgb_sub = target_dir / "rgb"
    nor_sub = target_dir / "ground_truth" / "normals"
    rough_sub = target_dir / "ground_truth" / "roughness"
    height_sub = target_dir / "ground_truth" / "height"

    for d in [rgb_sub, nor_sub, rough_sub, height_sub]:
        d.mkdir(parents=True, exist_ok=True)

    if "brick" in sample_id.lower():
        # Procedural Brick Pattern
        brick_x = (np.floor((xx + np.pi * 2) * 2) % 2).astype(np.float32)
        brick_y = (np.floor((yy + np.pi * 2) * 4) % 2).astype(np.float32)
        pattern = np.sin(xx * 6) * np.cos(yy * 12) * 0.5 + 0.5
        height_map = np.clip(pattern * 0.8 + brick_x * 0.2, 0, 1)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (180 + height_map * 50).clip(0, 255).astype(np.uint8)
        rgb[:, :, 1] = (60 + height_map * 30).clip(0, 255).astype(np.uint8)
        rgb[:, :, 2] = (40 + height_map * 20).clip(0, 255).astype(np.uint8)

        roughness_map = (0.7 - height_map * 0.4).clip(0, 1)

    elif "wood" in sample_id.lower():
        # Procedural Wood Ring Pattern
        r = np.sqrt(xx**2 + yy**2)
        rings = np.sin(r * 8 + np.sin(xx * 2) * 0.5) * 0.5 + 0.5
        height_map = np.clip(rings * 0.6 + 0.2, 0, 1)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = (150 * height_map + 70).clip(0, 255).astype(np.uint8)
        rgb[:, :, 1] = (90 * height_map + 40).clip(0, 255).astype(np.uint8)
        rgb[:, :, 2] = (40 * height_map + 20).clip(0, 255).astype(np.uint8)

        roughness_map = (0.3 + height_map * 0.5).clip(0, 1)

    elif "stone" in sample_id.lower() or "cobble" in sample_id.lower():
        # Procedural Stone / Cobblestone Pattern
        stones = np.sin(xx * 4) * np.sin(yy * 4) + np.cos(xx * 8) * 0.2
        height_map = np.clip((stones - stones.min()) / (stones.max() - stones.min()), 0, 1)

        rgb_gray = (height_map * 180 + 40).clip(0, 255).astype(np.uint8)
        rgb = cv2.merge([rgb_gray, rgb_gray, rgb_gray])

        roughness_map = (0.8 - height_map * 0.3).clip(0, 1)

    else:  # Metal Plate Pattern
        # Procedural Brushed Metal Plate
        grid = (np.sin(xx * 10) > 0.95).astype(np.float32) | (np.sin(yy * 10) > 0.95).astype(np.float32)
        height_map = np.clip(1.0 - grid * 0.6, 0, 1)

        rgb_base = (height_map * 120 + 100).clip(0, 255).astype(np.uint8)
        rgb = cv2.merge([rgb_base, rgb_base + 10, rgb_base + 20])

        roughness_map = (0.2 + grid * 0.6).clip(0, 1)

    # Compute Ground-Truth Sobel Normal Map for Synthetic Sample
    dx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3) * 2.0
    dy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3) * 2.0
    dz = np.ones_like(height_map, dtype=np.float32)

    mag = np.sqrt(dx**2 + dy**2 + dz**2)
    nx, ny, nz = -dx / mag, dy / mag, dz / mag

    norm_r = ((nx * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    norm_g = ((ny * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    norm_b = ((nz * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    normal_rgb = cv2.merge([norm_r, norm_g, norm_b])

    height_u8 = (height_map * 255).clip(0, 255).astype(np.uint8)
    rough_u8 = (roughness_map * 255).clip(0, 255).astype(np.uint8)

    # Save to disk under sample_id
    cv2.imwrite(str(rgb_sub / f"{sample_id}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(nor_sub / f"{sample_id}.png"), cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(rough_sub / f"{sample_id}.png"), rough_u8)
    cv2.imwrite(str(height_sub / f"{sample_id}.png"), height_u8)

    # Also mirror into top-level data/ folder for compatibility
    for dst, src_file in [
        (DATA_DIR / "rgb" / f"{sample_id}.jpg", rgb_sub / f"{sample_id}.jpg"),
        (DATA_DIR / "ground_truth" / "normals" / f"{sample_id}.png", nor_sub / f"{sample_id}.png"),
        (DATA_DIR / "ground_truth" / "roughness" / f"{sample_id}.png", rough_sub / f"{sample_id}.png"),
        (DATA_DIR / "ground_truth" / "height" / f"{sample_id}.png", height_sub / f"{sample_id}.png")
    ]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.imread(str(src_file)))

    print(f"  [Generated Synthetic Sample] {sample_name} ({sample_id})")


def generate_synthetic_dataset_fallback(target_dir: Path):
    """Generate 4 standard synthetic material samples (256x256) when offline or dataset URL fails."""
    print("[Dataset Sync] Generating synthetic sample dataset pairs (256x256)...")
    samples = [
        ("synth_brick_wall", "Red Brick Wall"),
        ("synth_wood_planks", "Wood Grain Planks"),
        ("synth_stone_cobble", "Cobblestone Pavement"),
        ("synth_metal_plate", "Brushed Metal Plate")
    ]
    for s_id, s_name in samples:
        create_synthetic_material_sample(s_id, s_name, target_dir, size=(256, 256))


def package_dataset(source_dir: Path = None, output_archive: Path = None, target_size: tuple = (256, 256)):
    """
    Packaging Script:
    Gather and compress paired training data (RGB, Normal, Roughness, Height maps resized to 256x256)
    into a single 'surface_dataset_v1.tar.gz' file to upload to GitHub Releases.
    """
    if source_dir is None:
        source_dir = DATA_DIR
    if output_archive is None:
        output_archive = BASE_DIR / "surface_dataset_v1.tar.gz"

    rgb_dir = source_dir / "rgb"
    gt_norm_dir = source_dir / "ground_truth" / "normals"
    gt_rough_dir = source_dir / "ground_truth" / "roughness"
    gt_height_dir = source_dir / "ground_truth" / "height"

    if not rgb_dir.exists() or len(list(rgb_dir.glob("*.*"))) == 0:
        generate_synthetic_dataset_fallback(DATASET_DIR)
        source_dir = DATASET_DIR
        rgb_dir = source_dir / "rgb"
        gt_norm_dir = source_dir / "ground_truth" / "normals"
        gt_rough_dir = source_dir / "ground_truth" / "roughness"
        gt_height_dir = source_dir / "ground_truth" / "height"

    # Temporary staging directory for 256x256 resized dataset
    stage_dir = BASE_DIR / "temp_stage_256"
    stage_rgb = stage_dir / "rgb"
    stage_norm = stage_dir / "ground_truth" / "normals"
    stage_rough = stage_dir / "ground_truth" / "roughness"
    stage_height = stage_dir / "ground_truth" / "height"

    for d in [stage_rgb, stage_norm, stage_rough, stage_height]:
        d.mkdir(parents=True, exist_ok=True)

    packed_count = 0
    for rgb_path in sorted(rgb_dir.glob("*.*")):
        if rgb_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        asset_id = rgb_path.stem

        img = cv2.imread(str(rgb_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(stage_rgb / f"{asset_id}.jpg"), img_resized)

        # Normals
        nor_file = gt_norm_dir / f"{asset_id}.png"
        if nor_file.exists():
            nor_img = cv2.imread(str(nor_file))
            if nor_img is not None:
                cv2.imwrite(str(stage_norm / f"{asset_id}.png"), cv2.resize(nor_img, target_size, interpolation=cv2.INTER_AREA))

        # Roughness
        rough_file = gt_rough_dir / f"{asset_id}.png"
        if rough_file.exists():
            rough_img = cv2.imread(str(rough_file), cv2.IMREAD_GRAYSCALE)
            if rough_img is not None:
                cv2.imwrite(str(stage_rough / f"{asset_id}.png"), cv2.resize(rough_img, target_size, interpolation=cv2.INTER_AREA))

        # Height
        height_file = gt_height_dir / f"{asset_id}.png"
        if height_file.exists():
            height_img = cv2.imread(str(height_file), cv2.IMREAD_GRAYSCALE)
            if height_img is not None:
                cv2.imwrite(str(stage_height / f"{asset_id}.png"), cv2.resize(height_img, target_size, interpolation=cv2.INTER_AREA))

        packed_count += 1

    # Create tar.gz archive
    print(f"[Dataset Sync] Compressing {packed_count} resized paired samples into {output_archive.name}...")
    with tarfile.open(output_archive, "w:gz") as tar:
        tar.add(stage_dir, arcname="surface_dataset_v1")

    # Cleanup temporary staging folder
    import shutil
    shutil.rmtree(stage_dir, ignore_errors=True)

    archive_size_mb = output_archive.stat().st_size / (1024 * 1024)
    print(f"[Dataset Sync] Successfully packaged '{output_archive.name}' ({archive_size_mb:.2f} MB)")
    return output_archive


def ensure_dataset_ready(dataset_dir: Path = None, release_url: str = None) -> bool:
    """
    Auto-Downloader:
    Checks if local data exists under 'data/dataset/' or 'data/rgb'.
    If missing, attempts to download 'surface_dataset_v1.tar.gz' from release_url,
    extracts it to dataset_dir, and cleans up temporary archive.
    If download fails or URL is generic/offline, falls back to generating synthetic dataset pairs.
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    if release_url is None:
        release_url = DEFAULT_RELEASE_URL

    rgb_check_1 = dataset_dir / "rgb"
    rgb_check_2 = DATA_DIR / "rgb"

    has_data = (rgb_check_1.exists() and len(list(rgb_check_1.glob("*.*"))) > 0) or \
               (rgb_check_2.exists() and len(list(rgb_check_2.glob("*.*"))) > 0)

    if has_data:
        print("[Dataset Sync] Dataset ready locally.")
        return True

    print("[Dataset Sync] Local dataset missing. Initiating dataset auto-sync...")
    tar_path = DATA_DIR / "surface_dataset_v1.tar.gz"
    download_success = False

    # Attempt download if URL appears valid and non-placeholder
    if release_url and "github.com/user/repo" not in release_url and release_url.startswith("http"):
        try:
            print(f"[Dataset Sync] Downloading surface_dataset_v1.tar.gz from:\n  {release_url}")
            resp = requests.get(release_url, stream=True, timeout=30)
            if resp.status_code == 200:
                with open(tar_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                download_success = True
                print("[Dataset Sync] Download complete. Extracting archive...")
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=DATA_DIR)
                if tar_path.exists():
                    os.remove(tar_path)
                print("[Dataset Sync] Dataset auto-sync successful!")
        except Exception as e:
            print(f"[Dataset Sync] Download failed ({e}). Switching to offline fallback...")

    if not download_success:
        generate_synthetic_dataset_fallback(dataset_dir)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Packager & Auto-Sync Module")
    parser.add_argument("--package", action="store_true", help="Package dataset into surface_dataset_v1.tar.gz")
    parser.add_argument("--ensure", action="store_true", help="Ensure dataset exists locally (download or fallback)")
    parser.add_argument("--url", type=str, default=DEFAULT_RELEASE_URL, help="Configurable GitHub Release URL")
    args = parser.parse_args()

    if args.package:
        package_dataset()
    else:
        ensure_dataset_ready(release_url=args.url)
