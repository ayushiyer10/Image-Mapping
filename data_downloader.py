#!/usr/bin/env python3
"""
Dual Dataset Downloader Script (Poly Haven PBR & DIODE Dataset)
Ingests textures from Poly Haven API and DIODE validation dataset with automated disk usage tracking (Strict ~1.5 GB limit).
"""

import os
import sys
import tarfile
import argparse
import requests
import numpy as np
import cv2
from pathlib import Path
from dip_engine import diode_npy_to_rgb

POLYHAVEN_API_ASSETS = "https://api.polyhaven.com/assets?type=textures"
POLYHAVEN_API_FILES = "https://api.polyhaven.com/files/{asset_id}"
DIODE_VAL_TAR_URL = "http://diode-dataset.s3.amazonaws.com/val.tar.gz"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RGB_DIR = DATA_DIR / "rgb"
GT_NORMALS_DIR = DATA_DIR / "ground_truth" / "normals"
GT_ROUGH_DIR = DATA_DIR / "ground_truth" / "roughness"
GT_HEIGHT_DIR = DATA_DIR / "ground_truth" / "height"


def setup_directories():
    for d in [RGB_DIR, GT_NORMALS_DIR, GT_ROUGH_DIR, GT_HEIGHT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_total_data_bytes():
    """Calculate cumulative disk usage of data/ folder in bytes."""
    total = 0
    if not DATA_DIR.exists():
        return 0
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def fetch_polyhaven_assets():
    """Fetch Poly Haven texture asset list."""
    try:
        resp = requests.get(POLYHAVEN_API_ASSETS, timeout=15)
        resp.raise_for_status()
        return list(resp.json().keys())
    except Exception as e:
        print(f"Error fetching Poly Haven assets: {e}")
        return []


def find_map_url(files_tree, map_keys, preferred_res=("1k", "2k"), preferred_exts=("png", "jpg")):
    for key in map_keys:
        if key not in files_tree or not isinstance(files_tree[key], dict):
            continue
        map_node = files_tree[key]

        res_found = None
        for res in preferred_res:
            if res in map_node:
                res_found = map_node[res]
                break
        if not res_found:
            first_res_key = next(iter(map_node.keys()), None)
            if first_res_key and isinstance(map_node[first_res_key], dict):
                res_found = map_node[first_res_key]

        if not res_found or not isinstance(res_found, dict):
            continue

        for ext in preferred_exts:
            if ext in res_found and isinstance(res_found[ext], dict) and "url" in res_found[ext]:
                return res_found[ext]["url"]

        for ext_key, ext_val in res_found.items():
            if isinstance(ext_val, dict) and "url" in ext_val:
                return ext_val["url"]

    return None


def download_file(url, target_path):
    try:
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return os.path.getsize(target_path)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return os.path.getsize(target_path)
    except Exception as e:
        print(f"\nFailed to download {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return 0


def download_polyhaven_batch(limit_bytes, max_assets=None):
    """Download Poly Haven PBR textures continuously until limit is met."""
    asset_ids = fetch_polyhaven_assets()
    if not asset_ids:
        return 0

    downloaded_bytes = 0
    count = 0

    for idx, asset_id in enumerate(asset_ids):
        curr_total = get_total_data_bytes()
        if curr_total >= limit_bytes:
            print(f"[Poly Haven] Reached disk usage cap of {limit_bytes / (1024**3):.2f} GB.")
            break
        if max_assets is not None and count >= max_assets:
            break

        rgb_file = RGB_DIR / f"{asset_id}.jpg"
        nor_file = GT_NORMALS_DIR / f"{asset_id}.png"
        rough_file = GT_ROUGH_DIR / f"{asset_id}.png"
        height_file = GT_HEIGHT_DIR / f"{asset_id}.png"

        if rgb_file.exists() and nor_file.exists() and rough_file.exists() and height_file.exists():
            count += 1
            continue

        try:
            resp = requests.get(POLYHAVEN_API_FILES.format(asset_id=asset_id), timeout=10)
            if resp.status_code != 200:
                continue
            files_tree = resp.json()
        except Exception:
            continue

        diff_url = find_map_url(files_tree, ['Diffuse', 'diffuse', 'diff', 'col'], preferred_res=['1k', '2k'], preferred_exts=['jpg', 'png'])
        nor_url = find_map_url(files_tree, ['nor_gl', 'nor', 'normal_gl'], preferred_res=['1k', '2k'], preferred_exts=['png', 'jpg'])
        rough_url = find_map_url(files_tree, ['Rough', 'rough', 'roughness'], preferred_res=['1k', '2k'], preferred_exts=['png', 'jpg'])
        disp_url = find_map_url(files_tree, ['Displacement', 'disp', 'height'], preferred_res=['1k', '2k'], preferred_exts=['png', 'jpg'])

        if not (diff_url and nor_url and rough_url and disp_url):
            continue

        b1 = download_file(diff_url, rgb_file)
        b2 = download_file(nor_url, nor_file)
        b3 = download_file(rough_url, rough_file)
        b4 = download_file(disp_url, height_file)

        asset_b = b1 + b2 + b3 + b4
        if asset_b > 0:
            downloaded_bytes += asset_b
            count += 1
            print(f"[OK] ({count}) Downloaded Poly Haven [{asset_id}] (Total used: {curr_total / (1024**2):.1f} MB)")

    return downloaded_bytes


def download_diode_samples(limit_bytes, max_samples=None):
    """
    Download and stream validation samples from DIODE dataset until cap.
    """
    print("Ingesting DIODE Dataset validation samples...")
    try:
        response = requests.get(DIODE_VAL_TAR_URL, stream=True, timeout=30)
        response.raise_for_status()

        tar = tarfile.open(fileobj=response.raw, mode='r|gz')

        pending = {}
        extracted_count = 0

        for member in tar:
            if get_total_data_bytes() >= limit_bytes:
                print(f"[DIODE] Disk limit reached ({limit_bytes / (1024**3):.2f} GB). Stopping DIODE ingestion.")
                break
            if max_samples is not None and extracted_count >= max_samples:
                break

            if not member.isfile():
                continue

            name = member.name
            
            if name.endswith('.png') and not name.endswith('_normal.png') and not name.endswith('_depth.png'):
                base = Path(name).stem
                if base not in pending:
                    pending[base] = {}
                f = tar.extractfile(member)
                if f:
                    pending[base]['rgb'] = f.read()

            elif name.endswith('_normal.npy'):
                base = Path(name[:-len('_normal.npy')]).stem
                if base not in pending:
                    pending[base] = {}
                f = tar.extractfile(member)
                if f:
                    pending[base]['normal'] = f.read()

            elif name.endswith('_normal_mask.npy'):
                base = Path(name[:-len('_normal_mask.npy')]).stem
                if base not in pending:
                    pending[base] = {}
                f = tar.extractfile(member)
                if f:
                    pending[base]['mask'] = f.read()

            to_remove = []
            for base, item in pending.items():
                if 'rgb' in item and 'normal' in item:
                    diode_id = f"diode_{base}"
                    rgb_path = RGB_DIR / f"{diode_id}.png"
                    nor_npy_path = GT_NORMALS_DIR / f"{diode_id}_normal.npy"
                    nor_mask_path = GT_NORMALS_DIR / f"{diode_id}_mask.npy"
                    nor_png_path = GT_NORMALS_DIR / f"{diode_id}.png"

                    with open(rgb_path, 'wb') as f_out:
                        f_out.write(item['rgb'])

                    with open(nor_npy_path, 'wb') as f_out:
                        f_out.write(item['normal'])

                    mask_npy = None
                    if 'mask' in item:
                        with open(nor_mask_path, 'wb') as f_out:
                            f_out.write(item['mask'])
                        try:
                            mask_npy = np.load(nor_mask_path)
                        except Exception:
                            pass

                    try:
                        normal_npy = np.load(nor_npy_path)
                        rgb_normal = diode_npy_to_rgb(normal_npy, mask_npy)
                        cv2.imwrite(str(nor_png_path), cv2.cvtColor(rgb_normal, cv2.COLOR_RGB2BGR))
                    except Exception as e:
                        print(f"Error rendering DIODE normal PNG for {diode_id}: {e}")

                    extracted_count += 1
                    curr_mb = get_total_data_bytes() / (1024**2)
                    print(f"[OK] Downloaded DIODE Sample [{diode_id}] ({extracted_count}) | Total used: {curr_mb:.1f} MB")
                    to_remove.append(base)

                    if max_samples is not None and extracted_count >= max_samples:
                        break

            for base in to_remove:
                del pending[base]

        tar.close()
    except Exception as e:
        print(f"DIODE dataset ingestion note: {e}")


def download_dataset(limit_gb=1.5, max_polyhaven=None, max_diode=None):
    setup_directories()
    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)

    print(f"Starting Dataset Ingestion (Strict Limit: {limit_gb:.2f} GB)...")
    print(f"Current Disk Usage: {get_total_data_bytes() / (1024**2):.2f} MB")

    # 1. Poly Haven Ingestion (allocate half the budget or until cap)
    print("\n--- Ingesting Poly Haven PBR Materials ---")
    download_polyhaven_batch(limit_bytes=int(limit_bytes * 0.6), max_assets=max_polyhaven)

    # 2. DIODE Ingestion (fill remainder up to full limit)
    print("\n--- Ingesting DIODE Dataset Validation Subset ---")
    download_diode_samples(limit_bytes=limit_bytes, max_samples=max_diode)

    final_size_gb = get_total_data_bytes() / (1024**3)
    print(f"\nIngestion Complete! Total Disk Usage: {final_size_gb:.3f} GB / {limit_gb:.2f} GB cap.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Poly Haven and DIODE dataset assets.")
    parser.add_argument("--limit-gb", type=float, default=1.5, help="Maximum disk limit in GB (default: 1.5)")
    parser.add_argument("--polyhaven", type=int, default=None, help="Optional max Poly Haven assets limit")
    parser.add_argument("--diode", type=int, default=None, help="Optional max DIODE samples limit")
    args = parser.parse_args()

    download_dataset(limit_gb=args.limit_gb, max_polyhaven=args.polyhaven, max_diode=args.diode)