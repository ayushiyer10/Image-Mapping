#!/usr/bin/env python3
"""
Manifest Builder (build_manifest.py)
Scans existing local dataset files under data/rgb/, data/ground_truth/, and extracted DIODE data under data/val/.
Ingests DIODE validation subset into data/rgb/ and data/ground_truth/normals/ for unified benchmarking.
Does NOT perform network downloads.
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from dip_engine import diode_npy_to_rgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VAL_DIR = DATA_DIR / "val"
RGB_DIR = DATA_DIR / "rgb"
GT_NORMALS_DIR = DATA_DIR / "ground_truth" / "normals"
GT_ROUGH_DIR = DATA_DIR / "ground_truth" / "roughness"
GT_HEIGHT_DIR = DATA_DIR / "ground_truth" / "height"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def sync_diode_samples(max_diode_samples: int = 150):
    """Scan data/val for DIODE dataset triplets and sync them into data/rgb & data/ground_truth/normals."""
    print(f"\n[Manifest Debug Log] Scanning DIODE raw directory: '{VAL_DIR.resolve()}'")
    if not VAL_DIR.exists():
        print(f"[Manifest Debug Log] Directory '{VAL_DIR}' does not exist! Printing root data/ directory tree:")
        for root, dirs, files in os.walk(DATA_DIR):
            print(f"  {root}: {files[:10]}")
        return

    all_png_files = sorted([p for p in VAL_DIR.rglob("*.png") if not p.name.endswith("_depth.png") and not p.name.endswith("_normal.png")])
    total_png = len(all_png_files)
    print(f"[Manifest Debug Log] Total RGB *.png files found under data/val: {total_png}")

    matching_normal_npy = 0
    matching_mask_npy = 0
    surviving_triplets = []

    for rgb_path in all_png_files:
        parent = rgb_path.parent
        stem = rgb_path.stem  # e.g. 00024_00201_outdoor_000_000
        
        nor_npy = parent / f"{stem}_normal.npy"
        mask_npy = parent / f"{stem}_normal_mask.npy"

        has_nor = nor_npy.exists()
        has_mask = mask_npy.exists()

        if has_nor:
            matching_normal_npy += 1
        if has_mask:
            matching_mask_npy += 1

        if has_nor:
            surviving_triplets.append({
                "stem": stem,
                "rgb_path": rgb_path,
                "nor_npy": nor_npy,
                "mask_npy": mask_npy if has_mask else None
            })

    print(f"[Manifest Debug Log] Matching *_normal.npy files found: {matching_normal_npy}")
    print(f"[Manifest Debug Log] Matching *_normal_mask.npy files found: {matching_mask_npy}")
    print(f"[Manifest Debug Log] Complete DIODE sample triplets surviving filtering: {len(surviving_triplets)}")

    RGB_DIR.mkdir(parents=True, exist_ok=True)
    GT_NORMALS_DIR.mkdir(parents=True, exist_ok=True)

    synced_count = 0
    for item in surviving_triplets[:max_diode_samples]:
        stem = item["stem"]
        diode_id = f"diode_{stem}"

        dst_rgb = RGB_DIR / f"{diode_id}.png"
        dst_nor_npy = GT_NORMALS_DIR / f"{diode_id}_normal.npy"
        dst_nor_mask = GT_NORMALS_DIR / f"{diode_id}_mask.npy"
        dst_nor_png = GT_NORMALS_DIR / f"{diode_id}.png"

        if not (dst_rgb.exists() and dst_nor_png.exists()):
            # Copy RGB PNG
            shutil.copy2(item["rgb_path"], dst_rgb)
            # Copy Normal NPY
            shutil.copy2(item["nor_npy"], dst_nor_npy)

            mask_np = None
            if item["mask_npy"] is not None and item["mask_npy"].exists():
                shutil.copy2(item["mask_npy"], dst_nor_mask)
                try:
                    mask_np = np.load(dst_nor_mask)
                except Exception:
                    pass

            # Render OpenGL Normal RGB PNG
            try:
                nor_np = np.load(dst_nor_npy)
                nor_rgb = diode_npy_to_rgb(nor_np, mask_np)
                cv2.imwrite(str(dst_nor_png), cv2.cvtColor(nor_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"[Manifest Error] Error rendering DIODE normal PNG for {diode_id}: {e}")

        synced_count += 1

    print(f"[Manifest Debug Log] Synced {synced_count} DIODE validation samples into data/rgb/ and data/ground_truth/normals/")


def build_manifest(max_diode_samples: int = 150) -> list:
    sync_diode_samples(max_diode_samples=max_diode_samples)

    if not RGB_DIR.exists():
        print(f"[Manifest] RGB directory '{RGB_DIR}' not found.")
        return []

    manifest = []
    rgb_files = sorted([p for p in RGB_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    counts = {"polyhaven": 0, "diode": 0, "synthetic": 0}

    for rgb_path in rgb_files:
        asset_id = rgb_path.stem
        rel_rgb_path = str(rgb_path.relative_to(BASE_DIR)).replace("\\", "/")

        is_diode = asset_id.startswith("diode_")
        is_synth = asset_id.startswith("synth_")

        if is_diode:
            source = "diode"
            scene_type = "indoor" if "indoor" in asset_id.lower() else "outdoor"
        elif is_synth:
            source = "synthetic"
            scene_type = "synth"
        else:
            source = "polyhaven"
            if any(k in asset_id.lower() for k in ["tile", "wood", "veneer", "floor", "plank", "marble"]):
                scene_type = "indoor"
            else:
                scene_type = "outdoor"

        counts[source] = counts.get(source, 0) + 1

        # Check Ground-Truth map paths
        gt_nor_png = GT_NORMALS_DIR / f"{asset_id}.png"
        gt_nor_npy = GT_NORMALS_DIR / f"{asset_id}_normal.npy"
        gt_nor_mask = GT_NORMALS_DIR / f"{asset_id}_mask.npy"

        gt_normal_path = None
        gt_mask_path = None

        if gt_nor_npy.exists():
            gt_normal_path = str(gt_nor_npy.relative_to(BASE_DIR)).replace("\\", "/")
            if gt_nor_mask.exists():
                gt_mask_path = str(gt_nor_mask.relative_to(BASE_DIR)).replace("\\", "/")
        elif gt_nor_png.exists():
            gt_normal_path = str(gt_nor_png.relative_to(BASE_DIR)).replace("\\", "/")

        gt_rough_png = GT_ROUGH_DIR / f"{asset_id}.png"
        gt_roughness_path = str(gt_rough_png.relative_to(BASE_DIR)).replace("\\", "/") if gt_rough_png.exists() else None

        gt_height_png = GT_HEIGHT_DIR / f"{asset_id}.png"
        gt_height_path = str(gt_height_png.relative_to(BASE_DIR)).replace("\\", "/") if gt_height_png.exists() else None

        item = {
            "asset_id": asset_id,
            "source": source,
            "scene_type": scene_type,
            "rgb_path": rel_rgb_path,
            "gt_normal_path": gt_normal_path,
            "gt_mask_path": gt_mask_path,
            "gt_roughness_path": gt_roughness_path,
            "gt_height_path": gt_height_path,
            "has_gt": gt_normal_path is not None
        }
        manifest.append(item)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Manifest Final Summary]")
    print(f"  - Poly Haven : {counts['polyhaven']} samples")
    print(f"  - DIODE      : {counts['diode']} samples")
    print(f"  - Synthetic  : {counts['synthetic']} samples")
    print(f"  - Total      : {len(manifest)} samples written to '{MANIFEST_PATH.name}'.\n")

    return manifest


if __name__ == "__main__":
    build_manifest()
