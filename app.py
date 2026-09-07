#!/usr/bin/env python3
"""
FastAPI Backend Server for Single-Image Surface Map Estimation and Real-Time Physical Relighting.
Supports Classical DIP (Sobel + 2D DFT + Poisson Integration) and Deep Learning (PyTorch U-Net with Cosine Angular Loss).
"""

import os
import io
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from data_sync import ensure_dataset_ready
from engine import process_surface_maps, image_to_base64, diode_npy_to_rgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RGB_DIR = DATA_DIR / "rgb"
GT_NORMALS_DIR = DATA_DIR / "ground_truth" / "normals"
GT_ROUGH_DIR = DATA_DIR / "ground_truth" / "roughness"
GT_HEIGHT_DIR = DATA_DIR / "ground_truth" / "height"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Single-Image Surface Map Estimation & Relighting API",
    description="Backend API for Classical DIP & PyTorch U-Net Surface Map Estimation, PBR Datasets, and Real-Time Relighting.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.on_event("startup")
async def startup_event():
    """Ensure dataset files exist locally or download/synthesize fallback data on server startup."""
    ensure_dataset_ready()


@app.get("/")
async def read_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "Surface Map Estimation API running."})


@app.get("/api/samples")
async def list_dataset_samples():
    """Return list of available dataset samples (Poly Haven, DIODE, Synthetic)."""
    samples = []
    if not RGB_DIR.exists():
        return {"samples": []}

    for rgb_path in sorted(RGB_DIR.glob("*.*")):
        if rgb_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        
        asset_id = rgb_path.stem
        is_diode = asset_id.startswith("diode_")
        is_synth = asset_id.startswith("synth_")
        
        if is_diode:
            source = "diode"
            clean_name = f"DIODE {asset_id.replace('diode_', '').replace('_', ' ').title()}"
        elif is_synth:
            source = "synthetic"
            clean_name = f"Synthetic {asset_id.replace('synth_', '').replace('_', ' ').title()}"
        else:
            source = "polyhaven"
            clean_name = f"Poly Haven {asset_id.replace('_', ' ').title()}"

        nor_png = GT_NORMALS_DIR / f"{asset_id}.png"
        nor_npy = GT_NORMALS_DIR / f"{asset_id}_normal.npy"
        has_gt = nor_png.exists() or nor_npy.exists()

        samples.append({
            "asset_id": asset_id,
            "name": clean_name,
            "dataset_source": source,
            "rgb_url": f"/data/rgb/{rgb_path.name}",
            "has_gt": has_gt
        })

    return {"samples": samples}


@app.post("/api/generate")
async def generate_maps(
    asset_id: str = Form(None),
    method: str = Form("both"),
    file: UploadFile = File(None)
):
    """
    Generate Normal (DIP & PyTorch U-Net), Roughness (2D DFT), and Height (Poisson Integration) maps.
    Computes evaluation metric scorecards (MAE, % < 11.25°, % < 22.5°, PSNR, SSIM) if GT maps exist.
    """
    rgb_img = None
    gt_normal = None
    gt_roughness = None
    gt_height = None
    gt_mask = None

    if asset_id:
        rgb_path = next(RGB_DIR.glob(f"{asset_id}.*"), None)
        if not rgb_path or not rgb_path.exists():
            raise HTTPException(status_code=404, detail=f"Asset ID '{asset_id}' not found.")

        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            raise HTTPException(status_code=500, detail=f"Failed to read image for asset '{asset_id}'.")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load Ground-Truth maps if available
        nor_npy_path = GT_NORMALS_DIR / f"{asset_id}_normal.npy"
        nor_mask_path = GT_NORMALS_DIR / f"{asset_id}_mask.npy"
        nor_png_path = GT_NORMALS_DIR / f"{asset_id}.png"

        if nor_npy_path.exists():
            gt_normal = np.load(nor_npy_path)
            if nor_mask_path.exists():
                gt_mask = np.load(nor_mask_path)
        elif nor_png_path.exists():
            nor_bgr = cv2.imread(str(nor_png_path))
            if nor_bgr is not None:
                gt_normal = cv2.cvtColor(nor_bgr, cv2.COLOR_BGR2RGB)

        rough_path = GT_ROUGH_DIR / f"{asset_id}.png"
        if rough_path.exists():
            gt_roughness = cv2.imread(str(rough_path), cv2.IMREAD_GRAYSCALE)

        height_path = GT_HEIGHT_DIR / f"{asset_id}.png"
        if height_path.exists():
            gt_height = cv2.imread(str(height_path), cv2.IMREAD_GRAYSCALE)

    elif file:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        rgb_img = np.array(pil_img)
    else:
        raise HTTPException(status_code=400, detail="Must provide asset_id or image file.")

    if rgb_img is None or rgb_img.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image input.")

    # Downscale max dim to 1024 for web performance
    h, w = rgb_img.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if gt_normal is not None and gt_normal.dtype == np.uint8:
            gt_normal = cv2.resize(gt_normal, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if gt_roughness is not None:
            gt_roughness = cv2.resize(gt_roughness, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if gt_height is not None:
            gt_height = cv2.resize(gt_height, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    result = process_surface_maps(
        rgb_image=rgb_img,
        method=method,
        gt_normal=gt_normal,
        gt_roughness=gt_roughness,
        gt_height=gt_height,
        gt_mask=gt_mask
    )

    # Attach visual GT maps as Base64 strings if available
    if gt_normal is not None:
        if gt_normal.dtype == np.uint8:
            result["maps"]["gt_normal"] = image_to_base64(gt_normal, fmt="PNG")
        else:
            gt_norm_rgb = diode_npy_to_rgb(gt_normal, gt_mask)
            result["maps"]["gt_normal"] = image_to_base64(gt_norm_rgb, fmt="PNG")

    if gt_roughness is not None:
        result["maps"]["gt_roughness"] = image_to_base64(gt_roughness, fmt="PNG")
    if gt_height is not None:
        result["maps"]["gt_height"] = image_to_base64(gt_height, fmt="PNG")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
