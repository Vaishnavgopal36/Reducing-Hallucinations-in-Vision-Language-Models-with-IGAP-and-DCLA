"""
src/data/mmhal_loader.py
========================
MMHal-Bench dataset loader.

Downloads and extracts the benchmark zip from Hugging Face if not already present,
then returns a list of records with PIL Images attached under the "image" key.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Dict, List

import requests
from PIL import Image

MMHAL_DIR: str = "mmhal_data"
ZIP_PATH: str = "test_data.zip"
JSON_PATH: str = os.path.join(MMHAL_DIR, "response_template.json")
IMG_DIR: str = os.path.join(MMHAL_DIR, "images")
MMHAL_URL: str = (
    "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main/test_data.zip"
)


def _mmhal_data_ready() -> bool:
    if not os.path.exists(JSON_PATH) or not os.path.exists(IMG_DIR):
        return False
    try:
        return any(f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for f in os.listdir(IMG_DIR))
    except OSError:
        return False


def _download_mmhal_zip() -> None:
    print("Downloading MMHal-Bench zip...")
    response = requests.get(MMHAL_URL, stream=True)
    response.raise_for_status()
    with open(ZIP_PATH, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)


def _extract_mmhal_zip() -> None:
    print("Extracting MMHal-Bench...")
    os.makedirs(MMHAL_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(MMHAL_DIR)


def manual_load_mmhal_bench() -> List[Dict[str, Any]]:
    """Load MMHal-Bench, downloading if necessary.

    Returns a list of dicts mirroring the JSON records, each with an added
    "image" key containing the loaded RGB PIL.Image.
    """
    if not _mmhal_data_ready():
        print("MMHal data not found locally. Preparing dataset...")
        if not os.path.exists(ZIP_PATH):
            _download_mmhal_zip()
        _extract_mmhal_zip()
    else:
        print("MMHal data already present.")

    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        data: List[Dict[str, Any]] = json.load(fh)

    formatted: List[Dict[str, Any]] = []
    missing_imgs: int = 0

    for item in data:
        local_img_path = os.path.join(IMG_DIR, os.path.basename(item["image_src"]))
        try:
            item["image"] = Image.open(local_img_path).convert("RGB")
            formatted.append(item)
        except Exception:
            missing_imgs += 1

    if missing_imgs > 0:
        print(f"Warning: {missing_imgs} image(s) missing or unreadable.")

    print(f"MMHal samples loaded: {len(formatted)}")
    return formatted
