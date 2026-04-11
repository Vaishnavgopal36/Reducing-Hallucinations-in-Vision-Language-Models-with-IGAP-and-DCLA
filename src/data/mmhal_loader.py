"""
src/data/mmhal_loader.py
========================
MMHal-Bench dataset loader.

Downloads the test data zip from Hugging Face, extracts it, and returns
a list of records with PIL Images attached.  If the data is already present
on disk, the download/extraction step is skipped.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Dict, List

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Paths and remote URL
# ---------------------------------------------------------------------------
MMHAL_DIR: str = "mmhal_data"
ZIP_PATH: str = "test_data.zip"
JSON_PATH: str = os.path.join(MMHAL_DIR, "response_template.json")
IMG_DIR: str = os.path.join(MMHAL_DIR, "images")

MMHAL_URL: str = (
    "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/"
    "resolve/main/test_data.zip"
)


def _mmhal_data_ready() -> bool:
    """Return ``True`` if the JSON and at least one image file are present."""
    if not os.path.exists(JSON_PATH):
        return False
    if not os.path.exists(IMG_DIR):
        return False
    try:
        img_files = [
            f
            for f in os.listdir(IMG_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        return len(img_files) > 0
    except OSError:
        return False


def _download_mmhal_zip() -> None:
    """Stream-download the MMHal-Bench zip to ``ZIP_PATH``."""
    print("⬇️  Downloading MMHal-Bench zip…")
    response = requests.get(MMHAL_URL, stream=True)
    response.raise_for_status()
    with open(ZIP_PATH, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    print(f"✅ Downloaded → {ZIP_PATH}")


def _extract_mmhal_zip() -> None:
    """Extract ``ZIP_PATH`` into ``MMHAL_DIR``."""
    print("📦 Extracting MMHal-Bench…")
    os.makedirs(MMHAL_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(MMHAL_DIR)
    print(f"✅ Extracted into: {MMHAL_DIR}")


def manual_load_mmhal_bench() -> List[Dict[str, Any]]:
    """Load the MMHal-Bench test set, downloading if necessary.

    For each record in ``response_template.json``, the function attempts to
    open the corresponding local image.  Records whose image cannot be read
    are silently dropped (a warning count is printed at the end).

    Returns
    -------
    List[Dict[str, Any]]
        Each dict mirrors the JSON record and additionally contains an
        ``"image"`` key holding the loaded RGB ``PIL.Image.Image``.

    Raises
    ------
    requests.HTTPError
        If the download fails.
    zipfile.BadZipFile
        If the downloaded archive is corrupt.
    """
    if not _mmhal_data_ready():
        print("⚠️  MMHal data not found locally. Preparing dataset…")
        if not os.path.exists(ZIP_PATH):
            _download_mmhal_zip()
        _extract_mmhal_zip()
    else:
        print("✅ MMHal data already present. Skipping download.")

    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        data: List[Dict[str, Any]] = json.load(fh)

    formatted: List[Dict[str, Any]] = []
    missing_imgs: int = 0

    for item in data:
        filename: str = os.path.basename(item["image_src"])
        local_img_path: str = os.path.join(IMG_DIR, filename)
        try:
            img = Image.open(local_img_path).convert("RGB")
            item["image"] = img
            formatted.append(item)
        except Exception:
            missing_imgs += 1

    if missing_imgs > 0:
        print(f"⚠️  Missing / unreadable images: {missing_imgs}")

    print(f"✅ MMHal samples loaded: {len(formatted)}")
    return formatted
