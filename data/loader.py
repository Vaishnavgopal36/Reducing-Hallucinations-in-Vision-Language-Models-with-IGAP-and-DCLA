"""MMHal-Bench dataset loader extracted from the research notebook."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image

MMHAL_URL: str = (
    "https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/resolve/main/test_data.zip"
)

DEFAULT_DATA_DIR: Path = Path("data/mmhal_data")
DEFAULT_ZIP_PATH: Path = Path("data/test_data.zip")


def _is_ready(data_dir: Path) -> bool:
    json_path = data_dir / "response_template.json"
    img_dir = data_dir / "images"
    if not json_path.exists() or not img_dir.exists():
        return False
    return any(
        image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        for image_path in img_dir.iterdir()
    )


def _download_zip(zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(MMHAL_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with zip_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def _extract_zip(zip_path: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)


def manual_load_mmhal_bench(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    zip_path: str | Path = DEFAULT_ZIP_PATH,
) -> List[Dict[str, Any]]:
    """Load MMHal-Bench records with PIL images."""
    data_dir = Path(data_dir)
    zip_path = Path(zip_path)

    if not _is_ready(data_dir):
        print("MMHal-Bench data not found locally. Preparing dataset...")
        if not zip_path.exists():
            print(f"Downloading MMHal-Bench from {MMHAL_URL}")
            _download_zip(zip_path)
        print("Extracting MMHal-Bench archive")
        _extract_zip(zip_path, data_dir)
    else:
        print("MMHal-Bench data already present. Skipping download.")

    json_path = data_dir / "response_template.json"
    image_dir = data_dir / "images"
    if not json_path.exists():
        raise FileNotFoundError(f"MMHal JSON not found: {json_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"MMHal image directory not found: {image_dir}")

    with json_path.open("r", encoding="utf-8") as file_obj:
        rows: List[Dict[str, Any]] = json.load(file_obj)

    records: List[Dict[str, Any]] = []
    missing = 0
    for row in rows:
        image_src = str(row.get("image_src", ""))
        image_path = image_dir / Path(image_src).name
        try:
            with Image.open(image_path) as img:
                record = dict(row)
                record["image"] = img.convert("RGB")
                records.append(record)
        except Exception:
            missing += 1

    if missing:
        print(f"Warning: skipped {missing} samples with missing/unreadable images.")
    print(f"MMHal-Bench loaded: {len(records)} samples")
    return records
