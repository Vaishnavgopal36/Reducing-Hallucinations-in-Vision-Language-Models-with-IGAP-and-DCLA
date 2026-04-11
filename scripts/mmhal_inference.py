"""
scripts/mmhal_inference.py
==========================
MMHal-Bench inference script for LLaVA-1.5 7B + IGAP + DCLA.

Runs four benchmarking configurations and saves results to ``results/``:

+--------------------------------+------------------------------------+
| Config name                    | Output file                        |
+================================+====================================+
| Baseline                       | results/response_baseline.json     |
+--------------------------------+------------------------------------+
| Just IGAP                      | results/response_just_igap.json    |
+--------------------------------+------------------------------------+
| Just DCLA                      | results/response_just_dcla.json    |
+--------------------------------+------------------------------------+
| IGAP-DCLA (full system)        | results/response_igap_dcla.json    |
+--------------------------------+------------------------------------+

Usage
-----
Run from the repository root so that the ``src`` package is on the path::

    PYTHONPATH=. python scripts/mmhal_inference.py

Requirements
------------
See ``requirements.txt``.  A CUDA-capable GPU with ≥16 GB VRAM is recommended
(tested on NVIDIA Tesla T4).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path for ``src.*`` imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.mmhal_loader import manual_load_mmhal_bench
from src.decoding.dcla_decode import dynamic_decode_one_sample
from src.model.igap_patch import (
    _igap_toggle,
    apply_igap_to_llava,
    get_llama_layers,
    igap_debug,
)
from src.model.vision_utils import get_image_token_range_hf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
RESULTS_DIR: str = "results"

# Benchmark run configuration
# Each entry controls which components are active for a given run.
BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "baseline",
        "file": os.path.join(RESULTS_DIR, "response_baseline.json"),
        "use_igap": False,
        "mode": "standard",
    },
    {
        "name": "just_igap",
        "file": os.path.join(RESULTS_DIR, "response_just_igap.json"),
        "use_igap": True,
        "mode": "standard",
    },
    {
        "name": "just_dcla",
        "file": os.path.join(RESULTS_DIR, "response_just_dcla.json"),
        "use_igap": False,
        "mode": "dcla",
    },
    {
        "name": "igap_dcla",
        "file": os.path.join(RESULTS_DIR, "response_igap_dcla.json"),
        "use_igap": True,
        "mode": "dcla",
    },
]

# Shared generation hyperparameters
MAX_NEW_TOKENS: int = 128
JS_THRESHOLD: float = 0.08
ALPHA1: float = 4.0
ALPHA2: float = 1.0
LAM: float = 0.2
TOKEN_ALPHA: float = 0.06

# IGAP hyperparameters (paper Table 6, LLaVA-1.5 7B)
KEEP_HEAD_RATIO: float = 0.95
SUPPRESSION_ALPHA: float = 0.08


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load model ----------------------------------------------------------
    print(f"Loading {MODEL_ID}…")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Fix processor config for transformers >= 4.46
    if not getattr(processor, "patch_size", None):
        processor.patch_size = model.config.vision_config.patch_size
    if not getattr(processor, "vision_feature_select_strategy", None):
        processor.vision_feature_select_strategy = "default"

    lm_device: torch.device = next(model.language_model.parameters()).device
    print(f"✅ Model loaded on {lm_device}")

    # ---- Load dataset --------------------------------------------------------
    dataset = manual_load_mmhal_bench()

    # ---- Discover image token range from a probe sample ---------------------
    probe_inputs = processor(
        text="USER: <image>\nDescribe.\nASSISTANT:",
        images=dataset[0]["image"].convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    for k, v in probe_inputs.items():
        if torch.is_tensor(v):
            probe_inputs[k] = v.to(
                lm_device,
                dtype=(torch.float16 if k == "pixel_values" else None),
            )

    rng = get_image_token_range_hf(model, probe_inputs)
    assert rng is not None, "Could not find image token range — check model/processor."
    img_start_idx, img_end_idx = rng
    print(f" Image token range: [{img_start_idx}, {img_end_idx})")

    # ---- Apply IGAP patch to all layers -------------------------------------
    apply_igap_to_llava(
        model,
        start_layer=0,
        end_layer=len(get_llama_layers(model)),
        img_start_idx=img_start_idx,
        img_end_idx=img_end_idx,
        keep_head_ratio=KEEP_HEAD_RATIO,
        suppression_alpha=SUPPRESSION_ALPHA,
        use_igap_img=True,          # will be toggled per-benchmark below
    )

    # ---- Benchmarking loop --------------------------------------------------
    for bench in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {bench['name'].upper()}")
        print(f"{'='*60}")

        # Configure IGAP enable/disable flag on every attention layer
        for layer in get_llama_layers(model):
            layer.self_attn.use_igap_img = bench["use_igap"]

        # Reset debug counters
        igap_debug.update({
            "forward_calls": 0,
            "igap_active_calls": 0,
            "q_len1_calls": 0,
            "avg_suppressed_fraction_sum": 0.0,
        })

        results: List[Dict[str, Any]] = []

        for item in tqdm(dataset, desc=f"  [{bench['name']}]"):
            question: str = item["question"]
            image = item["image"].convert("RGB")
            prompt: str = f"USER: <image>\n{question}\nASSISTANT:"

            if bench["mode"] == "standard":
                # ----------- Standard model.generate (Baseline / Just IGAP) ----------
                _igap_toggle["active"] = bench["use_igap"]
                inputs = processor(
                    text=prompt, images=image, return_tensors="pt", padding=True
                )
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(
                            lm_device,
                            dtype=(torch.float16 if k == "pixel_values" else None),
                        )
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        repetition_penalty=1.1,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                _igap_toggle["active"] = False
                ans: str = processor.batch_decode(
                    out_ids, skip_special_tokens=True
                )[0].split("ASSISTANT:")[-1].strip()
                trace: list = []

            else:
                # ----------- DCLA dual-pass decoding (Just DCLA / IGAP-DCLA) ----------
                ans, trace = dynamic_decode_one_sample(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    image=image,
                    img_start=img_start_idx,
                    img_end=img_end_idx,
                    max_new_tokens=MAX_NEW_TOKENS,
                    js_threshold=JS_THRESHOLD,
                    alpha1=ALPHA1,
                    alpha2=ALPHA2,
                    lam=LAM,
                    token_alpha=TOKEN_ALPHA,
                )

            results.append({
                "question_type":  item.get("question_type", ""),
                "question_topic": item.get("question_topic", ""),
                "image_id":       item.get("image_id", ""),
                "image_src":      item.get("image_src", ""),
                "image_content":  item.get("image_content", []),
                "question":       question,
                "gt_answer":      item.get("gt_answer", ""),
                "model_answer":   ans,
                "decode_trace":   trace[:25],
            })

        with open(bench["file"], "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f" Saved {len(results)} results → {bench['file']}")

        if igap_debug["igap_active_calls"] > 0:
            avg_supp = (
                igap_debug["avg_suppressed_fraction_sum"]
                / igap_debug["igap_active_calls"]
            )
            print(f"   IGAP avg suppressed head fraction: {avg_supp:.3f}")

    print("\nALL BENCHMARKS COMPLETE.")


if __name__ == "__main__":
    main()
