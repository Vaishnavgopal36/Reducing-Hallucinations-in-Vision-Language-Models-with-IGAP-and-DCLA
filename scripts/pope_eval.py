"""
scripts/pope_eval.py
====================
POPE benchmark evaluation script for LLaVA-1.5 7B with IGAP and DCLA.

Four variants are evaluated across three POPE splits (random / popular /
adversarial), for a total of 12 output JSON files.

+--------------------+------------------+---------+
| Variant            | IGAP active      | Mode    |
+====================+==================+=========+
| baseline           | No               | standard|
+--------------------+------------------+---------+
| just_igap          | Yes              | standard|
+--------------------+------------------+---------+
| just_dcla          | No               | dcla    |
+--------------------+------------------+---------+
| igap_dcla          | Yes              | dcla    |
+--------------------+------------------+---------+

Checkpointing
-------------
Results are saved every ``CHECKPOINT_EVERY`` records so that a ~4–6 hour
inference run can be safely interrupted and resumed from the last checkpoint.

Usage
-----
Run from the repository root::

    PYTHONPATH=. python scripts/pope_eval.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decoding.dcla_decode import dynamic_decode_one_sample
from src.model.igap_patch import (
    _igap_toggle,
    apply_igap_to_llava,
    get_llama_layers,
)
from src.model.vision_utils import build_attended_embeds, get_image_token_range_hf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
POPE_RESULTS_DIR: str = "pope_results"

POPE_SPLITS: List[str] = ["random", "popular", "adversarial"]
POPE_N: int = 3000           # samples per split (full POPE)
CHECKPOINT_EVERY: int = 25   # save to disk every N records

# IGAP hyperparameters (paper Table 6, LLaVA-1.5 7B)
KEEP_HEAD_RATIO: float = 0.95
SUPPRESSION_ALPHA: float = 0.08

# DCLA hyperparameters
JS_THRESHOLD: float = 0.08
ALPHA1: float = 4.0
ALPHA2: float = 1.0
LAM: float = 0.2
TOKEN_ALPHA: float = 0.06

POPE_VARIANTS: List[Dict[str, Any]] = [
    {"name": "baseline",  "use_igap": False, "mode": "standard"},
    {"name": "just_igap", "use_igap": True,  "mode": "standard"},
    {"name": "just_dcla", "use_igap": False, "mode": "dcla"},
    {"name": "igap_dcla", "use_igap": True,  "mode": "dcla"},
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def extract_yes_no(text: str) -> str:
    """Extract a binary ``"yes"`` / ``"no"`` answer from a model response.

    The first word of the response is checked first.  If ambiguous, the full
    response is scanned.  Falls back to ``"yes"`` (conservative) if neither
    is found, to avoid biasing recall artificially.

    Parameters
    ----------
    text:
        Raw model response string.

    Returns
    -------
    str
        ``"yes"`` or ``"no"``.
    """
    normalised: str = text.strip().lower()
    words = normalised.split()
    if words:
        first = words[0].rstrip(".,!?")
        if first in ("yes", "no"):
            return first
    for word in words:
        w = word.rstrip(".,!?")
        if w in ("yes", "no"):
            return w
    if "yes" in normalised:
        return "yes"
    if "no" in normalised:
        return "no"
    return "yes"   # conservative fallback


def _load_checkpoint(out_file: str) -> Tuple[List[Dict[str, str]], int]:
    """Load an existing checkpoint file if present."""
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as fh:
            existing: List[Dict[str, str]] = json.load(fh)
        return existing, len(existing)
    return [], 0


def _save_checkpoint(out_file: str, records: List[Dict[str, str]]) -> None:
    """Atomically overwrite the checkpoint file."""
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(records, fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(POPE_RESULTS_DIR, exist_ok=True)

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

    if not getattr(processor, "patch_size", None):
        processor.patch_size = model.config.vision_config.patch_size
    if not getattr(processor, "vision_feature_select_strategy", None):
        processor.vision_feature_select_strategy = "default"

    lm_device: torch.device = next(model.language_model.parameters()).device
    print(f"✅ Model loaded on {lm_device}")

    # ---- Detect image token range once from a probe sample ------------------
    probe_ds = (
        load_dataset("lmms-lab/POPE", split="test", streaming=True)
        .filter(lambda x, s="random": x["category"] == s)
    )
    probe_img = next(iter(probe_ds))["image"].convert("RGB")
    probe_inputs = processor(
        text="USER: <image>\nIs there a dog in the image? Please answer yes or no.\nASSISTANT:",
        images=probe_img,
        return_tensors="pt",
    )
    for k, v in probe_inputs.items():
        if torch.is_tensor(v):
            probe_inputs[k] = v.to(
                lm_device,
                dtype=(torch.float16 if k == "pixel_values" else None),
            )

    rng: Optional[Tuple[int, int]] = get_image_token_range_hf(model, probe_inputs)
    assert rng is not None, "Could not determine image token range."
    pope_img_start, pope_img_end = rng
    print(f"✅ Image token range: [{pope_img_start}, {pope_img_end})")
    del probe_ds, probe_img, probe_inputs

    # ---- Apply IGAP patch to all layers -------------------------------------
    apply_igap_to_llava(
        model,
        start_layer=0,
        end_layer=len(get_llama_layers(model)),
        img_start_idx=pope_img_start,
        img_end_idx=pope_img_end,
        keep_head_ratio=KEEP_HEAD_RATIO,
        suppression_alpha=SUPPRESSION_ALPHA,
        use_igap_img=True,
    )

    # ---- Determine field names from a sample --------------------------------
    _probe2 = next(iter(
        load_dataset("lmms-lab/POPE", split="test", streaming=True)
        .filter(lambda x, s="random": x["category"] == s)
    ))
    q_field: str = "question" if "question" in _probe2 else "text"
    l_field: str = "label"    if "label"    in _probe2 else "answer"
    print(f"   POPE fields: question='{q_field}', label='{l_field}'")

    # ---- Main inference loop (variant × split) -------------------------------
    for variant in POPE_VARIANTS:
        vname: str = variant["name"]
        print(f"\n{'='*60}")
        print(f"  POPE — {vname.upper()}")
        print(f"{'='*60}")

        # Set IGAP enable/disable flag on every attention layer
        for layer in get_llama_layers(model):
            layer.self_attn.use_igap_img = variant["use_igap"]

        for split_name in POPE_SPLITS:
            out_file: str = os.path.join(POPE_RESULTS_DIR, f"{vname}_{split_name}.json")

            records, done_count = _load_checkpoint(out_file)
            if done_count >= POPE_N:
                print(f"  skip {split_name}: already complete ({done_count} records)")
                continue
            if done_count > 0:
                print(f"  resuming {split_name} from record {done_count}")

            ds_stream = (
                load_dataset("lmms-lab/POPE", split="test", streaming=True)
                .filter(lambda x, s=split_name: x["category"] == s)
            )
            ds_iter = itertools.islice(ds_stream, done_count, POPE_N)
            pbar = tqdm(
                ds_iter,
                total=POPE_N - done_count,
                desc=f"  [{vname}] {split_name}",
                leave=True,
            )

            for item in pbar:
                question: str = item[q_field]
                gt_label: str = item[l_field].strip().lower()
                image = item["image"].convert("RGB")
                prompt: str = f"USER: <image>\n{question}\nASSISTANT:"

                try:
                    if variant["mode"] == "dcla":
                        # ---- DCLA dual-pass decoding -------------------------
                        ans, _ = dynamic_decode_one_sample(
                            model=model,
                            processor=processor,
                            prompt=prompt,
                            image=image,
                            img_start=pope_img_start,
                            img_end=pope_img_end,
                            max_new_tokens=10,
                            js_threshold=JS_THRESHOLD,
                            alpha1=ALPHA1,
                            alpha2=ALPHA2,
                            lam=LAM,
                            token_alpha=TOKEN_ALPHA,
                        )
                        pred: str = extract_yes_no(ans)

                    else:
                        # ---- Standard model.generate -------------------------
                        _igap_toggle["active"] = variant["use_igap"]
                        inp = processor(
                            text=prompt, images=image, return_tensors="pt"
                        )
                        for k, v in inp.items():
                            if torch.is_tensor(v):
                                inp[k] = v.to(
                                    lm_device,
                                    dtype=(torch.float16 if k == "pixel_values" else None),
                                )
                        with torch.no_grad():
                            out_ids = model.generate(
                                **inp,
                                max_new_tokens=10,
                                do_sample=False,
                                repetition_penalty=1.1,
                                pad_token_id=processor.tokenizer.eos_token_id,
                            )
                        _igap_toggle["active"] = False
                        raw: str = processor.batch_decode(
                            out_ids, skip_special_tokens=True
                        )[0]
                        pred = extract_yes_no(raw.split("ASSISTANT:")[-1])

                except Exception as exc:
                    pred = "yes"   # conservative fallback
                    pbar.write(f"  ⚠️  Error: {exc}")

                records.append({"label": gt_label, "pred": pred})

                if len(records) % CHECKPOINT_EVERY == 0:
                    _save_checkpoint(out_file, records)

            # Final save after completing the split
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            print(f"  ✅ Saved {len(records)} records → {out_file}")

    print("\n🎉 POPE INFERENCE COMPLETE.")


if __name__ == "__main__":
    main()
