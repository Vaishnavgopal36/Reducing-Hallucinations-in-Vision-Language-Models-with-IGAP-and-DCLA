"""
POPE evaluation script (streaming) for four benchmarks:
- baseline
- spin
- mod
- igap_dcla
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decoding.dcla_decode import dynamic_decode_one_sample
from src.model.igap_patch import _igap_toggle, apply_igap_to_llava, get_llama_layers
from src.model.vision_utils import get_image_token_range_hf

MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
POPE_RESULTS_DIR: Path = Path("pope_results")

POPE_SPLITS: List[str] = ["random", "popular", "adversarial"]
POPE_N: int = 3000
CHECKPOINT_EVERY: int = 25

KEEP_HEAD_RATIO: float = 0.95
SUPPRESSION_ALPHA: float = 0.08

JS_THRESHOLD: float = 0.08
ALPHA1: float = 4.0
ALPHA2: float = 1.0
LAM: float = 0.2
TOKEN_ALPHA: float = 0.06

BENCHMARKS: List[Dict[str, Any]] = [
    {"name": "baseline", "decode_mode": "generate", "use_igap": False},
    {"name": "spin", "decode_mode": "generate", "use_igap": True},
    {"name": "mod", "decode_mode": "dcla", "use_igap": False},
    {"name": "igap_dcla", "decode_mode": "dcla", "use_igap": True},
]


def extract_yes_no(text: str) -> str:
    """Extract yes/no from model output; fallback is 'unclear'."""
    normalised = str(text or "").strip().lower()
    if not normalised:
        return "unclear"

    words = normalised.split()
    if words:
        first = words[0].rstrip(".,!?:;")
        if first in {"yes", "no"}:
            return first

    for token in words:
        token = token.rstrip(".,!?:;")
        if token in {"yes", "no"}:
            return token

    if " yes" in f" {normalised}" or normalised == "yes":
        return "yes"
    if " no" in f" {normalised}" or normalised == "no":
        return "no"

    return "unclear"


def _iter_pope_split(split_name: str) -> Generator[Dict[str, Any], None, None]:
    """Yield POPE test records for one category using streaming mode."""
    stream = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    for item in stream:
        if str(item.get("category", "")).strip().lower() == split_name:
            yield item


def _load_checkpoint(path: Path) -> Tuple[List[Dict[str, str]], int]:
    if not path.exists():
        return [], 0

    with path.open("r", encoding="utf-8") as file_obj:
        records: List[Dict[str, str]] = json.load(file_obj)

    if len(records) > POPE_N:
        records = records[:POPE_N]
    return records, len(records)


def _save_checkpoint_atomic(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(records, file_obj, ensure_ascii=False, indent=2)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(tmp_path, path)


def _move_inputs_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            dtype = torch.float16 if key == "pixel_values" else None
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value
    return moved


def _standard_generate(
    model: torch.nn.Module,
    processor,
    prompt: str,
    image,
    device: torch.device,
    use_igap: bool,
) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    inputs = _move_inputs_to_device(inputs, device)

    _igap_toggle["active"] = bool(use_igap)
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    finally:
        _igap_toggle["active"] = False

    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _resolve_question_and_label(item: Dict[str, Any]) -> Tuple[str, str]:
    question = str(item.get("question") or item.get("text") or "").strip()
    label = str(item.get("label") or item.get("answer") or "").strip().lower()
    if label not in {"yes", "no"}:
        if label.startswith("y"):
            label = "yes"
        elif label.startswith("n"):
            label = "no"
    return question, label


def main() -> None:
    POPE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_ID}")
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
    print(f"Model loaded on {lm_device}")

    probe_item = next(_iter_pope_split("random"), None)
    if probe_item is None:
        raise RuntimeError("Could not read a probe sample from POPE random split.")

    probe_prompt = (
        "USER: <image>\nIs there a dog in this image? Answer yes or no.\nASSISTANT:"
    )
    probe_inputs = processor(
        text=probe_prompt,
        images=probe_item["image"].convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    probe_inputs = _move_inputs_to_device(probe_inputs, lm_device)

    image_span = get_image_token_range_hf(model, probe_inputs)
    if image_span is None:
        raise RuntimeError("Could not determine image token range from POPE probe sample.")
    img_start_idx, img_end_idx = image_span
    print(f"Image token range: [{img_start_idx}, {img_end_idx})")

    apply_igap_to_llava(
        model=model,
        start_layer=0,
        end_layer=len(get_llama_layers(model)),
        img_start_idx=img_start_idx,
        img_end_idx=img_end_idx,
        keep_head_ratio=KEEP_HEAD_RATIO,
        suppression_alpha=SUPPRESSION_ALPHA,
        use_igap_img=True,
    )

    for benchmark in BENCHMARKS:
        name = benchmark["name"]
        print(f"\n{'=' * 72}")
        print(f"POPE benchmark: {name}")
        print(f"{'=' * 72}")

        for layer in get_llama_layers(model):
            layer.self_attn.use_igap_img = bool(benchmark["use_igap"])

        for split_name in POPE_SPLITS:
            out_path = POPE_RESULTS_DIR / f"{name}_{split_name}.json"
            records, done = _load_checkpoint(out_path)

            if done >= POPE_N:
                print(f"Skipping {name}/{split_name}: already complete ({done}/{POPE_N}).")
                continue

            stream_slice = itertools.islice(_iter_pope_split(split_name), done, POPE_N)
            pbar = tqdm(
                stream_slice,
                total=POPE_N - done,
                desc=f"[{name}] {split_name}",
                leave=True,
            )

            for item in pbar:
                question, label = _resolve_question_and_label(item)
                if label not in {"yes", "no"}:
                    continue

                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                image = item["image"].convert("RGB")

                try:
                    if benchmark["decode_mode"] == "dcla":
                        answer, _ = dynamic_decode_one_sample(
                            model=model,
                            processor=processor,
                            prompt=prompt,
                            image=image,
                            img_start=img_start_idx,
                            img_end=img_end_idx,
                            max_new_tokens=10,
                            js_threshold=JS_THRESHOLD,
                            alpha1=ALPHA1,
                            alpha2=ALPHA2,
                            lam=LAM,
                            token_alpha=TOKEN_ALPHA,
                        )
                    else:
                        answer = _standard_generate(
                            model=model,
                            processor=processor,
                            prompt=prompt,
                            image=image,
                            device=lm_device,
                            use_igap=bool(benchmark["use_igap"]),
                        )
                    pred = extract_yes_no(answer)
                except Exception as exc:
                    _igap_toggle["active"] = False
                    pred = "unclear"
                    pbar.write(f"Warning ({name}/{split_name}): {exc}")

                records.append({"label": label, "pred": pred})

                if len(records) % CHECKPOINT_EVERY == 0:
                    _save_checkpoint_atomic(out_path, records)

            _save_checkpoint_atomic(out_path, records)
            print(f"Saved {len(records)} records to {out_path}")

    print("POPE evaluation inference complete.")


if __name__ == "__main__":
    main()
