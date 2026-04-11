"""
MMHal-Bench inference runner for LLaVA-1.5 with IGAP/DCLA ablations.

Benchmarks (exact names):
- baseline   : standard autoregressive decoding
- spin       : isolated attention pruning baseline (IGAP only)
- mod        : isolated logit-adjustment baseline (DCLA routing only)
- igap_dcla  : proposed hybrid method
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.mmhal_loader import manual_load_mmhal_bench
from src.decoding.dcla_decode import dynamic_decode_one_sample
from src.model.igap_patch import _igap_toggle, apply_igap_to_llava, get_llama_layers, igap_debug
from src.model.vision_utils import get_image_token_range_hf

MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
RESULTS_DIR: Path = Path("results")

BENCHMARKS: List[Dict[str, Any]] = [
    {"name": "baseline", "decode_mode": "generate", "use_igap": False},
    {"name": "spin", "decode_mode": "generate", "use_igap": True},
    {"name": "mod", "decode_mode": "dcla", "use_igap": False},
    {"name": "igap_dcla", "decode_mode": "dcla", "use_igap": True},
]

MAX_NEW_TOKENS: int = 128
JS_THRESHOLD: float = 0.08
ALPHA1: float = 4.0
ALPHA2: float = 1.0
LAM: float = 0.2
TOKEN_ALPHA: float = 0.06

KEEP_HEAD_RATIO: float = 0.95
SUPPRESSION_ALPHA: float = 0.08


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
    max_new_tokens: int,
    use_igap: bool,
) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    inputs = _move_inputs_to_device(inputs, device)

    _igap_toggle["active"] = bool(use_igap)
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    dataset = manual_load_mmhal_bench()
    if not dataset:
        raise RuntimeError("MMHal-Bench is empty after loading.")

    probe_prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
    probe = processor(
        text=probe_prompt,
        images=dataset[0]["image"].convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    probe = _move_inputs_to_device(probe, lm_device)

    image_span = get_image_token_range_hf(model, probe)
    if image_span is None:
        raise RuntimeError("Could not determine image token range for LLaVA inputs.")
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
        output_path = RESULTS_DIR / f"response_{name}.json"

        for layer in get_llama_layers(model):
            layer.self_attn.use_igap_img = bool(benchmark["use_igap"])

        igap_debug.update(
            {
                "forward_calls": 0,
                "igap_active_calls": 0,
                "q_len1_calls": 0,
                "suppressed_fraction_sum": 0.0,
            }
        )

        print(f"\n{'=' * 72}")
        print(f"Running benchmark: {name}")
        print(f"{'=' * 72}")

        results: List[Dict[str, Any]] = []
        for sample in tqdm(dataset, desc=f"[{name}]"):
            question = str(sample.get("question", ""))
            image = sample["image"].convert("RGB")
            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            if benchmark["decode_mode"] == "generate":
                answer = _standard_generate(
                    model=model,
                    processor=processor,
                    prompt=prompt,
                    image=image,
                    device=lm_device,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_igap=bool(benchmark["use_igap"]),
                )
                decode_trace: List[tuple[int, float, str]] = []
            else:
                answer, decode_trace = dynamic_decode_one_sample(
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

            results.append(
                {
                    "benchmark": name,
                    "question_type": sample.get("question_type", ""),
                    "question_topic": sample.get("question_topic", ""),
                    "image_id": sample.get("image_id", ""),
                    "image_src": sample.get("image_src", ""),
                    "image_content": sample.get("image_content", []),
                    "question": question,
                    "gt_answer": sample.get("gt_answer", ""),
                    "model_answer": answer,
                    "decode_trace": decode_trace[:25],
                }
            )

        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, ensure_ascii=False, indent=2)

        print(f"Saved {len(results)} samples to {output_path}")
        if igap_debug["igap_active_calls"]:
            avg_suppression = float(igap_debug["suppressed_fraction_sum"]) / float(
                igap_debug["igap_active_calls"]
            )
            print(f"IGAP avg suppressed-head fraction: {avg_suppression:.4f}")

    print("\nMMHal inference complete.")


if __name__ == "__main__":
    main()
