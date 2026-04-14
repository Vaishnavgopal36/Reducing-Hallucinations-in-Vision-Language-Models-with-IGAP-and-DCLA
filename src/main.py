"""Main MMHal-Bench inference runner for baseline, SPIN, MoD, and IGAP-DCLA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import manual_load_mmhal_bench
from model.igap_dcla import (
    _spin_toggle,
    apply_igap_to_llava,
    dynamic_decode_one_sample,
    get_image_token_range_hf,
    get_llama_layers,
    spin_debug,
)

MODEL_ID: str = "llava-hf/llava-1.5-7b-hf"
KEEP_HEAD_RATIO: float = 0.95
SUPPRESSION_ALPHA: float = 0.08
JS_THRESHOLD: float = 0.05
ALPHA1: float = 4.0
ALPHA2: float = 1.0
LAM: float = 0.2
TOKEN_ALPHA: float = 0.06

BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "baseline": {"name": "baseline", "decode_mode": "generate", "use_igap": False},
    "spin": {"name": "spin", "decode_mode": "generate", "use_igap": True},
    "mod": {"name": "mod", "decode_mode": "dcla", "use_igap": False},
    "igap_dcla": {"name": "igap_dcla", "decode_mode": "dcla", "use_igap": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["baseline", "spin", "mod", "igap_dcla", "all"],
        default="all",
        help="Benchmark variant to run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated answer tokens per sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap for smoke tests.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mmhal_data"),
        help="Path to the extracted MMHal-Bench directory.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data/test_data.zip"),
        help="Path to the MMHal-Bench zip archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/mmhal"),
        help="Directory for response_<benchmark>.json outputs.",
    )
    return parser.parse_args()


def _selected_benchmarks(name: str) -> List[Dict[str, Any]]:
    if name == "all":
        return [BENCHMARKS[key] for key in ("baseline", "spin", "mod", "igap_dcla")]
    return [BENCHMARKS[name]]


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

    _spin_toggle["active"] = bool(use_igap)
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    finally:
        _spin_toggle["active"] = False

    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _load_model_and_processor():
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

    return model, processor


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = _load_model_and_processor()
    lm_device: torch.device = next(model.language_model.parameters()).device
    print(f"Model loaded on {lm_device}")

    dataset = manual_load_mmhal_bench(data_dir=args.data_dir, zip_path=args.zip_path)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    if not dataset:
        raise RuntimeError("MMHal-Bench is empty after loading.")

    probe_prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
    probe_inputs = processor(
        text=probe_prompt,
        images=dataset[0]["image"].convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    probe_inputs = _move_inputs_to_device(probe_inputs, lm_device)

    image_span = get_image_token_range_hf(model, probe_inputs)
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
        use_spin_img=True,
    )

    for benchmark in _selected_benchmarks(args.benchmark):
        name = benchmark["name"]
        output_path = args.output_dir / f"response_{name}.json"

        for layer in get_llama_layers(model):
            layer.self_attn.use_spin_img = bool(benchmark["use_igap"])

        spin_debug.update(
            {
                "forward_calls": 0,
                "spin_active_calls": 0,
                "q_len1_calls": 0,
                "avg_suppressed_fraction_sum": 0.0,
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
                    max_new_tokens=args.max_new_tokens,
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
                    max_new_tokens=args.max_new_tokens,
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
        if spin_debug["spin_active_calls"]:
            avg_suppression = float(spin_debug["avg_suppressed_fraction_sum"]) / float(
                spin_debug["spin_active_calls"]
            )
            print(f"IGAP avg suppressed-head fraction: {avg_suppression:.4f}")

    print("\nMMHal inference complete.")


if __name__ == "__main__":
    main()
