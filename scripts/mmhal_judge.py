"""
MMHal-Bench judge pipeline using Qwen/Qwen2.5-3B-Instruct.

Reads:
- results/response_baseline.json
- results/response_spin.json
- results/response_mod.json
- results/response_igap_dcla.json

Writes:
- eval/eval_baseline.json
- eval/eval_spin.json
- eval/eval_mod.json
- eval/eval_igap_dcla.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.judge_prompts import TEMPLATE

JUDGE_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR: Path = Path("results")
EVAL_DIR: Path = Path("eval")
BENCHMARKS: List[str] = ["baseline", "spin", "mod", "igap_dcla"]
JUDGE_MAX_NEW_TOKENS: int = 256


def _format_image_content(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value or "")


def get_local_rating(tokenizer: AutoTokenizer, judge_model: AutoModelForCausalLM, prompt: str) -> str:
    """Run one local judge-model evaluation and return raw output text."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial multimodal evaluator. "
                "Provide a concise explanation followed by a line: Rating: <0-6>."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(judge_model.device)

    generated = judge_model.generate(
        **model_inputs,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        do_sample=False,
    )
    completion_ids = generated[:, model_inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading judge model: {JUDGE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    for benchmark in BENCHMARKS:
        input_path = RESULTS_DIR / f"response_{benchmark}.json"
        output_path = EVAL_DIR / f"eval_{benchmark}.json"

        if not input_path.exists():
            print(f"Skipping {benchmark}: missing {input_path}")
            continue

        with input_path.open("r", encoding="utf-8") as file_obj:
            records: List[Dict[str, Any]] = json.load(file_obj)

        evaluations: List[Dict[str, Any]] = []
        for idx, record in tqdm(
            enumerate(records),
            total=len(records),
            desc=f"[judge:{benchmark}]",
        ):
            prompt = TEMPLATE.format(
                _format_image_content(record.get("image_content", "")),
                record.get("question", ""),
                record.get("gt_answer", ""),
                record.get("model_answer", ""),
            )

            try:
                judge_response = get_local_rating(tokenizer, judge_model, prompt)
            except Exception as exc:
                judge_response = f"ERROR: {exc}"

            evaluations.append(
                {
                    "id": idx,
                    "benchmark": benchmark,
                    "question_type": str(record.get("question_type", "other")).strip().lower(),
                    "response": judge_response,
                }
            )

        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(evaluations, file_obj, ensure_ascii=False, indent=2)
        print(f"Saved {len(evaluations)} judgments to {output_path}")

    print("MMHal judging complete.")


if __name__ == "__main__":
    main()
