"""
scripts/mmhal_judge.py
======================
MMHal-Bench automated scoring script using Qwen2.5-3B-Instruct as judge.

For each of the four result files produced by ``mmhal_inference.py``, this
script formats the judge prompt, queries the local Qwen model, and writes a
corresponding ``eval_*.json`` file.

Input files (must exist in ``results/``)
-----------------------------------------
* ``response_baseline.json``
* ``response_just_igap.json``
* ``response_just_dcla.json``
* ``response_igap_dcla.json``

Output files (written to ``eval/``)
-------------------------------------
* ``eval_baseline.json``
* ``eval_just_igap.json``
* ``eval_just_dcla.json``
* ``eval_igap_dcla.json``

Usage
-----
Run from the repository root::

    PYTHONPATH=. python scripts/mmhal_judge.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.judge_prompts import TEMPLATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JUDGE_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR: str = "results"
EVAL_DIR: str = "eval"

# Maps result files → evaluation output files
EVAL_TASKS: List[Dict[str, str]] = [
    {
        "input":  os.path.join(RESULTS_DIR, "response_baseline.json"),
        "output": os.path.join(EVAL_DIR, "eval_baseline.json"),
    },
    {
        "input":  os.path.join(RESULTS_DIR, "response_just_igap.json"),
        "output": os.path.join(EVAL_DIR, "eval_just_igap.json"),
    },
    {
        "input":  os.path.join(RESULTS_DIR, "response_just_dcla.json"),
        "output": os.path.join(EVAL_DIR, "eval_just_dcla.json"),
    },
    {
        "input":  os.path.join(RESULTS_DIR, "response_igap_dcla.json"),
        "output": os.path.join(EVAL_DIR, "eval_igap_dcla.json"),
    },
]

JUDGE_MAX_NEW_TOKENS: int = 256


# ---------------------------------------------------------------------------
# Judge helper
# ---------------------------------------------------------------------------

def get_local_rating(
    tokenizer: AutoTokenizer,
    judge_model: AutoModelForCausalLM,
    prompt: str,
) -> str:
    """Query the Qwen judge model and return its raw text output.

    Parameters
    ----------
    tokenizer:
        Tokenizer for the Qwen judge model.
    judge_model:
        Loaded ``AutoModelForCausalLM`` inference model.
    prompt:
        The formatted judge prompt (already filled with image/question/answer).

    Returns
    -------
    str
        The full judge response, including explanation and ``Rating: N`` line.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial AI Judge. Evaluate the response based on "
                "accuracy and hallucination. Output the Explanation first, then "
                "the Rating."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(judge_model.device)

    generated_ids = judge_model.generate(
        **model_inputs,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,          # ignored when do_sample=False; silences warnings
    )
    # Strip the prompt tokens from the output
    output_ids = [
        out[len(inp):]
        for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(EVAL_DIR, exist_ok=True)

    print(f"Loading judge model: {JUDGE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, trust_remote_code=True)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ Judge model loaded.")

    for task in EVAL_TASKS:
        input_path: str = task["input"]
        output_path: str = task["output"]

        if not os.path.exists(input_path):
            print(f"⚠️  Skipping {input_path}: file not found.")
            continue

        print(f"\n👨‍⚖️  Judging: {os.path.basename(input_path)}")
        with open(input_path, "r", encoding="utf-8") as fh:
            records: List[Dict[str, Any]] = json.load(fh)

        evaluations: List[Dict[str, Any]] = []

        for i, record in tqdm(enumerate(records), total=len(records)):
            image_content: str = ", ".join(record.get("image_content", []))
            prompt_text: str = TEMPLATE.format(
                image_content,
                record.get("question", ""),
                record.get("gt_answer", ""),
                record.get("model_answer", ""),
            )

            try:
                response: str = get_local_rating(tokenizer, judge_model, prompt_text)
                evaluations.append({
                    "id":            i,
                    "question_type": (record.get("question_type") or "other").lower(),
                    "response":      response,
                })
            except Exception as exc:
                print(f"❌ Error on sample {i}: {exc}")

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(evaluations, fh, indent=2)
        print(f"✅ Saved {len(evaluations)} evaluations → {output_path}")

    print("\n🎉 ALL SCORING TASKS COMPLETE.")


if __name__ == "__main__":
    main()
