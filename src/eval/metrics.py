"""
src/eval/metrics.py
===================
Evaluation metric utilities for both MMHal-Bench and POPE benchmarks.

Functions
---------
parse_rating(text)
    Extract the 0–6 integer score from a Qwen judge response string.

get_stats(file_path)
    Load an evaluation JSON and compute per-category and overall average
    scores for the MMHal-Bench benchmark.

pope_metrics(records)
    Compute Accuracy, Precision, Recall, F1, and Yes%-bias from a list of
    POPE prediction records.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

# Ordered MMHal-Bench question type labels (matches the benchmark's taxonomy)
QUESTION_TYPE_NAMES: List[str] = [
    "holistic",
    "counting",
    "relation",
    "environment",
    "other",
    "attribute",
    "adversarial",
    "comparison",
]


def parse_rating(text: str) -> int:
    """Extract the integer rating (0–6) from a judge model response.

    The judge is expected to emit a line of the form::

        Rating: <N>

    where ``<N>`` ∈ {0, 1, 2, 3, 4, 5, 6}.  The regex is case-insensitive
    and tolerates common formatting variations (colons, asterisks, dashes).

    Parameters
    ----------
    text:
        Raw string output from the Qwen2.5-3B-Instruct judge model.

    Returns
    -------
    int
        The extracted rating, or ``0`` if no rating pattern is found.
    """
    normalised = (text or "").lower()
    match = re.search(r"rating[\s:\*\-]*([0-6])", normalised)
    if not match:
        return 0
    return int(match.group(1))


def get_stats(file_path: str) -> Optional[Dict[str, float]]:
    """Compute per-category and overall average scores from an evaluation JSON.

    The JSON file is expected to be a list of dicts, each with:

    * ``"question_type"`` – one of the values in ``QUESTION_TYPE_NAMES``
      (case-insensitive; unknown types are mapped to ``"other"``).
    * ``"response"``      – raw judge output string.

    Parameters
    ----------
    file_path:
        Path to the evaluation JSON (e.g. ``eval_baseline.json``).

    Returns
    -------
    Dict[str, float] or None
        Keys are the 8 MMHal-Bench category names plus ``"OVERALL"``, mapping
        to their average scores (0–6 scale).  Returns ``None`` if the file
        does not exist.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as fh:
        evaluations: List[dict] = json.load(fh)

    cat_scores: Dict[str, List[int]] = {k: [] for k in QUESTION_TYPE_NAMES}
    all_scores: List[int] = []

    for ev in evaluations:
        score: int = parse_rating(ev.get("response", ""))
        qtype: str = (ev.get("question_type") or "other").lower().strip()
        if qtype not in cat_scores:
            qtype = "other"
        cat_scores[qtype].append(score)
        all_scores.append(score)

    averages: Dict[str, float] = {
        cat: (sum(scores) / len(scores) if scores else 0.0)
        for cat, scores in cat_scores.items()
    }
    averages["OVERALL"] = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return averages


def pope_metrics(records: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute POPE benchmark evaluation metrics.

    Each record in ``records`` must contain:

    * ``"label"`` – ground-truth answer, either ``"yes"`` or ``"no"``
      (case-insensitive).
    * ``"pred"``  – model prediction, either ``"yes"`` or ``"no"``.

    Parameters
    ----------
    records:
        List of prediction records, typically loaded from a checkpoint JSON.

    Returns
    -------
    Dict[str, float]
        Keys:
        * ``"acc"``     – Accuracy (%) rounded to 2 decimal places.
        * ``"f1"``      – F1 score (%) rounded to 2 decimal places.
        * ``"yes_pct"`` – Fraction of responses that are "yes" (%).
        * ``"n"``       – Total number of evaluated records.

    Notes
    -----
    Positive class = ``"yes"`` (object present).
    Negative class = ``"no"`` (object absent).

    A ``yes_pct`` far above 50 % indicates a yes-bias, which is a known
    symptom of hallucination in VLMs.
    """
    tp = fp = tn = fn = yes_count = 0

    for record in records:
        gt: str = record["label"].strip().lower()
        pred: str = record["pred"].strip().lower()

        if pred == "yes":
            yes_count += 1

        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    total: int = tp + fp + tn + fn
    acc: float = (tp + tn) / total if total else 0.0
    precision: float = tp / (tp + fp) if (tp + fp) else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) else 0.0
    f1: float = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "acc": round(acc * 100, 2),
        "f1": round(f1 * 100, 2),
        "yes_pct": round(yes_count / total * 100, 1) if total else 0.0,
        "n": total,
    }
