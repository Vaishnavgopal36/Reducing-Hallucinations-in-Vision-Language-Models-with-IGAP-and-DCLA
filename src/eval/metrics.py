"""
Evaluation metrics for MMHal-Bench and POPE.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

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
    """Extract a 0-6 rating from judge text.

    Returns ``-1`` if no valid rating is found.
    """
    raw = (text or "").strip().lower()
    if not raw:
        return -1

    patterns = [
        r"(?:^|\b)rating\s*[:=\-]?\s*\**\s*([0-6])\b",
        r"(?:^|\b)score\s*[:=\-]?\s*\**\s*([0-6])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return -1


def get_stats(file_path: str) -> Optional[Dict[str, float]]:
    """Compute MMHal category and overall averages from an ``eval_*.json`` file.

    Unparseable ratings (``-1`` sentinel) are excluded from averages.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as file_obj:
        evaluations: List[dict] = json.load(file_obj)

    by_category: Dict[str, List[int]] = {name: [] for name in QUESTION_TYPE_NAMES}
    all_scores: List[int] = []

    for item in evaluations:
        score = parse_rating(item.get("response", ""))
        if score < 0:
            continue

        qtype = str(item.get("question_type") or "other").strip().lower()
        if qtype not in by_category:
            qtype = "other"

        by_category[qtype].append(score)
        all_scores.append(score)

    stats: Dict[str, float] = {}
    for qtype in QUESTION_TYPE_NAMES:
        scores = by_category[qtype]
        stats[qtype] = sum(scores) / len(scores) if scores else 0.0

    stats["OVERALL"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return stats


def pope_metrics(records: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute POPE metrics: Accuracy, F1, and Yes-bias.

    ``pred`` may be ``yes``, ``no``, or ``unclear``.
    ``unclear`` is counted as incorrect for accuracy and does not increase yes-bias.
    """
    tp = fp = tn = fn = 0
    yes_count = 0
    total = 0

    for record in records:
        gt = str(record.get("label", "")).strip().lower()
        pred = str(record.get("pred", "unclear")).strip().lower()

        if gt not in {"yes", "no"}:
            continue

        total += 1

        if pred == "yes":
            yes_count += 1
            if gt == "yes":
                tp += 1
            else:
                fp += 1
        elif pred == "no":
            if gt == "no":
                tn += 1
            else:
                fn += 1
        else:
            # Treat undecidable predictions as incorrect without yes-bias inflation.
            if gt == "yes":
                fn += 1
            else:
                fp += 1

    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "acc": round(acc * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "yes_pct": round((yes_count / total) * 100.0, 2) if total else 0.0,
        "n": total,
    }
