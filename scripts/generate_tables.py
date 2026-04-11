"""
scripts/generate_tables.py
==========================
Aggregate all evaluation JSONs and print the MMHal and POPE results tables.

MMHal Table
-----------
Reads ``eval/eval_*.json`` files and prints a per-category (8 categories)
+ overall score table.

POPE Table
----------
Reads ``pope_results/*.json`` files and prints Accuracy, F1, and Yes%-bias
across all three splits (Random / Popular / Adversarial) plus an Overall
column.

Usage
-----
Run from the repository root after both evaluation scripts have completed::

    PYTHONPATH=. python scripts/generate_tables.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import QUESTION_TYPE_NAMES, get_stats, pope_metrics

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
EVAL_DIR: str = "eval"
POPE_RESULTS_DIR: str = "pope_results"

POPE_SPLITS: List[str] = ["random", "popular", "adversarial"]

# Display-order config for MMHal
MMHAL_RUNS: Dict[str, str] = {
    "Baseline":   os.path.join(EVAL_DIR, "eval_baseline.json"),
    "Just IGAP":  os.path.join(EVAL_DIR, "eval_just_igap.json"),
    "Just DCLA":  os.path.join(EVAL_DIR, "eval_just_dcla.json"),
    "IGAP-DCLA":  os.path.join(EVAL_DIR, "eval_igap_dcla.json"),
}

# Display-order config for POPE
POPE_VARIANTS: Dict[str, str] = {
    "Baseline":   "baseline",
    "Just IGAP":  "just_igap",
    "Just DCLA":  "just_dcla",
    "IGAP-DCLA":  "igap_dcla",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], width: int = 10, decimals: int = 2) -> str:
    """Format a float for a fixed-width table column."""
    if value is None:
        return "N/A".ljust(width)
    return f"{value:.{decimals}f}".rjust(width)


def _hr(width: int, char: str = "-") -> str:
    return char * width


# ---------------------------------------------------------------------------
# MMHal table
# ---------------------------------------------------------------------------

def print_mmhal_table() -> None:
    """Load evaluation JSONs and print the MMHal-Bench results table."""
    col_width: int = 12
    label_width: int = 16

    # Column headers
    run_names = list(MMHAL_RUNS.keys())
    header_row = f"{'Category':<{label_width}}" + "".join(
        n.rjust(col_width) for n in run_names
    )
    sep = _hr(label_width + col_width * len(run_names))

    print("\n" + "=" * len(sep))
    print("  MMHal-Bench Average Scores (0–6, higher is better)")
    print("=" * len(sep))
    print(header_row)
    print(sep)

    # Load stats once
    all_stats: Dict[str, Optional[Dict[str, float]]] = {
        name: get_stats(path) for name, path in MMHAL_RUNS.items()
    }

    for cat in QUESTION_TYPE_NAMES:
        row = f"{cat.capitalize():<{label_width}}"
        for run_name in run_names:
            stats = all_stats[run_name]
            val = stats.get(cat) if stats else None
            row += _fmt(val, width=col_width)
        print(row)

    print(sep)
    # Overall row
    row = f"{'OVERALL':<{label_width}}"
    for run_name in run_names:
        stats = all_stats[run_name]
        val = stats.get("OVERALL") if stats else None
        row += _fmt(val, width=col_width)
    print(row)
    print("=" * len(sep))


# ---------------------------------------------------------------------------
# POPE table
# ---------------------------------------------------------------------------

def _load_pope_split(
    variant_key: str, split_name: str
) -> Optional[Dict[str, float]]:
    """Load pope_results/{variant_key}_{split_name}.json and compute metrics."""
    fpath = os.path.join(POPE_RESULTS_DIR, f"{variant_key}_{split_name}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r", encoding="utf-8") as fh:
        records: List[Dict[str, str]] = json.load(fh)
    return pope_metrics(records)


def _pope_overall(
    per_split: Dict[str, Optional[Dict[str, float]]]
) -> Optional[Dict[str, float]]:
    """Macro-average Accuracy and F1 across available splits."""
    vals = [v for v in per_split.values() if v is not None]
    if not vals:
        return None
    return {
        "acc": round(sum(v["acc"] for v in vals) / len(vals), 2),
        "f1":  round(sum(v["f1"]  for v in vals) / len(vals), 2),
    }


def print_pope_table() -> None:
    """Load POPE result JSONs and print the accuracy / F1 / Yes-bias table."""
    # Pre-load all split metrics
    all_pope: Dict[str, Dict[str, Optional[Dict[str, float]]]] = {}
    for display_name, variant_key in POPE_VARIANTS.items():
        all_pope[display_name] = {
            split: _load_pope_split(variant_key, split)
            for split in POPE_SPLITS
        }

    col_w: int = 9        # width per metric column pair
    label_w: int = 16
    split_cols = POPE_SPLITS + ["overall"]
    split_labels = ["Random", "Popular", "Adversarial", "Overall"]

    # Header rows
    top_sep = "=" * (label_w + col_w * 2 * len(split_cols) + len(split_cols) * 2)

    print("\n" + top_sep)
    print("  POPE Evaluation Table")
    print(top_sep)

    # Sub-header: split names spanning 2 metric cols each
    sub_hdr = " " * label_w
    for slabel in split_labels:
        sub_hdr += slabel.center(col_w * 2 + 2)
    print(sub_hdr)

    # Metric sub-header
    metric_hdr = f"{'Method':<{label_w}}"
    for _ in split_labels:
        metric_hdr += "Acc(%)".rjust(col_w) + "F1(%)".rjust(col_w) + "  "
    print(metric_hdr)
    print(_hr(len(top_sep)))

    for display_name in POPE_VARIANTS:
        row = f"{display_name:<{label_w}}"
        per_split = all_pope[display_name]
        ov = _pope_overall(per_split)
        for split in POPE_SPLITS:
            m = per_split.get(split)
            row += _fmt(m["acc"] if m else None, width=col_w)
            row += _fmt(m["f1"]  if m else None, width=col_w)
            row += "  "
        row += _fmt(ov["acc"] if ov else None, width=col_w)
        row += _fmt(ov["f1"]  if ov else None, width=col_w)
        row += "  "
        print(row)

    print(top_sep)

    # Yes-bias sub-table
    print("\n" + _hr(70))
    print("Yes% Bias  (ideal ~50%  |  >70% = over-predicts 'yes' = hallucination bias)")
    print(_hr(70))
    print(f"  {'Method':<20}" + "".join(f"{s:>14}" for s in split_labels[:-1]))
    for display_name, variant_key in POPE_VARIANTS.items():
        row = f"  {display_name:<20}"
        for split in POPE_SPLITS:
            m = _load_pope_split(variant_key, split)
            row += f"{(str(m['yes_pct']) + '%') if m else 'N/A':>14}"
        print(row)
    print(_hr(70))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print_mmhal_table()
    print_pope_table()


if __name__ == "__main__":
    main()
