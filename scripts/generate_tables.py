"""
Generate terminal tables matching the paper's evaluation structure:
- Table 1: MMHal-Bench category scores + overall
- Table 2: POPE split-wise metrics (Acc, F1, Yes-bias) + overall
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.metrics import QUESTION_TYPE_NAMES, get_stats, pope_metrics

EVAL_DIR: Path = Path("eval")
POPE_RESULTS_DIR: Path = Path("pope_results")
BENCHMARKS: List[str] = ["baseline", "spin", "mod", "igap_dcla"]
POPE_SPLITS: List[str] = ["random", "popular", "adversarial"]


def _fmt(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "  N/A"
    return f"{value:.{decimals}f}".rjust(6)


def _load_pope_metrics(benchmark: str, split_name: str) -> Optional[Dict[str, float]]:
    file_path = POPE_RESULTS_DIR / f"{benchmark}_{split_name}.json"
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as file_obj:
        rows = json.load(file_obj)
    return pope_metrics(rows)


def _macro_average(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "acc": sum(x["acc"] for x in metrics_list) / len(metrics_list),
        "f1": sum(x["f1"] for x in metrics_list) / len(metrics_list),
        "yes_pct": sum(x["yes_pct"] for x in metrics_list) / len(metrics_list),
    }


def print_table_1_mmhal() -> None:
    print("\n" + "=" * 120)
    print("Table 1: MMHal-Bench (score 0-6, higher is better)")
    print("=" * 120)

    headers = ["method", *QUESTION_TYPE_NAMES, "overall"]
    print(" | ".join(h.ljust(12) for h in headers))
    print("-" * 120)

    for benchmark in BENCHMARKS:
        stats = get_stats(str(EVAL_DIR / f"eval_{benchmark}.json"))
        row_values: List[str] = [benchmark.ljust(12)]
        for category in QUESTION_TYPE_NAMES:
            value = stats.get(category) if stats else None
            row_values.append(_fmt(value).rjust(12))
        overall = stats.get("OVERALL") if stats else None
        row_values.append(_fmt(overall).rjust(12))
        print(" | ".join(row_values))


def print_table_2_pope() -> None:
    print("\n" + "=" * 120)
    print("Table 2: POPE (Accuracy/F1/Yes-bias %) across splits")
    print("=" * 120)

    header = [
        "method",
        "random_acc",
        "random_f1",
        "random_yes",
        "popular_acc",
        "popular_f1",
        "popular_yes",
        "adversarial_acc",
        "adversarial_f1",
        "adversarial_yes",
        "overall_acc",
        "overall_f1",
        "overall_yes",
    ]
    print(" | ".join(h.ljust(14) for h in header))
    print("-" * 120)

    for benchmark in BENCHMARKS:
        split_metrics = [_load_pope_metrics(benchmark, split_name) for split_name in POPE_SPLITS]
        valid = [x for x in split_metrics if x is not None]
        overall = _macro_average(valid) if valid else None

        row: List[str] = [benchmark.ljust(14)]
        for metric in split_metrics:
            row.append(_fmt(metric["acc"] if metric else None).rjust(14))
            row.append(_fmt(metric["f1"] if metric else None).rjust(14))
            row.append(_fmt(metric["yes_pct"] if metric else None).rjust(14))

        row.append(_fmt(overall["acc"] if overall else None).rjust(14))
        row.append(_fmt(overall["f1"] if overall else None).rjust(14))
        row.append(_fmt(overall["yes_pct"] if overall else None).rjust(14))

        print(" | ".join(row))


def main() -> None:
    print_table_1_mmhal()
    print_table_2_pope()


if __name__ == "__main__":
    main()
