"""Generate markdown summaries for MMHal-Bench and POPE results."""

from __future__ import annotations

import json
import re
from pathlib import Path
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
BENCHMARKS: List[str] = ["baseline", "spin", "mod", "igap_dcla"]
POPE_SPLITS: List[str] = ["random", "popular", "adversarial"]

OUTPUT_ROOT = Path("output")
MMHAL_DIR = OUTPUT_ROOT / "mmhal"
POPE_DIR = OUTPUT_ROOT / "pope"
LEGACY_MMHAL_DIR = Path("eval")
LEGACY_POPE_DIR = Path("pope_results")
REPORT_DIR = Path("Report")


def parse_rating(text: str) -> int:
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


def _read_json(path: Path) -> Optional[List[dict]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _resolve_mmhal_eval_path(benchmark: str) -> Optional[Path]:
    candidates = [
        MMHAL_DIR / f"eval_{benchmark}.json",
        LEGACY_MMHAL_DIR / f"eval_{benchmark}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_pope_path(benchmark: str, split_name: str) -> Optional[Path]:
    candidates = [
        POPE_DIR / f"{benchmark}_{split_name}.json",
        LEGACY_POPE_DIR / f"{benchmark}_{split_name}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_stats(benchmark: str) -> Optional[Dict[str, float]]:
    file_path = _resolve_mmhal_eval_path(benchmark)
    if file_path is None:
        return None

    evaluations = _read_json(file_path)
    if evaluations is None:
        return None

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

    stats = {qtype: (sum(scores) / len(scores) if scores else 0.0) for qtype, scores in by_category.items()}
    stats["OVERALL"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return stats


def pope_metrics(records: List[Dict[str, str]]) -> Dict[str, float]:
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
    }


def _macro_average(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "acc": sum(item["acc"] for item in metrics_list) / len(metrics_list),
        "f1": sum(item["f1"] for item in metrics_list) / len(metrics_list),
        "yes_pct": sum(item["yes_pct"] for item in metrics_list) / len(metrics_list),
    }


def _build_mmhal_markdown() -> str:
    lines = [
        "# MMHal-Bench Results",
        "",
        "| Method | Holistic | Counting | Relation | Environment | Other | Attribute | Adversarial | Comparison | Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for benchmark in BENCHMARKS:
        stats = get_stats(benchmark)
        if stats is None:
            values = ["N/A"] * (len(QUESTION_TYPE_NAMES) + 1)
        else:
            values = [f"{stats[name]:.2f}" for name in QUESTION_TYPE_NAMES]
            values.append(f"{stats['OVERALL']:.2f}")

        pretty_name = "Our Hybrid" if benchmark == "igap_dcla" else benchmark.upper() if benchmark in {"spin", "mod"} else benchmark.capitalize()
        lines.append(f"| {pretty_name} | " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def _build_pope_markdown() -> str:
    lines = [
        "# POPE Results",
        "",
        "| Method | Random Acc | Random F1 | Random Yes% | Popular Acc | Popular F1 | Popular Yes% | Adversarial Acc | Adversarial F1 | Adversarial Yes% | Overall Acc | Overall F1 | Overall Yes% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for benchmark in BENCHMARKS:
        split_metrics: List[Optional[Dict[str, float]]] = []
        for split_name in POPE_SPLITS:
            file_path = _resolve_pope_path(benchmark, split_name)
            rows = _read_json(file_path) if file_path is not None else None
            split_metrics.append(pope_metrics(rows) if rows is not None else None)

        valid_metrics = [metric for metric in split_metrics if metric is not None]
        overall = _macro_average(valid_metrics) if valid_metrics else None

        row_values: List[str] = []
        for metric in split_metrics:
            if metric is None:
                row_values.extend(["N/A", "N/A", "N/A"])
            else:
                row_values.extend(
                    [f"{metric['acc']:.2f}", f"{metric['f1']:.2f}", f"{metric['yes_pct']:.2f}"]
                )

        if overall is None:
            row_values.extend(["N/A", "N/A", "N/A"])
        else:
            row_values.extend(
                [f"{overall['acc']:.2f}", f"{overall['f1']:.2f}", f"{overall['yes_pct']:.2f}"]
            )

        pretty_name = "Our Hybrid" if benchmark == "igap_dcla" else benchmark.upper() if benchmark in {"spin", "mod"} else benchmark.capitalize()
        lines.append(f"| {pretty_name} | " + " | ".join(row_values) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    mmhal_table = _build_mmhal_markdown()
    pope_table = _build_pope_markdown()

    mmhal_path = REPORT_DIR / "mmhal_table.md"
    pope_path = REPORT_DIR / "pope_table.md"

    with mmhal_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(mmhal_table)
    with pope_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(pope_table)

    print(mmhal_table)
    print(pope_table)
    print(f"Saved {mmhal_path}")
    print(f"Saved {pope_path}")


if __name__ == "__main__":
    main()
