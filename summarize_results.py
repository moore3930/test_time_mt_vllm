#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

SUMMARY_KEY_RE = re.compile(r"^avg_(?P<metric>.+)_hypo_(?P<round>\d+)$")


def iter_result_files(results_dir: Path) -> Iterable[Path]:
    # Expected layout: results/<model>/<decoding>/<lang_pair>.jsonl
    yield from results_dir.glob("*/*/*.jsonl")


def load_summary_from_jsonl(path: Path) -> Dict[str, object]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty result file: {path}")
    last = json.loads(lines[-1])
    if not isinstance(last, dict):
        raise ValueError(f"Last line is not a summary dict: {path}")
    return last


def extract_metric_round_values(summary_row: Dict[str, object]) -> Dict[str, Dict[int, float]]:
    metric_map: DefaultDict[str, Dict[int, float]] = defaultdict(dict)
    for key, value in summary_row.items():
        match = SUMMARY_KEY_RE.match(key)
        if not match:
            continue
        metric = match.group("metric")
        round_idx = int(match.group("round"))
        metric_map[metric][round_idx] = float(value)
    return dict(metric_map)


def collect_scores(
    results_dir: Path,
) -> Dict[Tuple[str, str], Dict[str, Dict[str, Dict[int, float]]]]:
    # (model, decoding) -> language -> metric -> round -> avg_score
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Dict[int, float]]]] = {}
    files = sorted(iter_result_files(results_dir))
    for path in files:
        model = path.parent.parent.name
        decoding = path.parent.name
        language = path.stem

        summary = load_summary_from_jsonl(path)
        metric_values = extract_metric_round_values(summary)
        if not metric_values:
            continue

        key = (model, decoding)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][language] = metric_values
    return grouped


def print_report(grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Dict[int, float]]]]) -> None:
    if not grouped:
        print("No result files found.")
        return

    for (model, decoding) in sorted(grouped.keys()):
        lang_data = grouped[(model, decoding)]
        print(f"Model: {model}")
        print(f"Decoding: {decoding}")

        all_metrics = sorted({m for data in lang_data.values() for m in data.keys()})
        for metric in all_metrics:
            rounds = sorted({r for data in lang_data.values() if metric in data for r in data[metric].keys()})
            if not rounds:
                continue
            print(f"  Metric: {metric}")

            headers = ["language", *[f"round_{r}" for r in rounds]]
            languages = sorted(lang_data.keys())
            rows: List[List[str]] = []
            all_lang_scores_by_round: Dict[int, List[float]] = {r: [] for r in rounds}

            for language in languages:
                round_map = lang_data[language].get(metric, {})
                row = [language]
                for r in rounds:
                    if r in round_map:
                        score = round_map[r]
                        all_lang_scores_by_round[r].append(score)
                        row.append(f"{score:.4f}")
                    else:
                        row.append("-")
                rows.append(row)

            avg_row = ["all_languages_avg"]
            for r in rounds:
                scores = all_lang_scores_by_round[r]
                avg_row.append(f"{(sum(scores) / len(scores)):.4f}" if scores else "-")
            rows.append(avg_row)

            col_widths = []
            for i, header in enumerate(headers):
                width = len(header)
                for row in rows:
                    width = max(width, len(row[i]))
                col_widths.append(width)

            def fmt(row: List[str]) -> str:
                return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

            print(f"    {fmt(headers)}")
            print(f"    {'-+-'.join('-' * w for w in col_widths)}")
            for row in rows:
                print(f"    {fmt(row)}")
            print()
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize per-language and all-language average scores from translation result files."
    )
    parser.add_argument("--results-dir", default="results", help="Result root directory (default: results)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    grouped = collect_scores(results_dir)
    print_report(grouped)


if __name__ == "__main__":
    main()
