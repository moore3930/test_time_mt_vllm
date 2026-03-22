from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_bsn.parallel_vs_sequential import summarize_best_of_n

DIR_PATTERN = re.compile(r"^hypo_temp-(?P<temp>\d+\.\d+)_top-p-(?P<top_p>\d+\.\d+)$")


def collect_sampling_summary(sampling_root: Path) -> Dict[str, Dict[str, Dict[str, list]]]:
    result: Dict[str, Dict[str, Dict[str, list]]] = {}

    for run_dir in sorted(p for p in sampling_root.iterdir() if p.is_dir()):
        matched = DIR_PATTERN.match(run_dir.name)
        if not matched:
            continue

        temp = matched.group("temp")
        lang_map: Dict[str, Dict[str, list]] = {}

        for fp in sorted(run_dir.glob("*.jsonl")):
            lang_pair = fp.stem
            stats_by_n = summarize_best_of_n(fp)

            n_values = []
            mean_values = []
            se_values = []
            for n in sorted(stats_by_n.keys()):
                stats = stats_by_n[n]
                if stats["count"] <= 0:
                    continue
                n_values.append(n)
                mean_values.append(round(stats["comet_mean"], 4))
                se_values.append(round(stats["comet_se"], 4))

            lang_map[lang_pair] = {
                "n": n_values,
                "mean": mean_values,
                "se": se_values,
            }

        if lang_map:
            result[temp] = lang_map

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export best-of-N sampling summary JSON grouped by temperature."
    )
    parser.add_argument(
        "--sampling-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results_32" / "google_gemma-3-27b-it" / "noise-0.000" / "sampling",
        help="Directory containing hypo_temp-*_top-p-* subdirectories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "results_32" / "google_gemma-3-27b-it" / "best_of_n_summary.json",
        help="Path to write consolidated JSON summary.",
    )
    args = parser.parse_args()

    if not args.sampling_root.exists():
        raise FileNotFoundError(f"Sampling root not found: {args.sampling_root}")

    summary = collect_sampling_summary(args.sampling_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
