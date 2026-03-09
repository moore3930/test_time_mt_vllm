from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

SAMPLE_SIZES = [1, 2, 4, 8, 16, 32, 64]
SELECT_KEY_PREFIX_CANDIDATES = ["cometkiwixl_hypo_", "avg_cometkiwixl_hypo_"]
REPORT_KEY_PREFIX_CANDIDATES = ["comet_hypo_", "avg_comet_hypo_"]


def detect_key_prefix(line_obj: Dict, candidates: List[str]) -> str | None:
    for prefix in candidates:
        if f"{prefix}1" in line_obj:
            return prefix
    return None


def read_contiguous_scores(line_obj: Dict, key_prefix: str) -> List[float]:
    scores = []
    for i in range(1, 10_000):
        key = f"{key_prefix}{i}"
        if key not in line_obj:
            break
        scores.append(float(line_obj[key]))
    if not scores:
        raise KeyError(f"No scores found for prefix: {key_prefix}")
    return scores


def summarize_best_of_n(jsonl_path: Path) -> Dict[int, Dict[str, float]]:
    best_values_by_n: Dict[int, List[float]] = {n: [] for n in SAMPLE_SIZES}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if row.get("record_type") == "summary":
                continue

            select_prefix = detect_key_prefix(row, SELECT_KEY_PREFIX_CANDIDATES)
            if select_prefix is None:
                raise KeyError(
                    f"No cometkiwixl score prefix found at {jsonl_path}:{line_idx}"
                )
            report_prefix = detect_key_prefix(row, REPORT_KEY_PREFIX_CANDIDATES)
            if report_prefix is None:
                raise KeyError(f"No comet score prefix found at {jsonl_path}:{line_idx}")

            select_scores = read_contiguous_scores(row, select_prefix)
            report_scores = read_contiguous_scores(row, report_prefix)
            num_rounds = min(len(select_scores), len(report_scores))

            for n in SAMPLE_SIZES:
                if n > num_rounds:
                    continue
                # Use cometkiwi-xl to choose index, then report comet score at the same index.
                best_idx = max(range(n), key=lambda idx: select_scores[idx])
                best_values_by_n[n].append(report_scores[best_idx])

    summary: Dict[int, Dict[str, float]] = {}
    for n, values in best_values_by_n.items():
        if not values:
            summary[n] = {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
            }
            continue

        summary[n] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values),
        }

    return summary


def process_data_dir(data_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    results: Dict[str, Dict[int, Dict[str, float]]] = {}

    for fp in sorted(data_dir.glob("*.jsonl")):
        lang_pair = fp.stem
        results[lang_pair] = summarize_best_of_n(fp)

    return results


def print_table(results: Dict[str, Dict[int, Dict[str, float]]]) -> None:
    for lang_pair, stats_by_n in results.items():
        print(
            "\n=== "
            f"{lang_pair} (select: cometkiwi-xl / report: comet) ==="
        )
        print("N\tcount\tmean(×100)\tstd(×100)")
        for n in SAMPLE_SIZES:
            s = stats_by_n[n]
            if s["count"] == 0:
                continue
            mean_x100 = s["mean"] * 100
            std_x100 = s["std"] * 100
            print(f"{n}\t{s['count']}\t{mean_x100:.1f}\t{std_x100:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute best-of-N distribution and mean using cometkiwi-xl scores from JSONL files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing language-pair JSONL files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full summary as JSON.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    results = process_data_dir(args.data_dir)
    print_table(results)

    if args.output_json is not None:
        serializable = {
            lp: {str(n): stats for n, stats in n_stats.items()}
            for lp, n_stats in results.items()
        }
        args.output_json.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON summary to: {args.output_json}")


if __name__ == "__main__":
    main()
