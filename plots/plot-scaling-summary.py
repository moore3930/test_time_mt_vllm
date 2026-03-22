from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

# Use non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT_DIR / "analysis_bsn" / "comparing_data" / "summary.txt"
DEFAULT_OUT_DIR = ROOT_DIR / "analysis_bsn" / "comparing_data" / "plots_summary"
TARGET_LANGS = ["en-ru", "en-nl", "en-zh"]
TARGET_METRICS = ["cometkiwixl", "comet"]


def parse_summary(summary_path: Path) -> Dict[str, List[Dict[str, str]]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    data: Dict[str, List[Dict[str, str]]] = {}
    current_lang: str | None = None
    columns: List[str] | None = None

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("===") and line.endswith("==="):
            title = line.strip("=").strip()
            lang = title.split()[0]
            current_lang = lang
            data[current_lang] = []
            columns = None
            continue

        if current_lang is None:
            continue

        if line.startswith("N\t"):
            columns = line.split("\t")
            continue

        if columns is None:
            continue

        values = line.split("\t")
        if len(values) != len(columns):
            continue
        row = dict(zip(columns, values))
        data[current_lang].append(row)

    return data


def base_lang(section_name: str) -> str:
    parts = section_name.split("-")
    if len(parts) < 2:
        return section_name
    return f"{parts[0]}-{parts[1]}"


def collect_settings_by_base_lang(
    data: Dict[str, List[Dict[str, str]]]
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = {lang: {} for lang in TARGET_LANGS}
    for section_name, rows in data.items():
        lang = base_lang(section_name)
        if lang in grouped:
            grouped[lang][section_name] = rows
    return grouped


def build_series(rows: List[Dict[str, str]], metric: str) -> tuple[List[int], List[float], List[float]]:
    mean_key = f"{metric}_mean"
    se_key = f"{metric}_se"
    xs: List[int] = []
    means: List[float] = []
    ses: List[float] = []

    for row in rows:
        xs.append(int(row["N"]))
        means.append(float(row[mean_key]))
        ses.append(float(row[se_key]))

    return xs, means, ses


def sort_settings(settings: List[str], lang: str) -> List[str]:
    preferred = [f"{lang}-32-1.000-seq", f"{lang}-32-seq", f"{lang}-greedy", lang]
    rank = {name: idx for idx, name in enumerate(preferred)}
    return sorted(settings, key=lambda s: (rank.get(s, 999), s))


def plot_one(
    lang: str,
    metric: str,
    settings_rows: Dict[str, List[Dict[str, str]]],
    out_dir: Path,
) -> Path:
    plt.figure(figsize=(7.2, 4.8), dpi=180)
    cmap = plt.get_cmap("tab10")
    xticks_set = set()

    ordered_settings = sort_settings(list(settings_rows.keys()), lang)
    for idx, setting in enumerate(ordered_settings):
        rows = settings_rows[setting]
        x, y, se = build_series(rows, metric)
        xticks_set.update(x)
        lower = [m - s for m, s in zip(y, se)]
        upper = [m + s for m, s in zip(y, se)]
        color = cmap(idx % 10)
        plt.plot(x, y, marker="o", linewidth=2.0, color=color, label=setting)
        plt.fill_between(x, lower, upper, color=color, alpha=0.15)

    xticks = sorted(xticks_set)
    plt.xscale("log", base=2)
    plt.xticks(xticks, xticks)
    plt.xlabel("N")
    plt.ylabel(metric)
    plt.title(f"{lang} | {metric} (settings comparison)")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = out_dir / f"{lang}_{metric}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot metric-vs-N charts from comparing_data/summary.txt."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to summary.txt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    data = parse_summary(args.summary)
    grouped = collect_settings_by_base_lang(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for lang in TARGET_LANGS:
        settings_rows = grouped.get(lang, {})
        if not settings_rows:
            raise KeyError(f"No settings found for language pair: {lang}")
        for metric in TARGET_METRICS:
            saved.append(plot_one(lang, metric, settings_rows, args.out_dir))

    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
