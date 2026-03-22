from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
# Force a file-rendering backend to avoid IDE backend_interagg issues.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_JSON = ROOT_DIR / "results" / "results_32" / "google_gemma-3-27b-it" / "best_of_n_summary.json"


def load_data(data_json: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    if not data_json.exists():
        raise FileNotFoundError(f"Data JSON not found: {data_json}")
    raw = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Data JSON root must be an object keyed by temperature.")
    return raw


def build_temperature_colors(temperatures: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {temp: cmap(i % 10) for i, temp in enumerate(temperatures)}


def plot_one(
    lang_pair: str,
    data_by_temperature: dict,
    out_dir: Path,
    temperature_colors: Dict[str, tuple],
    show_se: bool = False,
) -> Path:
    temperatures = sorted(data_by_temperature.keys(), key=float)
    all_lower_or_mean = []
    all_upper_or_mean = []

    plt.figure(figsize=(8, 5), dpi=160)
    for temp in temperatures:
        if lang_pair not in data_by_temperature[temp]:
            continue
        series = data_by_temperature[temp][lang_pair]
        x = series["n"]
        mean = series["mean"]
        se = series["se"]
        color = temperature_colors[temp]
        lower = [m - e for m, e in zip(mean, se)] if show_se else mean
        upper = [m + e for m, e in zip(mean, se)] if show_se else mean
        all_lower_or_mean.extend(lower)
        all_upper_or_mean.extend(upper)

        plt.plot(
            x,
            mean,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"T={temp} mean",
        )
        if show_se:
            plt.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.14,
                label=f"T={temp} ±1 SE",
            )

    if not all_lower_or_mean:
        plt.close()
        raise ValueError(f"No data found for language pair: {lang_pair}")

    plt.xscale("log", base=2)
    xticks = sorted({n for temp in temperatures for n in data_by_temperature[temp].get(lang_pair, {}).get("n", [])})
    if xticks:
        plt.xticks(xticks, xticks)
    plt.xlabel("N (number of samples)")
    plt.ylabel("COMET score")
    title = f"Best-of-N Quality Trend ({lang_pair})\nselect=COMETKiwi-XL, report=COMET"
    if show_se:
        title += ", band=±1 SE"
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.ylim(min(all_lower_or_mean) - 0.004, max(all_upper_or_mean) + 0.004)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()

    out_path = out_dir / f"best_of_n_{lang_pair}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot best-of-N COMET trends across temperatures."
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        default=DEFAULT_DATA_JSON,
        help="Path to summary JSON (temperature -> lang_pair -> {n, mean, se}).",
    )
    parser.add_argument(
        "--show-se",
        action="store_true",
        help="Draw ±1 SE shaded bands (default: off).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default="best_of_n",
        help="Directory to save output figures.",
    )
    args = parser.parse_args()

    data = load_data(args.data_json)
    temperatures = sorted(data.keys(), key=float)
    temperature_colors = build_temperature_colors(temperatures)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_pairs = sorted({lp for temp in temperatures for lp in data[temp].keys()})
    saved = []
    for lang_pair in lang_pairs:
        saved.append(
            plot_one(
                lang_pair,
                data,
                out_dir,
                temperature_colors=temperature_colors,
                show_se=args.show_se,
            )
        )

    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
