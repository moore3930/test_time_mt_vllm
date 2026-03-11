#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
# Force a file-rendering backend to avoid IDE/backend conflicts (e.g. backend_interagg).
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Metric: comet, all_languages_avg
comet_greedy = [0.8303, 0.8379, 0.8366, 0.8351]
comet_noise_0_2 = [0.8303, 0.8363, 0.837, 0.8361]
comet_noise_0_4 = [0.8303, 0.8356, 0.8381, 0.8368]
comet_noise_0_6 = [0.8303, 0.8337, 0.8374, 0.837]
comet_noise_0_8 = [0.8304, 0.8339, 0.8376, 0.837]
comet_noise_1_0 = [0.8301, 0.8334, 0.8375, 0.8376]

# Metric: cometkiwi, all_languages_avg
cometkiwi_greedy = [0.8019, 0.8056, 0.8044, 0.8028]
cometkiwi_noise_0_2 = [0.8019, 0.8055, 0.8054, 0.8041]
cometkiwi_noise_0_4 = [0.8019, 0.8041, 0.8057, 0.8041]
cometkiwi_noise_0_6 = [0.8019, 0.8036, 0.8053, 0.8056]
cometkiwi_noise_0_8 = [0.8018, 0.8043, 0.8067, 0.8061]
cometkiwi_noise_1_0 = [0.8018, 0.8035, 0.8056, 0.8046]

# Metric: cometkiwixl, all_languages_avg
cometkiwixl_greedy = [0.699, 0.712, 0.7126, 0.7125]
cometkiwixl_noise_0_2 = [0.6986, 0.7091, 0.7118, 0.7125]
cometkiwixl_noise_0_4 = [0.6986, 0.7061, 0.7123, 0.7132]
cometkiwixl_noise_0_6 = [0.6986, 0.7041, 0.7114, 0.7134]
cometkiwixl_noise_0_8 = [0.6986, 0.7047, 0.7129, 0.7132]
cometkiwixl_noise_1_0 = [0.6984, 0.7048, 0.7114, 0.7125]


def plot_metric(metric_name: str, series: dict[str, list[float]], out_dir: Path) -> Path:
    rounds = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, values in series.items():
        ax.plot(rounds, values, marker="o", linewidth=2, label=label)

    ax.set_title(f"{metric_name} (all_languages_avg)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"round-{r}" for r in rounds])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{metric_name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = Path("plot2")

    comet_series = {
        "greedy": comet_greedy,
        "noise_0.2": comet_noise_0_2,
        "noise_0.4": comet_noise_0_4,
        "noise_0.6": comet_noise_0_6,
        "noise_0.8": comet_noise_0_8,
        "noise_1.0": comet_noise_1_0,
    }
    cometkiwi_series = {
        "greedy": cometkiwi_greedy,
        "noise_0.2": cometkiwi_noise_0_2,
        "noise_0.4": cometkiwi_noise_0_4,
        "noise_0.6": cometkiwi_noise_0_6,
        "noise_0.8": cometkiwi_noise_0_8,
        "noise_1.0": cometkiwi_noise_1_0,
    }
    cometkiwixl_series = {
        "greedy": cometkiwixl_greedy,
        "noise_0.2": cometkiwixl_noise_0_2,
        "noise_0.4": cometkiwixl_noise_0_4,
        "noise_0.6": cometkiwixl_noise_0_6,
        "noise_0.8": cometkiwixl_noise_0_8,
        "noise_1.0": cometkiwixl_noise_1_0,
    }

    written = [
        plot_metric("comet", comet_series, out_dir),
        plot_metric("cometkiwi", cometkiwi_series, out_dir),
        plot_metric("cometkiwixl", cometkiwixl_series, out_dir),
    ]

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
