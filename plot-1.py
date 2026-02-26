#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
# Force a file-rendering backend to avoid IDE/backend conflicts (e.g. backend_interagg).
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Metric: comet, all_languages_avg
comet_greedy = [0.8303, 0.8379, 0.8366, 0.8351]
comet_sampling_0_2 = [0.8306, 0.838, 0.8364, 0.8365]
comet_sampling_0_4 = [0.8291, 0.837, 0.8365, 0.8356]
comet_sampling_0_6 = [0.8279, 0.8366, 0.8356, 0.8349]
comet_sampling_0_8 = [0.8264, 0.8354, 0.8345, 0.8335]
comet_sampling_1_0 = [0.8208, 0.8313, 0.8323, 0.8308]
comet_sampling_1_2 = [0.8092, 0.8194, 0.8206, 0.821]

# Metric: cometkiwi, all_languages_avg
cometkiwi_greedy = [0.8019, 0.8056, 0.8044, 0.8028]
cometkiwi_sampling_0_2 = [0.8024, 0.8065, 0.8047, 0.804]
cometkiwi_sampling_0_4 = [0.8007, 0.8053, 0.8033, 0.8024]
cometkiwi_sampling_0_6 = [0.7996, 0.8047, 0.8026, 0.8015]
cometkiwi_sampling_0_8 = [0.7969, 0.8021, 0.7997, 0.7989]
cometkiwi_sampling_1_0 = [0.7905, 0.7969, 0.7968, 0.7957]
cometkiwi_sampling_1_2 = [0.7753, 0.7818, 0.7823, 0.7826]

# Metric: cometkiwixl, all_languages_avg
cometkiwixl_greedy = [0.699, 0.712, 0.7126, 0.7125]
cometkiwixl_sampling_0_2 = [0.6979, 0.7124, 0.7119, 0.7131]
cometkiwixl_sampling_0_4 = [0.6966, 0.7102, 0.7104, 0.7126]
cometkiwixl_sampling_0_6 = [0.6951, 0.7099, 0.7083, 0.7097]
cometkiwixl_sampling_0_8 = [0.691, 0.7047, 0.7048, 0.7061]
cometkiwixl_sampling_1_0 = [0.681, 0.6977, 0.701, 0.7021]
cometkiwixl_sampling_1_2 = [0.6591, 0.6777, 0.6789, 0.6818]


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
    out_dir = Path("plots")

    comet_series = {
        "greedy": comet_greedy,
        "sampling_0.2": comet_sampling_0_2,
        "sampling_0.4": comet_sampling_0_4,
        "sampling_0.6": comet_sampling_0_6,
        "sampling_0.8": comet_sampling_0_8,
        "sampling_1.0": comet_sampling_1_0,
        "sampling_1.2": comet_sampling_1_2,
    }
    cometkiwi_series = {
        "greedy": cometkiwi_greedy,
        "sampling_0.2": cometkiwi_sampling_0_2,
        "sampling_0.4": cometkiwi_sampling_0_4,
        "sampling_0.6": cometkiwi_sampling_0_6,
        "sampling_0.8": cometkiwi_sampling_0_8,
        "sampling_1.0": cometkiwi_sampling_1_0,
        "sampling_1.2": cometkiwi_sampling_1_2,
    }
    cometkiwixl_series = {
        "greedy": cometkiwixl_greedy,
        "sampling_0.2": cometkiwixl_sampling_0_2,
        "sampling_0.4": cometkiwixl_sampling_0_4,
        "sampling_0.6": cometkiwixl_sampling_0_6,
        "sampling_0.8": cometkiwixl_sampling_0_8,
        "sampling_1.0": cometkiwixl_sampling_1_0,
        "sampling_1.2": cometkiwixl_sampling_1_2,
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
