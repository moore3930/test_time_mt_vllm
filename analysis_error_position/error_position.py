#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {path}") from exc
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return rows


def discover_metric_steps(rows: List[Dict]) -> Dict[str, List[int]]:
    metric_steps: Dict[str, set] = defaultdict(set)
    pattern = re.compile(r"^([a-zA-Z0-9]+)_hypo_(\d+)$")
    for row in rows:
        for key in row.keys():
            m = pattern.match(key)
            if not m:
                continue
            metric = m.group(1)
            if "noise" in metric.lower():
                continue
            metric_steps[metric].add(int(m.group(2)))

    resolved: Dict[str, List[int]] = {}
    for metric, steps in metric_steps.items():
        if not steps:
            continue
        resolved[metric] = sorted(steps)

    if not resolved:
        raise ValueError("No metric_hypo_step fields found, e.g. comet_hypo_1")
    return resolved


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return mean(vals)


def compute_doc_means(rows: List[Dict], metric: str, steps: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for step in steps:
        key = f"{metric}_hypo_{step}"
        vals = [float(r[key]) for r in rows if key in r]
        out[step] = safe_mean(vals)
    return out


def compute_sentence_means(rows: List[Dict], metric: str, steps: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for step in steps:
        key = f"{metric}_hypo_{step}_sentences"
        vals: List[float] = []
        for row in rows:
            sent_scores = row.get(key)
            if isinstance(sent_scores, list):
                vals.extend(float(x) for x in sent_scores)
        out[step] = safe_mean(vals)
    return out


def discover_max_sentence_position(
    rows: List[Dict], metric: str, steps: List[int], cap: Optional[int]
) -> int:
    max_len = 0
    for row in rows:
        for step in steps:
            key = f"{metric}_hypo_{step}_sentences"
            sent_scores = row.get(key)
            if isinstance(sent_scores, list):
                max_len = max(max_len, len(sent_scores))
    if cap is not None:
        return min(max_len, cap)
    return max_len


def compute_sentence_position_means(
    rows: List[Dict], metric: str, steps: List[int], max_position: int
) -> Dict[int, Dict[int, Dict[str, float]]]:
    # Returns: step -> position(1-based) -> {"mean": x, "count": y}
    out: Dict[int, Dict[int, Dict[str, float]]] = {}
    for step in steps:
        key = f"{metric}_hypo_{step}_sentences"
        per_pos_values: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            sent_scores = row.get(key)
            if not isinstance(sent_scores, list):
                continue
            for pos0, score in enumerate(sent_scores):
                pos = pos0 + 1
                if pos > max_position:
                    break
                per_pos_values[pos].append(float(score))

        out[step] = {}
        for pos in range(1, max_position + 1):
            vals = per_pos_values.get(pos, [])
            out[step][pos] = {
                "mean": safe_mean(vals),
                "count": float(len(vals)),
            }
    return out


def fmt(v: float) -> str:
    if v != v:  # NaN
        return "nan"
    return f"{v:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze translation quality by step and sentence position")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("en-zh.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=None,
        help="最多统计到第几个句子位置（默认自动使用数据中的最大位置）",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    metric_steps = discover_metric_steps(rows)

    print(f"Loaded rows: {len(rows)}")
    print(f"Input: {args.input}")

    for metric in sorted(metric_steps.keys()):
        steps = metric_steps[metric]
        print("\n" + "=" * 70)
        print(f"Metric: {metric} | steps: {steps}")

        doc_means = compute_doc_means(rows, metric, steps)
        sent_means = compute_sentence_means(rows, metric, steps)
        max_position = discover_max_sentence_position(rows, metric, steps, args.max_position)
        pos_means = compute_sentence_position_means(rows, metric, steps, max_position=max_position)

        print("[1] Doc 平均质量（每一步）")
        print("step\tdoc_mean")
        for step in steps:
            print(f"{step}\t{fmt(doc_means[step])}")

        print("\n[2] Sentence 平均质量（每一步）")
        print("step\tsentence_mean")
        for step in steps:
            print(f"{step}\t{fmt(sent_means[step])}")

        if max_position <= 0:
            print("\n[3] 按句子位置平均分：无可用句子级分数")
            continue

        print("\n[3] 按句子绝对位置的平均分（短文档自动跳过该位置）")
        header = ["position"] + [f"step{step}" for step in steps]
        print("\t".join(header))
        for pos in range(1, max_position + 1):
            values = [f"sent_{pos}"] + [fmt(pos_means[step][pos]["mean"]) for step in steps]
            print("\t".join(values))

        print("\n[3.1] 每个位置参与平均的样本数")
        print("\t".join(header))
        for pos in range(1, max_position + 1):
            counts = [f"sent_{pos}"] + [str(int(pos_means[step][pos]["count"])) for step in steps]
            print("\t".join(counts))


if __name__ == "__main__":
    main()
