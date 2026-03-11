import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Union


def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    path = Path(path)
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}") from e
    return records


def get_step_keys(records: List[Dict], prefix: str = "cometkiwixl_hypo_") -> List[str]:
    if not records:
        return []

    base_record = None
    for rec in records:
        if any(k.startswith(prefix) for k in rec.keys()):
            base_record = rec
            break
    if base_record is None:
        return []

    steps = []
    for key in base_record.keys():
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix.isdigit():
                steps.append((int(suffix), key))
    steps.sort(key=lambda x: x[0])
    return [key for _, key in steps]


def compute_step_means(records: List[Dict], step_keys: List[str]) -> List[float]:
    step_means = []
    for key in step_keys:
        vals = [item[key] for item in records if key in item]
        if not vals:
            raise ValueError(f"No values found for key: {key}")
        step_means.append(mean(vals))
    return step_means


def compute_prefix_max_means(records: List[Dict], step_keys: List[str]) -> List[float]:
    # For each sample, compute running max up to each step; then average per step.
    running_max_by_step: List[List[float]] = [[] for _ in step_keys]
    for item in records:
        current_max = None
        for i, key in enumerate(step_keys):
            if key not in item:
                raise ValueError(f"Missing key {key} in record id={item.get('id')}")
            value = item[key]
            if current_max is None or value > current_max:
                current_max = value
            running_max_by_step[i].append(current_max)
    return [mean(vals) for vals in running_max_by_step]


if __name__ == "__main__":
    input_path = Path("en_nl.jsonl")
    if not input_path.exists():
        input_path = Path("en-nl.jsonl")

    data = load_jsonl(input_path)
    step_keys = get_step_keys(data, prefix="cometkiwixl_hypo_")
    if not step_keys:
        raise ValueError("No cometkiwixl_hypo_* fields found")

    sample_data = [item for item in data if all(key in item for key in step_keys)]
    if not sample_data:
        raise ValueError("No complete sample records found for cometkiwixl_hypo_*")

    step_means = compute_step_means(sample_data, step_keys)
    prefix_max_means = compute_prefix_max_means(sample_data, step_keys)

    print(f"Loaded {len(data)} records from {input_path}")
    print(f"Using {len(sample_data)} sample records with cometkiwixl_hypo_*")
    print("step\tmean(cometkiwixl_hypo_step)\tmean(max_up_to_step)")
    for i, (m1, m2) in enumerate(zip(step_means, prefix_max_means), start=1):
        print(f"{i}\t{m1:.6f}\t{m2:.6f}")
