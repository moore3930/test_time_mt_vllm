#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

METRIC_MODEL_SPECS = {
    "comet": {
        "field_prefix": "comet",
        "default_model": "Unbabel/wmt22-comet-da",
        "needs_ref": True,
        "label": "COMET",
    },
    "comet-kiwi": {
        "field_prefix": "cometkiwi",
        "default_model": "Unbabel/wmt22-cometkiwi-da",
        "needs_ref": False,
        "label": "COMETKiwi",
    },
    "comet-kiwi-xl": {
        "field_prefix": "cometkiwixl",
        "default_model": "Unbabel/wmt23-cometkiwi-da-xl",
        "needs_ref": False,
        "label": "COMETKiwiXL",
    },
}

_COMET_MODEL_CACHE: Dict[str, object] = {}


def iter_result_files(results_dir: Path) -> Iterable[Path]:
    yield from results_dir.glob("*/*/*/*.jsonl")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_metric_models(metric_models: str) -> List[str]:
    models = [x.strip() for x in metric_models.split(",") if x.strip()]
    if not models:
        raise ValueError("Invalid --metric-models")
    unknown = [m for m in models if m not in METRIC_MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unsupported metric models: {unknown}")
    return models


def _extract_scores(prediction_result: object) -> List[float]:
    if hasattr(prediction_result, "scores"):
        scores = getattr(prediction_result, "scores")
        if isinstance(scores, list):
            return [float(x) for x in scores]
    if isinstance(prediction_result, tuple) and len(prediction_result) > 0:
        scores = prediction_result[0]
        if isinstance(scores, list):
            return [float(x) for x in scores]
    raise RuntimeError("Unexpected COMET predict output format.")


def _get_comet_model(model_name: str) -> object:
    if model_name in _COMET_MODEL_CACHE:
        return _COMET_MODEL_CACHE[model_name]
    try:
        from comet import download_model, load_from_checkpoint
    except ModuleNotFoundError as e:
        missing = e.name or "comet"
        raise SystemExit(f"Missing dependency: {missing}. Install with: pip install unbabel-comet") from e
    model = load_from_checkpoint(download_model(model_name))
    _COMET_MODEL_CACHE[model_name] = model
    return model


def _release_comet_model(model_name: str) -> None:
    model = _COMET_MODEL_CACHE.pop(model_name, None)
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def infer_num_rounds(rows: List[Dict]) -> int:
    max_round = 0
    for row in rows:
        if "id" not in row:
            continue
        for key in row:
            m = re.match(r"^hypo_(\d+)$", key)
            if m:
                max_round = max(max_round, int(m.group(1)))
    if max_round <= 0:
        raise ValueError("No hypo_* fields found in translation rows.")
    return max_round


def add_metric_scores(
    rows: List[Dict],
    num_rounds: int,
    score_batch_size: int,
    metric_models: List[str],
    comet_gpus: int,
) -> None:
    if score_batch_size <= 0:
        raise ValueError("--score-batch-size must be > 0")
    for metric_name in metric_models:
        spec = METRIC_MODEL_SPECS[metric_name]
        field_prefix = str(spec["field_prefix"])
        model_name = str(spec["default_model"])
        needs_ref = bool(spec["needs_ref"])
        model = _get_comet_model(model_name)
        for round_idx in range(1, num_rounds + 1):
            hypo_key = f"hypo_{round_idx}"
            score_key = f"{field_prefix}_{hypo_key}"
            if needs_ref:
                inputs = [{"src": r["source_text"], "mt": r[hypo_key], "ref": r["reference_text"]} for r in rows]
            else:
                inputs = [{"src": r["source_text"], "mt": r[hypo_key]} for r in rows]
            pred = model.predict(inputs, batch_size=score_batch_size, gpus=comet_gpus, progress_bar=True)
            scores = _extract_scores(pred)
            if len(scores) != len(rows):
                raise RuntimeError("Score count mismatch with translation rows.")
            for row, score in zip(rows, scores):
                row[score_key] = score
        _release_comet_model(model_name)


def build_summary_row(
    rows: List[Dict],
    num_rounds: int,
    metric_models: List[str],
    existing_summary: Dict | None,
) -> Dict:
    summary: Dict = {
        "record_type": "summary",
        "num_rows": len(rows),
        "num_rounds": num_rounds,
        "metric_models": metric_models,
    }
    if isinstance(existing_summary, dict):
        for k in ["model", "decoding", "temperature", "top_p", "beam_width", "length_penalty", "early_stopping"]:
            if k in existing_summary:
                summary[k] = existing_summary[k]
    for i in range(1, num_rounds + 1):
        for metric_name in metric_models:
            field_prefix = str(METRIC_MODEL_SPECS[metric_name]["field_prefix"])
            key = f"{field_prefix}_hypo_{i}"
            summary[f"avg_{key}"] = sum(float(r[key]) for r in rows) / len(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw translations with COMET metric models.")
    parser.add_argument("--input-dir", default="results_raw")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--comet-gpus", type=int, default=1)
    parser.add_argument("--metric-models", default="comet,comet-kiwi,comet-kiwi-xl")
    args = parser.parse_args()

    metric_models = parse_metric_models(args.metric_models)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(iter_result_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No jsonl files found under: {input_dir}")

    for in_path in files:
        rows_all = load_jsonl(in_path)
        translation_rows = [r for r in rows_all if isinstance(r, dict) and "id" in r and "hypo_1" in r]
        passthrough_rows = [r for r in rows_all if not (isinstance(r, dict) and "id" in r and "hypo_1" in r)]
        existing_summary = None
        if passthrough_rows and isinstance(passthrough_rows[-1], dict):
            if str(passthrough_rows[-1].get("record_type", "")).endswith("summary"):
                existing_summary = passthrough_rows.pop()

        if not translation_rows:
            print(f"Skip (no translation rows): {in_path}")
            continue
        num_rounds = infer_num_rounds(translation_rows)
        add_metric_scores(
            rows=translation_rows,
            num_rounds=num_rounds,
            score_batch_size=args.score_batch_size,
            metric_models=metric_models,
            comet_gpus=args.comet_gpus,
        )
        summary_row = build_summary_row(translation_rows, num_rounds, metric_models, existing_summary)
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        save_jsonl(out_path, [*translation_rows, *passthrough_rows, summary_row])
        print(f"Scored: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
