#!/usr/bin/env python3
"""Use vLLM + gemma-3-4b-it for batched dataset translation."""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LANG_NAME_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "pt": "Portuguese",
    "cs": "Czech",
    "uk": "Ukrainian",
    "ko": "Korean",
    "hi": "Hindi",
    "nl": "Dutch",
    "is": "Icelandic",
}

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


def build_first_turn_messages(text: str, src_lang: str, tgt_lang: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a professional translator. Your will be ask to conduct translation-related tasks. Output ONLY the "
            "transaltion segment - do NOT generate additional content. Do not include any <think> tags. Do not show your reasoning.",
        },
        {
            "role": "user",
            "content": f"Translate the following text from {src_lang} to {tgt_lang}:\n\n{src_lang}: {text}\n\n{tgt_lang}: ",
        },
    ]


def render_chat_prompt(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    disable_thinking: bool = False,
) -> str:
    template_kwargs: Dict[str, object] = {}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def batched(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_local_pair_inputs(
    dataset_root: str,
    lang_pair: str,
    src_key: str,
    tgt_key: str,
    max_samples: int,
) -> List[Dict]:
    pair_dir = Path(dataset_root) / lang_pair
    src_path = pair_dir / f"test.{lang_pair}.{src_key}"
    tgt_path = pair_dir / f"test.{lang_pair}.{tgt_key}"

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target file not found: {tgt_path}")

    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
    items: List[Dict] = []
    for idx, (src_text, ref_text) in enumerate(zip(src_lines, tgt_lines)):
        if not src_text.strip():
            continue
        items.append(
            {
                "id": idx,
                "source_text": src_text,
                "reference_text": ref_text,
            }
        )
        if len(items) >= max_samples:
            break
    return items


def run_batch_translation(
    llm: LLM,
    tokenizer: AutoTokenizer,
    items: List[Dict],
    src_lang: str,
    tgt_lang: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    decoding: str,
    beam_width: int,
    length_penalty: float,
    early_stopping: bool,
    batch_size: int,
    num_rounds: int,
    disable_thinking: bool = False,
) -> List[Dict]:
    from vllm import SamplingParams

    if num_rounds <= 0:
        raise ValueError("--num-rounds must be > 0")

    if decoding == "greedy":
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
    elif decoding == "beam_search":
        if beam_width < 1:
            raise ValueError("--beam-width must be >= 1")
        try:
            sampling = SamplingParams(
                use_beam_search=True,
                best_of=beam_width,
                temperature=temperature,
                top_p=1.0,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                max_tokens=max_tokens,
            )
        except TypeError as e:
            raise RuntimeError(
                "Beam search was requested, but this vLLM version/config does not support "
                "`use_beam_search` in SamplingParams. Please upgrade vLLM or use another decoding mode."
            ) from e
    else:
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    results: List[Dict] = []
    for chunk in batched(items, batch_size):
        messages_per_item: List[List[Dict[str, str]]] = [
            build_first_turn_messages(
                text=item["source_text"],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            for item in chunk
        ]
        round_predictions_per_item: List[List[str]] = [[] for _ in chunk]

        for round_idx in range(1, num_rounds + 1):
            prompts = [
                render_chat_prompt(tokenizer, messages, disable_thinking=disable_thinking)
                for messages in messages_per_item
            ]
            outputs = llm.generate(prompts, sampling)
            round_texts = [output.outputs[0].text.strip() for output in outputs]

            for i, text in enumerate(round_texts):
                round_predictions_per_item[i].append(text)
                messages_per_item[i].append({"role": "assistant", "content": text})
                if round_idx < num_rounds:
                    messages_per_item[i].append(
                        {
                            "role": "user",
                            "content": (
                                f"Please translate again for a better version: "
                            ),
                        }
                    )

        for item, round_texts in zip(chunk, round_predictions_per_item):
            row = {
                "id": item["id"],
                "source_text": item["source_text"],
                "reference_text": item["reference_text"],
            }
            for i, text in enumerate(round_texts, start=1):
                row[f"hypo_{i}"] = text
            results.append(row)
            
        # debug
        # break
    
    # debug
    message_example = messages_per_item[0]

    return results, message_example


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def build_output_jsonl_path(
    results_dir: str,
    lang_pair: str,
    model: str,
    decoding: str,
    temperature: float,
    top_p: float,
    beam_width: int,
    length_penalty: float,
    early_stopping: bool,
) -> Path:
    pair_name = _sanitize_for_filename(lang_pair)
    model_name = _sanitize_for_filename(model)
    decode_name = _sanitize_for_filename(decoding)
    if decoding == "sampling":
        hypo_name = f"hypo_temp-{temperature:.3f}_top-p-{top_p:.3f}"
    elif decoding == "beam_search":
        hypo_name = (
            f"hypo_temp-{temperature:.3f}_beam-{beam_width}_"
            f"len-pen-{length_penalty:.3f}_early-stop-{int(early_stopping)}"
        )
    else:
        hypo_name = "hypo_greedy"
    hypo_name = _sanitize_for_filename(hypo_name)
    filename = f"{pair_name}.jsonl"
    out_dir = Path(results_dir) / model_name / decode_name / hypo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def print_round_average_scores(rows: List[Dict], num_rounds: int, metric_models: List[str]) -> None:
    if not rows:
        return
    print("Average scores by round:")
    for i in range(1, num_rounds + 1):
        score_parts: List[str] = []
        for metric_name in metric_models:
            spec = METRIC_MODEL_SPECS[metric_name]
            key = f"{spec['field_prefix']}_hypo_{i}"
            avg = sum(float(row[key]) for row in rows) / len(rows)
            score_parts.append(f"{spec['label']}={avg:.4f}")
        print(f"- round {i}: {' '.join(score_parts)}")


def build_summary_row(
    rows: List[Dict],
    num_rounds: int,
    args: argparse.Namespace,
    metric_models: List[str],
) -> Dict:
    summary: Dict = {
        "record_type": "summary",
        "num_rows": len(rows),
        "num_rounds": num_rounds,
        "model": args.model,
        "temperature": args.temperature,
        "decoding": args.decoding,
        "top_p": args.top_p,
        "beam_width": args.beam_width,
        "length_penalty": args.length_penalty,
        "early_stopping": args.early_stopping,
        "metric_models": metric_models,
    }
    for i in range(1, num_rounds + 1):
        for metric_name in metric_models:
            spec = METRIC_MODEL_SPECS[metric_name]
            metric_key = f"{spec['field_prefix']}_hypo_{i}"
            summary[f"avg_{metric_key}"] = sum(float(row[metric_key]) for row in rows) / len(rows)
    return summary


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


_COMET_MODEL_CACHE: Dict[str, object] = {}


def _get_comet_model(model_name: str) -> object:
    if model_name in _COMET_MODEL_CACHE:
        return _COMET_MODEL_CACHE[model_name]

    try:
        from comet import download_model, load_from_checkpoint
    except ModuleNotFoundError as e:
        missing = e.name or "comet"
        raise SystemExit(
            f"Missing dependency: {missing}. Install with: pip install unbabel-comet"
        ) from e

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
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


def add_comet_scores(
    rows: List[Dict],
    num_rounds: int,
    score_batch_size: int,
    metric_models: List[str],
    metric_model_name_map: Dict[str, str],
    comet_gpus: int,
) -> None:
    if not rows:
        return
    if score_batch_size <= 0:
        raise ValueError("score batch size must be > 0")

    for metric_name in metric_models:
        spec = METRIC_MODEL_SPECS[metric_name]
        field_prefix = str(spec["field_prefix"])
        model_name = metric_model_name_map[metric_name]
        needs_ref = bool(spec["needs_ref"])
        model = _get_comet_model(model_name)

        for round_idx in range(1, num_rounds + 1):
            hypo_key = f"hypo_{round_idx}"
            score_key = f"{field_prefix}_{hypo_key}"
            if needs_ref:
                score_inputs = [
                    {"src": row["source_text"], "mt": row[hypo_key], "ref": row["reference_text"]}
                    for row in rows
                ]
            else:
                score_inputs = [{"src": row["source_text"], "mt": row[hypo_key]} for row in rows]

            pred = model.predict(
                score_inputs,
                batch_size=score_batch_size,
                gpus=comet_gpus,
                progress_bar=True,
            )
            scores = _extract_scores(pred)
            if len(scores) != len(rows):
                raise RuntimeError("COMET score count mismatch with translation rows.")

            for row, score in zip(rows, scores):
                row[score_key] = score

        # Avoid GPU memory accumulation across multiple metric models.
        _release_comet_model(model_name)


def infer_pair_settings(
    lang_pair: str,
) -> tuple[str, str, str, str]:
    parts = lang_pair.split("-")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid --lang-pair: {lang_pair}. Expected format like en-zh.")
    src_key, tgt_key = parts[0], parts[1]
    src_lang = LANG_NAME_MAP.get(src_key, src_key)
    tgt_lang = LANG_NAME_MAP.get(tgt_key, tgt_key)
    return src_key, tgt_key, src_lang, tgt_lang


def parse_lang_pairs(lang_pairs: str) -> List[str]:
    pairs = [x.strip() for x in lang_pairs.split(",") if x.strip()]
    if not pairs:
        raise ValueError("Invalid --lang-pairs: provide at least one pair like en-zh,en-ru.")
    return pairs


def parse_metric_models(metric_models: str) -> List[str]:
    models = [x.strip() for x in metric_models.split(",") if x.strip()]
    if not models:
        raise ValueError(
            "Invalid --metric-models: provide at least one value like comet,comet-kiwi,comet-kiwi-xl."
        )
    unknown = [m for m in models if m not in METRIC_MODEL_SPECS]
    if unknown:
        supported = ", ".join(sorted(METRIC_MODEL_SPECS.keys()))
        raise ValueError(f"Unsupported metric models: {unknown}. Supported: {supported}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate text using vLLM + Gemma 3 4B IT")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Hugging Face model id")

    parser.add_argument("--dataset-root", default="datasets/wmt24pp/test", help="Local dataset root directory")
    parser.add_argument(
        "--lang-pairs",
        required=True,
        help="Comma-separated language pairs, e.g. en-zh,en-ru,en-nl",
    )
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum dataset samples to translate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM generation")
    parser.add_argument("--num-rounds", type=int, default=4, help="Total translation rounds in one chat context")
    parser.add_argument("--results-dir", default="results", help="Directory to save result jsonl files")
    parser.add_argument("--score-batch-size", type=int, default=8, help="Batch size for metric scoring")
    parser.add_argument("--comet-gpus", type=int, default=1, help="GPU count used by COMET/COMETKiwi; set 0 for CPU")
    parser.add_argument(
        "--metric-models",
        default="comet,comet-kiwi,comet-kiwi-xl",
        help="Comma-separated metric model aliases to run in order, e.g. comet,comet-kiwi,comet-kiwi-xl",
    )

    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens")
    parser.add_argument(
        "--decoding",
        choices=["sampling", "greedy", "beam_search"],
        default="sampling",
        help="Decoding strategy",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling decoding")
    parser.add_argument("--beam-width", type=int, default=1, help="Beam width (for beam_search)")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty (for beam_search)")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping in beam search",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for multi-GPU")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel size for multi-GPU")
    args = parser.parse_args()
    disable_thinking = "qwen3" in args.model.lower()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ModuleNotFoundError as e:
        missing = e.name or "required package"
        raise SystemExit(f"Missing dependency: {missing}. Install with: pip install transformers vllm") from e

    lang_pairs = parse_lang_pairs(args.lang_pairs)
    metric_models = parse_metric_models(args.metric_models)
    metric_model_name_map = {
        name: str(METRIC_MODEL_SPECS[name]["default_model"])
        for name in METRIC_MODEL_SPECS
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    for lang_pair in lang_pairs:
        src_key, tgt_key, src_lang, tgt_lang = infer_pair_settings(lang_pair=lang_pair)

        items = load_local_pair_inputs(
            dataset_root=args.dataset_root,
            lang_pair=lang_pair,
            src_key=src_key,
            tgt_key=tgt_key,
            max_samples=args.max_samples,
        )
        if not items:
            raise ValueError(f"No valid samples found for {lang_pair}. Check language keys/columns and split.")

        results, msg_example = run_batch_translation(
            llm=llm,
            tokenizer=tokenizer,
            items=items,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            decoding=args.decoding,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            batch_size=args.batch_size,
            num_rounds=args.num_rounds,
            disable_thinking=disable_thinking,
        )
        add_comet_scores(
            rows=results,
            num_rounds=args.num_rounds,
            score_batch_size=args.score_batch_size,
            metric_models=metric_models,
            metric_model_name_map=metric_model_name_map,
            comet_gpus=args.comet_gpus,
        )
        output_path = build_output_jsonl_path(
            results_dir=args.results_dir,
            lang_pair=lang_pair,
            model=args.model,
            decoding=args.decoding,
            temperature=args.temperature,
            top_p=args.top_p,
            beam_width=args.beam_width,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
        )
        summary_row = build_summary_row(results, args.num_rounds, args, metric_models=metric_models)
        rows_to_save = [*results, msg_example, summary_row]
        save_jsonl(str(output_path), rows_to_save)
        print(f"[{lang_pair}] Saved {len(results)} translations to {output_path}")
        print(f"[{lang_pair}] Preview:")
        for row in results[:3]:
            hypo_preview = " ".join([f"hypo_{i}={row[f'hypo_{i}'][:28]!r}" for i in range(1, args.num_rounds + 1)])
            score_parts: List[str] = []
            for i in range(1, args.num_rounds + 1):
                per_round = []
                for metric_name in metric_models:
                    field_prefix = str(METRIC_MODEL_SPECS[metric_name]["field_prefix"])
                    per_round.append(f"{field_prefix}_hypo_{i}={row[f'{field_prefix}_hypo_{i}']:.4f}")
                score_parts.append(" ".join(per_round))
            score_preview = " ".join(score_parts)
            print(
                f"- id={row['id']} src={row['source_text'][:40]!r} {hypo_preview} {score_preview}"
            )
        print_round_average_scores(results, args.num_rounds, metric_models=metric_models)


if __name__ == "__main__":
    main()
