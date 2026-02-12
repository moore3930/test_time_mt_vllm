#!/usr/bin/env python3
"""Use vLLM + gemma-3-4b-it for batched dataset translation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

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


def build_first_turn_messages(text: str, src_lang: str, tgt_lang: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a professional translator. Return only the translation result.",
        },
        {
            "role": "user",
            "content": f"Translate the following text from {src_lang} to {tgt_lang}:\n\n{text}",
        },
    ]


def render_chat_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    batch_size: int,
    num_rounds: int,
) -> List[Dict]:
    from vllm import SamplingParams

    if num_rounds <= 0:
        raise ValueError("--num-rounds must be > 0")

    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
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
            prompts = [render_chat_prompt(tokenizer, messages) for messages in messages_per_item]
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
                                f"Translate again for a better version.\n"
                                f"Return only the improved {tgt_lang} translation."
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
    return results


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def add_comet_scores(
    rows: List[Dict],
    num_rounds: int,
    score_batch_size: int,
    comet_model_name: str,
    comet_kiwi_model_name: str,
    comet_gpus: int,
) -> None:
    try:
        from comet import download_model, load_from_checkpoint
    except ModuleNotFoundError as e:
        missing = e.name or "comet"
        raise SystemExit(
            f"Missing dependency: {missing}. Install with: pip install unbabel-comet"
        ) from e

    comet_model_path = download_model(comet_model_name)
    comet_kiwi_model_path = download_model(comet_kiwi_model_name)
    comet_model = load_from_checkpoint(comet_model_path)
    comet_kiwi_model = load_from_checkpoint(comet_kiwi_model_path)

    for round_idx in range(1, num_rounds + 1):
        hypo_key = f"hypo_{round_idx}"
        comet_key = f"comet_{hypo_key}"
        comet_kiwi_key = f"cometkiwi_{hypo_key}"

        comet_inputs = [
            {"src": row["source_text"], "mt": row[hypo_key], "ref": row["reference_text"]}
            for row in rows
        ]
        comet_kiwi_inputs = [{"src": row["source_text"], "mt": row[hypo_key]} for row in rows]

        comet_pred = comet_model.predict(
            comet_inputs,
            batch_size=score_batch_size,
            gpus=comet_gpus,
            progress_bar=True,
        )
        comet_kiwi_pred = comet_kiwi_model.predict(
            comet_kiwi_inputs,
            batch_size=score_batch_size,
            gpus=comet_gpus,
            progress_bar=True,
        )

        comet_scores = _extract_scores(comet_pred)
        comet_kiwi_scores = _extract_scores(comet_kiwi_pred)

        if len(comet_scores) != len(rows) or len(comet_kiwi_scores) != len(rows):
            raise RuntimeError("COMET score count mismatch with translation rows.")

        for row, comet_score, comet_kiwi_score in zip(rows, comet_scores, comet_kiwi_scores):
            row[comet_key] = comet_score
            row[comet_kiwi_key] = comet_kiwi_score


def infer_pair_settings(
    lang_pair: str,
    src_key_override: str | None,
    tgt_key_override: str | None,
    src_lang_override: str | None,
    tgt_lang_override: str | None,
) -> tuple[str, str, str, str]:
    parts = lang_pair.split("-")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid --lang-pair: {lang_pair}. Expected format like en-zh.")
    inferred_src_key, inferred_tgt_key = parts[0], parts[1]
    src_key = src_key_override or inferred_src_key
    tgt_key = tgt_key_override or inferred_tgt_key
    src_lang = src_lang_override or LANG_NAME_MAP.get(src_key, src_key)
    tgt_lang = tgt_lang_override or LANG_NAME_MAP.get(tgt_key, tgt_key)
    return src_key, tgt_key, src_lang, tgt_lang


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate text using vLLM + Gemma 3 4B IT")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Hugging Face model id")
    parser.add_argument("--src-lang", help="Source language (optional override)")
    parser.add_argument("--tgt-lang", help="Target language (optional override)")
    parser.add_argument("--src-key", help="Source language key (optional override)")
    parser.add_argument("--tgt-key", help="Target language key (optional override)")

    parser.add_argument("--dataset-root", default="datasets/wmt24pp/test", help="Local dataset root directory")
    parser.add_argument("--lang-pair", required=True, help="Language pair folder name, e.g. en-zh")
    parser.add_argument("--max-samples", type=int, default=1024, help="Maximum dataset samples to translate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM generation")
    parser.add_argument("--num-rounds", type=int, default=4, help="Total translation rounds in one chat context")
    parser.add_argument("--output-jsonl", default="translations.jsonl", help="Output path for dataset mode")
    parser.add_argument("--score-batch-size", type=int, default=32, help="Batch size for COMET/COMETKiwi scoring")
    parser.add_argument("--comet-gpus", type=int, default=1, help="GPU count used by COMET/COMETKiwi; set 0 for CPU")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da", help="COMET model name")
    parser.add_argument("--comet-kiwi-model", default="Unbabel/wmt22-cometkiwi-da", help="COMETKiwi model name")

    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ModuleNotFoundError as e:
        missing = e.name or "required package"
        raise SystemExit(f"Missing dependency: {missing}. Install with: pip install transformers vllm") from e

    src_key, tgt_key, src_lang, tgt_lang = infer_pair_settings(
        lang_pair=args.lang_pair,
        src_key_override=args.src_key,
        tgt_key_override=args.tgt_key,
        src_lang_override=args.src_lang,
        tgt_lang_override=args.tgt_lang,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    items = load_local_pair_inputs(
        dataset_root=args.dataset_root,
        lang_pair=args.lang_pair,
        src_key=src_key,
        tgt_key=tgt_key,
        max_samples=args.max_samples,
    )
    if not items:
        raise ValueError("No valid samples found. Check language keys/columns and split.")

    results = run_batch_translation(
        llm=llm,
        tokenizer=tokenizer,
        items=items,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_rounds=args.num_rounds,
    )
    add_comet_scores(
        rows=results,
        num_rounds=args.num_rounds,
        score_batch_size=args.score_batch_size,
        comet_model_name=args.comet_model,
        comet_kiwi_model_name=args.comet_kiwi_model,
        comet_gpus=args.comet_gpus,
    )
    save_jsonl(args.output_jsonl, results)
    print(f"Saved {len(results)} translations to {args.output_jsonl}")
    print("Preview:")
    for row in results[:3]:
        hypo_preview = " ".join([f"hypo_{i}={row[f'hypo_{i}'][:28]!r}" for i in range(1, args.num_rounds + 1)])
        score_preview = " ".join(
            [
                f"comet_hypo_{i}={row[f'comet_hypo_{i}']:.4f} cometkiwi_hypo_{i}={row[f'cometkiwi_hypo_{i}']:.4f}"
                for i in range(1, args.num_rounds + 1)
            ]
        )
        print(
            f"- id={row['id']} src={row['source_text'][:40]!r} {hypo_preview} {score_preview}"
        )


if __name__ == "__main__":
    main()
