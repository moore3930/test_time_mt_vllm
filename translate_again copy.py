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


def build_second_turn_messages(
    source_text: str,
    hypo_1: str,
    src_lang: str,
    tgt_lang: str,
) -> List[Dict[str, str]]:
    return [
        *build_first_turn_messages(source_text, src_lang, tgt_lang),
        {"role": "assistant", "content": hypo_1},
        {
            "role": "user",
            "content": (
                f"Please translate again for a better version: "
            ),
        },
    ]


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
) -> List[Dict]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    results: List[Dict] = []
    for chunk in batched(items, batch_size):
        first_round_prompts = []
        for item in chunk:
            messages = build_first_turn_messages(
                text=item["source_text"],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            first_round_prompts.append(render_chat_prompt(tokenizer, messages))
        first_round_outputs = llm.generate(first_round_prompts, sampling)
        hypo_1_texts = [output.outputs[0].text.strip() for output in first_round_outputs]

        second_round_prompts = [
            render_chat_prompt(
                tokenizer=tokenizer,
                messages=build_second_turn_messages(
                source_text=item["source_text"],
                hypo_1=hypo_1,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                ),
            )
            for item, hypo_1 in zip(chunk, hypo_1_texts)
        ]
        second_round_outputs = llm.generate(second_round_prompts, sampling)

        for item, hypo_1, output in zip(chunk, hypo_1_texts, second_round_outputs):
            results.append(
                {
                    "id": item["id"],
                    "source_text": item["source_text"],
                    "reference_text": item["reference_text"],
                    "hypo_1": hypo_1,
                    "hypo_2": output.outputs[0].text.strip(),
                }
            )
    return results


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument("--output-jsonl", default="translations.jsonl", help="Output path for dataset mode")

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
    )
    save_jsonl(args.output_jsonl, results)
    print(f"Saved {len(results)} translations to {args.output_jsonl}")
    print("Preview:")
    for row in results[:3]:
        print(
            f"- id={row['id']} src={row['source_text'][:40]!r} "
            f"hypo_1={row['hypo_1'][:40]!r} hypo_2={row['hypo_2'][:60]!r}"
        )


if __name__ == "__main__":
    main()
