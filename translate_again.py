#!/usr/bin/env python3
"""Use vLLM + gemma-3-4b-it for batched dataset translation."""

import argparse
import json
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompt(tokenizer: AutoTokenizer, text: str, src_lang: str, tgt_lang: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Return only the translation result.",
        },
        {
            "role": "user",
            "content": f"Translate the following text from {src_lang} to {tgt_lang}:\n\n{text}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def batched(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def extract_text_pair(
    example: Dict,
    src_key: str,
    tgt_key: str,
    source_column: Optional[str],
    target_column: Optional[str],
    translation_column: str,
) -> Optional[Tuple[str, Optional[str]]]:
    if source_column:
        src_text = example.get(source_column)
        if not isinstance(src_text, str) or not src_text.strip():
            return None
        tgt_text = None
        if target_column:
            tgt_raw = example.get(target_column)
            if isinstance(tgt_raw, str):
                tgt_text = tgt_raw
        return src_text, tgt_text

    translation_item = example.get(translation_column)
    if not isinstance(translation_item, dict):
        return None

    src_text = translation_item.get(src_key)
    tgt_text = translation_item.get(tgt_key)
    if not isinstance(src_text, str) or not src_text.strip():
        return None
    if tgt_text is not None and not isinstance(tgt_text, str):
        tgt_text = None
    return src_text, tgt_text


def load_dataset_inputs(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_samples: int,
    src_key: str,
    tgt_key: str,
    source_column: Optional[str],
    target_column: Optional[str],
    translation_column: str,
) -> List[Dict]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    items: List[Dict] = []
    for idx, example in enumerate(dataset):
        pair = extract_text_pair(
            example=example,
            src_key=src_key,
            tgt_key=tgt_key,
            source_column=source_column,
            target_column=target_column,
            translation_column=translation_column,
        )
        if pair is None:
            continue
        src_text, ref_text = pair
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
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    results: List[Dict] = []
    for chunk in batched(items, batch_size):
        prompts = [
            build_prompt(
                tokenizer=tokenizer,
                text=item["source_text"],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
            for item in chunk
        ]
        outputs = llm.generate(prompts, sampling)
        for item, output in zip(chunk, outputs):
            results.append(
                {
                    "id": item["id"],
                    "source_text": item["source_text"],
                    "reference_text": item["reference_text"],
                    "prediction": output.outputs[0].text.strip(),
                }
            )
    return results


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate text using vLLM + Gemma 3 4B IT")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Hugging Face model id")
    parser.add_argument("--src-lang", default="Chinese", help="Source language")
    parser.add_argument("--tgt-lang", default="English", help="Target language")
    parser.add_argument("--src-key", default="zh", help="Source language key in dataset translation dict")
    parser.add_argument("--tgt-key", default="en", help="Target language key in dataset translation dict")

    parser.add_argument("--dataset-name", help="HF dataset name, e.g. wmt/wmt19")
    parser.add_argument("--dataset-config", help="HF dataset config, often a language pair like zh-en")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum dataset samples to translate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for vLLM generation")
    parser.add_argument("--source-column", help="Source text column; if unset, use translation dict")
    parser.add_argument("--target-column", help="Reference text column (optional)")
    parser.add_argument("--translation-column", default="translation", help="Translation dict column name")
    parser.add_argument("--output-jsonl", default="translations.jsonl", help="Output path for dataset mode")

    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    if not args.dataset_name:
        raise ValueError("Provide --dataset-name for dataset batch translation.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    items = load_dataset_inputs(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
        src_key=args.src_key,
        tgt_key=args.tgt_key,
        source_column=args.source_column,
        target_column=args.target_column,
        translation_column=args.translation_column,
    )
    if not items:
        raise ValueError("No valid samples found. Check language keys/columns and split.")

    results = run_batch_translation(
        llm=llm,
        tokenizer=tokenizer,
        items=items,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )
    save_jsonl(args.output_jsonl, results)
    print(f"Saved {len(results)} translations to {args.output_jsonl}")
    print("Preview:")
    for row in results[:3]:
        print(f"- id={row['id']} src={row['source_text'][:40]!r} pred={row['prediction'][:60]!r}")


if __name__ == "__main__":
    main()
