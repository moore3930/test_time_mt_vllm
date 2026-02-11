#!/usr/bin/env python3
"""Use vLLM + gemma-3-4b-it for batched dataset translation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


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
    parser.add_argument("--src-key", default="zh", help="Source language key, e.g. zh")
    parser.add_argument("--tgt-key", default="en", help="Target language key, e.g. en")

    parser.add_argument("--dataset-root", default="datasets/wmt24pp/test", help="Local dataset root directory")
    parser.add_argument("--lang-pair", required=True, help="Language pair folder name, e.g. en-zh")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum dataset samples to translate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for vLLM generation")
    parser.add_argument("--output-jsonl", default="translations.jsonl", help="Output path for dataset mode")

    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ModuleNotFoundError as e:
        missing = e.name or "required package"
        raise SystemExit(f"Missing dependency: {missing}. Install with: pip install transformers vllm") from e

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    items = load_local_pair_inputs(
        dataset_root=args.dataset_root,
        lang_pair=args.lang_pair,
        src_key=args.src_key,
        tgt_key=args.tgt_key,
        max_samples=args.max_samples,
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
