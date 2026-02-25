#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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
            "content": "You are a professional translator. Output ONLY the translation segment.",
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
        items.append({"id": idx, "source_text": src_text, "reference_text": ref_text})
        if len(items) >= max_samples:
            break
    return items


def shuffle_word_order_with_ratio(text: str, noise_ratio: float) -> str:
    if noise_ratio <= 0.0:
        return text

    has_whitespace = bool(re.search(r"\s", text))
    if has_whitespace:
        tokens = text.split()
    else:
        # Fallback tokenization for languages/scripts without spaces.
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7a3]|[^\w\s]", text)

    token_count = len(tokens)
    if token_count < 2:
        return text

    num_shuffle = int(round(token_count * noise_ratio))
    if num_shuffle < 2:
        return text
    num_shuffle = min(num_shuffle, token_count)

    indices = random.sample(range(token_count), k=num_shuffle)
    selected_tokens = [tokens[idx] for idx in indices]
    random.shuffle(selected_tokens)
    for idx, shuffled_token in zip(indices, selected_tokens):
        tokens[idx] = shuffled_token

    return " ".join(tokens) if has_whitespace else "".join(tokens)


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
    history_noise_ratio: float = 0.0,
    disable_thinking: bool = False,
) -> List[Dict]:
    from vllm import SamplingParams

    if decoding == "greedy":
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    elif decoding == "beam_search":
        sampling = SamplingParams(
            use_beam_search=True,
            best_of=beam_width,
            temperature=temperature,
            top_p=1.0,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            max_tokens=max_tokens,
        )
    else:
        sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    results: List[Dict] = []
    for chunk in batched(items, batch_size):
        messages_per_item: List[List[Dict[str, str]]] = [
            build_first_turn_messages(item["source_text"], src_lang, tgt_lang) for item in chunk
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
                history_text = text
                if round_idx == 1:
                    history_text = shuffle_word_order_with_ratio(text, history_noise_ratio)
                messages_per_item[i].append({"role": "assistant", "content": history_text})
                if round_idx < num_rounds:
                    messages_per_item[i].append({"role": "user", "content": "Please translate again for a better version."})

        for item, round_texts in zip(chunk, round_predictions_per_item):
            row = {"id": item["id"], "source_text": item["source_text"], "reference_text": item["reference_text"]}
            for i, text in enumerate(round_texts, start=1):
                row[f"hypo_{i}"] = text
            results.append(row)
    return results


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    history_noise_ratio: float,
) -> Path:
    pair_name = _sanitize_for_filename(lang_pair)
    model_name = _sanitize_for_filename(model)
    noise_name = _sanitize_for_filename(f"noise-{history_noise_ratio:.3f}")
    decode_name = _sanitize_for_filename(decoding)
    if decoding == "sampling":
        hypo_name = f"hypo_temp-{temperature:.3f}_top-p-{top_p:.3f}"
    elif decoding == "beam_search":
        hypo_name = f"hypo_temp-{temperature:.3f}_beam-{beam_width}_len-pen-{length_penalty:.3f}_early-stop-{int(early_stopping)}"
    else:
        hypo_name = "hypo_greedy"
    hypo_name = _sanitize_for_filename(hypo_name)
    return Path(results_dir) / model_name / noise_name / decode_name / hypo_name / f"{pair_name}.jsonl"


def infer_pair_settings(lang_pair: str) -> tuple[str, str, str, str]:
    parts = lang_pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid --lang-pairs entry: {lang_pair}")
    src_key, tgt_key = parts
    return src_key, tgt_key, LANG_NAME_MAP.get(src_key, src_key), LANG_NAME_MAP.get(tgt_key, tgt_key)


def parse_lang_pairs(lang_pairs: str) -> List[str]:
    pairs = [x.strip() for x in lang_pairs.split(",") if x.strip()]
    if not pairs:
        raise ValueError("Invalid --lang-pairs")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate translations only (no COMET scoring).")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--dataset-root", default="datasets/wmt24pp/test")
    parser.add_argument("--lang-pairs", required=True, help="e.g. en-zh,en-ru,en-nl")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument(
        "--history-noise-ratio",
        type=float,
        default=0.0,
        help="Shuffle token order in assistant history with this ratio (0.0-1.0).",
    )
    parser.add_argument("--results-dir", default="results_raw")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--decoding", choices=["sampling", "greedy", "beam_search"], default="sampling")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if not (0.0 <= args.history_noise_ratio <= 1.0):
        raise ValueError("--history-noise-ratio must be in [0.0, 1.0]")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM
    except ModuleNotFoundError as e:
        missing = e.name or "required package"
        raise SystemExit(f"Missing dependency: {missing}. Install with: pip install transformers vllm") from e

    disable_thinking = "qwen3" in args.model.lower()
    lang_pairs = parse_lang_pairs(args.lang_pairs)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    for lang_pair in lang_pairs:
        src_key, tgt_key, src_lang, tgt_lang = infer_pair_settings(lang_pair)
        items = load_local_pair_inputs(args.dataset_root, lang_pair, src_key, tgt_key, args.max_samples)
        if not items:
            raise ValueError(f"No valid samples found for {lang_pair}")
        results = run_batch_translation(
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
            history_noise_ratio=args.history_noise_ratio,
            disable_thinking=disable_thinking,
        )
        meta = {
            "record_type": "translation_summary",
            "num_rows": len(results),
            "num_rounds": args.num_rounds,
            "history_noise_ratio": args.history_noise_ratio,
            "model": args.model,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "beam_width": args.beam_width,
            "length_penalty": args.length_penalty,
            "early_stopping": args.early_stopping,
        }
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
            history_noise_ratio=args.history_noise_ratio,
        )
        save_jsonl(output_path, [*results, meta])
        print(f"[{lang_pair}] Saved {len(results)} raw translations to {output_path}")


if __name__ == "__main__":
    main()
