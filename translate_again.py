#!/usr/bin/env python3
"""Use vLLM + gemma-3-4b-it for batched dataset translation."""
from __future__ import annotations

import argparse
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


def build_first_turn_messages(text: str, src_lang: str, tgt_lang: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a professional translator. Your will be ask to conduct translation-related tasks. Output ONLY the "
            "transaltion segment - do NOT generate additional content.",
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
        sampling = SamplingParams(
            use_beam_search=True,
            best_of=beam_width,
            temperature=0.0,
            top_p=1.0,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            max_tokens=max_tokens,
        )
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
                                f"Please translate again for a better version. Return only the translation result.\n\n{tgt_lang}: "
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
        break
    
    # debug
    message_example = messages_per_item[0]

    return results, message_example


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def build_output_jsonl_path(results_dir: str, model: str, temperature: float, decoding: str) -> Path:
    model_name = _sanitize_for_filename(model.replace("/", "__"))
    temp_name = _sanitize_for_filename(f"{temperature:.3f}")
    decode_name = _sanitize_for_filename(decoding)
    filename = f"translations__model-{model_name}__temp-{temp_name}__decode-{decode_name}.jsonl"
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def print_round_average_scores(rows: List[Dict], num_rounds: int) -> None:
    if not rows:
        return
    print("Average scores by round:")
    for i in range(1, num_rounds + 1):
        comet_key = f"comet_hypo_{i}"
        comet_kiwi_key = f"cometkiwi_hypo_{i}"
        comet_avg = sum(float(row[comet_key]) for row in rows) / len(rows)
        comet_kiwi_avg = sum(float(row[comet_kiwi_key]) for row in rows) / len(rows)
        print(f"- round {i}: COMET={comet_avg:.4f} COMETKiwi={comet_kiwi_avg:.4f}")


def build_summary_row(rows: List[Dict], num_rounds: int, args: argparse.Namespace) -> Dict:
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
    }
    for i in range(1, num_rounds + 1):
        comet_key = f"comet_hypo_{i}"
        comet_kiwi_key = f"cometkiwi_hypo_{i}"
        summary[f"avg_{comet_key}"] = sum(float(row[comet_key]) for row in rows) / len(rows)
        summary[f"avg_{comet_kiwi_key}"] = sum(float(row[comet_kiwi_key]) for row in rows) / len(rows)
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


_COMET_MODEL_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}


def _get_comet_models(comet_model_name: str, comet_kiwi_model_name: str) -> Tuple[object, object]:
    cache_key = (comet_model_name, comet_kiwi_model_name)
    if cache_key in _COMET_MODEL_CACHE:
        return _COMET_MODEL_CACHE[cache_key]

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
    _COMET_MODEL_CACHE[cache_key] = (comet_model, comet_kiwi_model)
    return comet_model, comet_kiwi_model


def add_comet_scores(
    rows: List[Dict],
    num_rounds: int,
    score_batch_size: int,
    comet_model_name: str,
    comet_kiwi_model_name: str,
    comet_gpus: int,
) -> None:
    if not rows:
        return

    comet_model, comet_kiwi_model = _get_comet_models(
        comet_model_name=comet_model_name,
        comet_kiwi_model_name=comet_kiwi_model_name,
    )

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
    parser.add_argument("--results-dir", default="results", help="Directory to save result jsonl files")
    parser.add_argument("--score-batch-size", type=int, default=32, help="Batch size for COMET/COMETKiwi scoring")
    parser.add_argument("--comet-gpus", type=int, default=1, help="GPU count used by COMET/COMETKiwi; set 0 for CPU")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da", help="COMET model name")
    parser.add_argument("--comet-kiwi-model", default="Unbabel/wmt22-cometkiwi-da", help="COMETKiwi model name")

    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens")
    parser.add_argument(
        "--decoding",
        choices=["sampling", "greedy", "beam_search"],
        default="sampling",
        help="Decoding strategy",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling decoding")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width (for beam_search)")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty (for beam_search)")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping in beam search",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    disable_thinking = "qwen3" in args.model.lower()

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
        comet_model_name=args.comet_model,
        comet_kiwi_model_name=args.comet_kiwi_model,
        comet_gpus=args.comet_gpus,
    )
    output_path = build_output_jsonl_path(
        results_dir=args.results_dir,
        model=args.model,
        temperature=args.temperature,
        decoding=args.decoding,
    )
    summary_row = build_summary_row(results, args.num_rounds, args)
    rows_to_save = [*results, msg_example, summary_row]
    save_jsonl(str(output_path), rows_to_save)
    print(f"Saved {len(results)} translations to {output_path}")
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
    print_round_average_scores(results, args.num_rounds)


if __name__ == "__main__":
    main()
