#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ALL_LANG_PAIRS = [
    "en-cs",
    "en-de",
    "en-fr",
    "en-he",
    "en-ja",
    "en-ru",
    "en-uk",
    "en-zh",
    "en-nl",
]


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
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_joiner(original_text: str, sentences: List[str], fallback: str) -> str:
    if not sentences:
        return fallback
    for sep in ("\n", " ", ""):
        if sep.join(sentences) == original_text:
            return sep
    return fallback


def build_windowed_rows(rows: List[Dict], window_size: int = 2) -> Tuple[List[Dict], int]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    out_rows: List[Dict] = []
    skipped_short_docs = 0

    for row_idx, row in enumerate(rows):
        src_sents = row.get("source_sentences")
        tgt_sents = row.get("target_sentences")
        seg_ids = row.get("seg_ids")

        if not isinstance(src_sents, list) or not isinstance(tgt_sents, list):
            raise ValueError("Each row must contain list fields: source_sentences, target_sentences")
        if len(src_sents) != len(tgt_sents):
            raise ValueError(f"Sentence length mismatch for doc_id={row.get('doc_id')}")
        if seg_ids is not None and isinstance(seg_ids, list) and len(seg_ids) != len(src_sents):
            raise ValueError(f"seg_ids length mismatch for doc_id={row.get('doc_id')}")

        n = len(src_sents)
        if n < window_size:
            skipped_short_docs += 1
            continue

        src_joiner = infer_joiner(row.get("source_text", ""), src_sents, "\n")
        tgt_joiner = infer_joiner(row.get("target_text", ""), tgt_sents, " ")

        for start in range(0, n - window_size + 1):
            end = start + window_size
            new_row = dict(row)

            new_src_sents = src_sents[start:end]
            new_tgt_sents = tgt_sents[start:end]
            new_seg_ids = seg_ids[start:end] if isinstance(seg_ids, list) else seg_ids

            chunk_tag = row.get("chunk_id")
            if chunk_tag is None:
                chunk_tag = f"row{row_idx}"
            suffix = f"__c{chunk_tag}_w{start + 1}-{end}"
            new_row["doc_id"] = f"{row.get('doc_id', '')}{suffix}"
            new_row["source_sentences"] = new_src_sents
            new_row["target_sentences"] = new_tgt_sents
            new_row["seg_ids"] = new_seg_ids
            new_row["num_sentences"] = window_size
            new_row["source_text"] = src_joiner.join(new_src_sents)
            new_row["target_text"] = tgt_joiner.join(new_tgt_sents)

            out_rows.append(new_row)

    return out_rows, skipped_short_docs


def parse_lang_pairs(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return ALL_LANG_PAIRS
    return [x.strip() for x in value.split(",") if x.strip()]


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + ".window2.jsonl")


def process_one_file(input_path: Path, output_path: Path, window_size: int) -> Tuple[int, int, int]:
    rows = load_jsonl(input_path)
    windowed_rows, skipped_short_docs = build_windowed_rows(rows, window_size=window_size)
    write_jsonl(output_path, windowed_rows)
    return len(rows), len(windowed_rows), skipped_short_docs


def process_lang_pairs(base_dir: Path, lang_pairs: List[str], window_size: int) -> None:
    for lang_pair in lang_pairs:
        input_path = (base_dir / lang_pair / f"test.{lang_pair}.jsonl").resolve()
        output_path = default_output_path(input_path)
        original_docs, new_docs, skipped_short_docs = process_one_file(
            input_path=input_path,
            output_path=output_path,
            window_size=window_size,
        )
        print(f"{lang_pair}:")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Original docs: {original_docs}")
        print(f"  New docs: {new_docs}")
        print(f"  Skipped docs (num_sentences < {window_size}): {skipped_short_docs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split document-level JSONL into sliding sentence windows."
    )
    parser.add_argument(
        "--lang-pairs",
        type=str,
        default="en-zh,en-ru,en-nl",
        help="Comma-separated language pairs or 'all'. Ignored when --input is set.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory containing <lang_pair>/test.<lang_pair>.jsonl",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single input JSONL path (optional). If set, only this file is processed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input>.window2.jsonl). Only used with --input.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Number of sentences in each new document window",
    )
    args = parser.parse_args()

    if args.input is not None:
        input_path = args.input.resolve()
        output_path = args.output.resolve() if args.output else default_output_path(input_path)
        original_docs, new_docs, skipped_short_docs = process_one_file(
            input_path=input_path,
            output_path=output_path,
            window_size=args.window_size,
        )
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Original docs: {original_docs}")
        print(f"New docs: {new_docs}")
        print(f"Skipped docs (num_sentences < {args.window_size}): {skipped_short_docs}")
        return

    base_dir = args.base_dir.resolve()
    lang_pairs = parse_lang_pairs(args.lang_pairs)
    process_lang_pairs(base_dir=base_dir, lang_pairs=lang_pairs, window_size=args.window_size)


if __name__ == "__main__":
    main()
