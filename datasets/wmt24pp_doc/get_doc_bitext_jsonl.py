from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

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


def create_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_lang_pairs(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ALL_LANG_PAIRS
    return [x.strip() for x in value.split(",") if x.strip()]


def load_parallel_sentence_map(base_dir: Path, lang_pair: str) -> dict[str, str]:
    src_lang, tgt_lang = lang_pair.split("-")
    src_path = base_dir / ".." / "wmt24pp" / "test" / lang_pair / f"test.{lang_pair}.{src_lang}"
    tgt_path = base_dir / ".." / "wmt24pp" / "test" / lang_pair / f"test.{lang_pair}.{tgt_lang}"

    bitext_map: dict[str, str] = {}
    with src_path.open("r", encoding="utf-8") as src_fin, tgt_path.open("r", encoding="utf-8") as tgt_fin:
        for src_line, tgt_line in zip(src_fin, tgt_fin):
            src_line = src_line.strip()
            tgt_line = tgt_line.strip()
            bitext_map[src_line] = tgt_line
    return bitext_map


def flush_chunk(
    jsonl_fout,
    src_fout,
    tgt_fout,
    lang_pair: str,
    doc_id: str,
    chunk_idx: int,
    seg_ids: list[str],
    src_sentences: list[str],
    tgt_sentences: list[str],
) -> None:
    if not src_sentences:
        return

    src_doc = " ".join(src_sentences)
    tgt_doc = " ".join(tgt_sentences)
    src_fout.write(src_doc + "\n")
    tgt_fout.write(tgt_doc + "\n")

    row = {
        "lang_pair": lang_pair,
        "doc_id": doc_id,
        "chunk_id": chunk_idx,
        "num_sentences": len(src_sentences),
        "seg_ids": seg_ids,
        "source_sentences": src_sentences,
        "target_sentences": tgt_sentences,
        "source_text": src_doc,
        "target_text": tgt_doc,
    }
    jsonl_fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_doc_bitext(base_dir: Path, lang_pair: str, max_src_words: int) -> int:
    out_dir = base_dir / lang_pair
    create_clean_dir(out_dir)

    bitext_map = load_parallel_sentence_map(base_dir, lang_pair)

    # WMT24 document metadata uses en-zh XML ids for all language pairs.
    root = ET.parse(base_dir / "raw" / "wmttest2024.src.en-zh.xml").getroot()
    num_chunks = 0

    src_lang, tgt_lang = lang_pair.split("-")
    src_out = out_dir / f"test.{lang_pair}.{src_lang}"
    tgt_out = out_dir / f"test.{lang_pair}.{tgt_lang}"
    jsonl_out = out_dir / f"test.{lang_pair}.jsonl"

    with src_out.open("w", encoding="utf-8") as src_fout, \
        tgt_out.open("w", encoding="utf-8") as tgt_fout, \
        jsonl_out.open("w", encoding="utf-8") as jsonl_fout:
        for doc in root.findall(".//doc"):
            doc_id = doc.get("id", "")
            chunk_idx = 0
            word_count = 0
            seg_ids: list[str] = []
            src_sentences: list[str] = []
            tgt_sentences: list[str] = []

            for seg in doc.findall(".//seg"):
                seg_text = (seg.text or "").strip()
                if not seg_text or seg_text not in bitext_map:
                    continue

                seg_word_count = len(seg_text.split())
                if src_sentences and word_count + seg_word_count >= max_src_words:
                    flush_chunk(
                        jsonl_fout,
                        src_fout,
                        tgt_fout,
                        lang_pair,
                        doc_id,
                        chunk_idx,
                        seg_ids,
                        src_sentences,
                        tgt_sentences,
                    )
                    num_chunks += 1
                    chunk_idx += 1
                    word_count = 0
                    seg_ids = []
                    src_sentences = []
                    tgt_sentences = []

                seg_ids.append(seg.get("id", ""))
                src_sentences.append(seg_text)
                tgt_sentences.append(bitext_map[seg_text])
                word_count += seg_word_count

            if src_sentences:
                flush_chunk(
                    jsonl_fout,
                    src_fout,
                    tgt_fout,
                    lang_pair,
                    doc_id,
                    chunk_idx,
                    seg_ids,
                    src_sentences,
                    tgt_sentences,
                )
                num_chunks += 1

    return num_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build WMT24++ document-level bitext and save both txt and jsonl."
    )
    parser.add_argument(
        "--lang-pairs",
        type=str,
        default="en-nl,en-zh,en-ru",
        help="Comma-separated language pairs or 'all'.",
    )
    parser.add_argument(
        "--max-src-words",
        type=int,
        default=150,
        help="Max source words per chunk before splitting.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    lang_pairs = parse_lang_pairs(args.lang_pairs)
    for lang_pair in lang_pairs:
        chunks = build_doc_bitext(base_dir, lang_pair, args.max_src_words)
        print(f"{lang_pair}: wrote {chunks} doc chunks to {base_dir / lang_pair}")


if __name__ == "__main__":
    main()
