#!/usr/bin/env python3
"""Use vLLM + gemma-3-4b-it for simple translation."""

import argparse

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


def translate(
    llm: LLM,
    tokenizer: AutoTokenizer,
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_tokens: int,
    temperature: float,
) -> str:
    prompt = build_prompt(tokenizer, text, src_lang, tgt_lang)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    outputs = llm.generate([prompt], sampling)
    return outputs[0].outputs[0].text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate text using vLLM + Gemma 3 4B IT")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Hugging Face model id")
    parser.add_argument("--text", required=True, help="Input text to translate")
    parser.add_argument("--src-lang", default="Chinese", help="Source language")
    parser.add_argument("--tgt-lang", default="English", help="Target language")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    result = translate(
        llm=llm,
        tokenizer=tokenizer,
        text=args.text,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(result)


if __name__ == "__main__":
    main()