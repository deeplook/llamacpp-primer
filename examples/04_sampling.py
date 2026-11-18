#!/usr/bin/env python3
"""Lesson 04: Temperature, top-p, top-k, and repeat penalty.

Shows how sampling parameters change the same prompt's output.
Skips cleanly when no GGUF is found.
"""

from __future__ import annotations

import sys
from typing import Any, cast

from _common import require_gguf

CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("greedy (temp=0)", {"temperature": 0.0}),
    ("temp=0.8, top_p=0.9", {"temperature": 0.8, "top_p": 0.9}),
    ("temp=1.2, top_k=50", {"temperature": 1.2, "top_k": 50}),
    ("temp=0.7, repeat_penalty=1.5", {"temperature": 0.7, "repeat_penalty": 1.5}),
]


def main() -> None:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("SKIP: llama-cpp-python not installed. Run: uv sync --all-groups")
        sys.exit(0)

    model_path = require_gguf()
    if model_path is None:
        sys.exit(0)

    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)
    prompt = "The meaning of life is"

    for label, params in CONFIGS:
        output = cast(
            dict[str, Any],
            llm(prompt, max_tokens=24, stop=["\n"], echo=False, **params),
        )
        text = output["choices"][0]["text"].strip().replace("\n", " ")
        print(f"{label:<28} -> {text[:80]}")
        if params["temperature"] == 0.0:
            again = cast(
                dict[str, Any],
                llm(prompt, max_tokens=24, stop=["\n"], echo=False, **params),
            )
            assert again["choices"][0]["text"].strip().replace("\n", " ") == text
            print(f"{'':<28}    (greedy decoding repeats exactly)")


if __name__ == "__main__":
    main()
