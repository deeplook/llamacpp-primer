#!/usr/bin/env python3
"""Lesson 02: Load a GGUF model and generate one completion.

Uses llama-cpp-python. Skips cleanly when no GGUF is found.
"""

from __future__ import annotations

import sys
from typing import Any, cast

from _common import require_gguf


def main() -> None:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("SKIP: llama-cpp-python not installed. Run: uv sync --all-groups")
        sys.exit(0)

    model_path = require_gguf()
    if model_path is None:
        sys.exit(0)

    print(f"Loading: {model_path.name}")
    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)

    prompt = "The capital of France is"
    output = cast(dict[str, Any], llm(prompt, max_tokens=16, stop=["\n"], echo=True))
    print(output["choices"][0]["text"])


if __name__ == "__main__":
    main()
