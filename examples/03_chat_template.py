#!/usr/bin/env python3
"""Lesson 03: Apply a model's chat template to multi-turn messages.

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

    llm = Llama(model_path=str(model_path), n_ctx=1024, verbose=False)

    messages: list[Any] = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]

    # create_chat_completion applies the model's chat template automatically
    output = cast(
        dict[str, Any], llm.create_chat_completion(messages=messages, max_tokens=32)
    )
    print(output["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
