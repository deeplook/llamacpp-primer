#!/usr/bin/env python3
"""Lesson 05: Stream tokens as they are generated.

Uses llama-cpp-python's create_completion with stream=True.
Skips cleanly when no GGUF is found.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
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

    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)

    print("Streaming: ", end="", flush=True)
    stream = cast(
        Iterator[dict[str, Any]],
        llm("Count from one to five: ", max_tokens=32, stream=True),
    )
    for chunk in stream:
        print(chunk["choices"][0].get("text", ""), end="", flush=True)
    print()  # newline after stream ends


if __name__ == "__main__":
    main()
