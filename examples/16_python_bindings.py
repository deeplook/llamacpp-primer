#!/usr/bin/env python3
"""Lesson 16: llama-cpp-python, direct model access without a server.

Loads a GGUF with Llama(), generates a completion, then streams tokens
to show per-token access. Skips without llama-cpp-python or a GGUF.
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

    path = require_gguf()
    if path is None:
        sys.exit(0)

    llm = Llama(model_path=str(path), n_ctx=512, n_gpu_layers=-1, verbose=False)
    out = cast(
        dict[str, Any],
        llm.create_completion(
            "The capital of France is", max_tokens=16, temperature=0.2
        ),
    )
    print("Completion:", out["choices"][0]["text"].strip())

    # Tokenizer round-trip
    text = b"Hello, llama.cpp!"
    ids = llm.tokenize(text, add_bos=False)
    print(f"Tokens ({len(ids)}):", ids)
    back = llm.detokenize(ids)
    print("Detokenized:", back.decode())
    assert back == text, f"round trip changed the text: {back!r}"

    # Per-token streaming
    stream = cast(
        Iterator[dict[str, Any]],
        llm.create_completion(
            "Count to five:", max_tokens=16, temperature=0.0, stream=True
        ),
    )
    pieces = [chunk["choices"][0]["text"] for chunk in stream]
    print("Streamed pieces:", pieces)


if __name__ == "__main__":
    main()
