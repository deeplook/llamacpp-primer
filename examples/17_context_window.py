#!/usr/bin/env python3
"""Lesson 17: Context window, KV cache, and overflow behavior.

Reads the trained context length from the GGUF header, then starts
llama-server with a deliberately tiny --ctx-size and sends a prompt that
does not fit, to show the server rejecting it. Skips without a GGUF.
"""

from __future__ import annotations

import sys
import urllib.error

from _common import (
    context_length,
    http_json,
    llama_server,
    read_gguf_metadata,
    require_binary,
    require_gguf,
)

PORT = 18088
SMALL_CTX = 256


def main() -> None:
    path = require_gguf()
    if path is None:
        sys.exit(0)

    ctx = context_length(read_gguf_metadata(path))
    print(f"{path.name}: trained context length = {ctx} tokens")
    print("The KV cache grows with the context actually in use, so a longer")
    print("--ctx-size costs memory even when most of it stays empty.")
    if require_binary("llama-server") is None:
        sys.exit(0)

    with llama_server(path, PORT, "--ctx-size", str(SMALL_CTX), "-ngl", "99") as base:
        long_prompt = "The history of computing begins with calculators. " * 100
        try:
            http_json(
                f"{base}/v1/chat/completions",
                {
                    "model": path.name,
                    "messages": [{"role": "user", "content": long_prompt}],
                    "max_tokens": 8,
                },
            )
            print("Server accepted the prompt (it may have truncated it).")
        except urllib.error.HTTPError as err:
            print(
                f"--ctx-size {SMALL_CTX}: server rejected the long prompt ({err.code})."
            )


if __name__ == "__main__":
    main()
