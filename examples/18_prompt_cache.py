#!/usr/bin/env python3
"""Lesson 18: Reuse a stable prompt prefix across generations.

llama-server keeps the KV cache of the previous request per slot and reuses
the longest common prefix. This lesson sends the same long prefix twice with
different suffixes; the second call should be faster and report cached
tokens. Skips without llama-server or a GGUF.
"""

from __future__ import annotations

import sys
import time

from _common import http_json, llama_server, require_binary, require_gguf

PORT = 18086

PREFIX = (
    "You are a helpful assistant. Always answer concisely. "
    "Here is a long document for reference: "
    + "The history of computing begins with mechanical calculators. "
    * 40
)


def timed_call(base: str, model: str, suffix: str) -> tuple[float, str, int]:
    t0 = time.perf_counter()
    body = http_json(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": PREFIX + suffix}],
            "max_tokens": 32,
            "cache_prompt": True,
        },
    )
    elapsed = time.perf_counter() - t0
    cached = body.get("timings", {}).get("cache_n", 0)
    return elapsed, body["choices"][0]["message"]["content"], cached


def main() -> None:
    if require_binary("llama-server") is None:
        sys.exit(0)
    path = require_gguf()
    if path is None:
        sys.exit(0)

    with llama_server(path, PORT, "--ctx-size", "2048", "-ngl", "99") as base:
        t1, a1, c1 = timed_call(base, path.name, "Question one: what is 2+2?")
        print(
            f"Call 1 (cold prefix): {t1:.2f}s, {c1} cached tokens -> {a1.strip()[:60]}"
        )
        t2, a2, c2 = timed_call(base, path.name, "Question two: what is 3+3?")
        print(
            f"Call 2 (warm prefix): {t2:.2f}s, {c2} cached tokens -> {a2.strip()[:60]}"
        )
    if c2 > c1:
        print(f"Prefix reuse confirmed: {c2} tokens skipped on the second call.")
    else:
        print("No cache hit reported; check the server version's /timings output.")


if __name__ == "__main__":
    main()
