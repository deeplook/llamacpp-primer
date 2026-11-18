#!/usr/bin/env python3
"""Lesson 14: Generate embeddings with llama-server's /v1/embeddings.

Needs an embedding model in models/ with "embed" in its file name, e.g.
nomic-embed-text-v1.5.Q4_K_M.gguf. Skips without one or without the server.
"""

from __future__ import annotations

import math
import sys

from _common import http_json, llama_server, require_binary, require_gguf

PORT = 18083


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.hypot(*a) * math.hypot(*b))


def main() -> None:
    if require_binary("llama-server") is None:
        sys.exit(0)
    path = require_gguf(embedding=True)
    if path is None:
        print("      e.g. download nomic-embed-text-v1.5.Q4_K_M.gguf into models/")
        sys.exit(0)

    texts = ["Hello world", "Hi there, world", "Quarterly tax filing deadlines"]
    with llama_server(
        path, PORT, "--ctx-size", "512", "--embedding", "-ngl", "99"
    ) as base:
        body = http_json(f"{base}/v1/embeddings", {"model": path.name, "input": texts})
    vecs = [d["embedding"] for d in body["data"]]
    print(f"Embedding dim: {len(vecs[0])}")
    print(f"First 5 values: {vecs[0][:5]}")
    print(f"cos(hello, hi)  = {cosine(vecs[0], vecs[1]):.3f}")
    print(f"cos(hello, tax) = {cosine(vecs[0], vecs[2]):.3f}")


if __name__ == "__main__":
    main()
