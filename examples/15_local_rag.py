#!/usr/bin/env python3
"""Lesson 15: Minimal local RAG over a text file.

Chunks a document, embeds each chunk, retrieves the best match for a query
by cosine similarity, and asks the LLM to answer using it. Without an
embedding model it falls back to bag-of-words vectors, so the pipeline is
still demonstrable; without a chat GGUF the answer comes from the mock server.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from _common import (
    ROOT,
    find_gguf,
    http_json,
    llama_server,
    mock_server,
    require_binary,
)

DOC = ROOT / "README.md"
EMBED_PORT = 18084
CHAT_PORT = 18085


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    chunks = [text[i : i + size] for i in range(0, len(text), size - overlap)]
    return [c for c in chunks if c.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.hypot(*a), math.hypot(*b)
    return dot / (na * nb) if na and nb else 0.0


def bow_embed(texts: list[str]) -> list[list[float]]:
    """Token-count vectors over a shared vocabulary (no model needed)."""

    def tokenize(t: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_-]{3,}", t.lower())

    vocab = sorted({tok for t in texts for tok in tokenize(t)})
    return [[float(tokenize(t).count(v)) for v in vocab] for t in texts]


def retrieve(
    chunks: list[str], query: str, embed: Callable[[list[str]], list[list[float]]]
) -> tuple[float, int]:
    """Return (score, index) of the chunk most similar to the query."""
    vectors = embed(chunks + [query])
    qvec = vectors[-1]
    return max((cosine(qvec, v), i) for i, v in enumerate(vectors[:-1]))


def embed_texts(texts: list[str]) -> list[list[float]]:
    embed_gguf = find_gguf(embedding=True)
    if embed_gguf is None or require_binary("llama-server") is None:
        print("(no embedding model: using bag-of-words vectors)")
        return bow_embed(texts)
    with llama_server(
        embed_gguf, EMBED_PORT, "--ctx-size", "512", "--embedding", "-ngl", "99"
    ) as base:
        body = http_json(
            f"{base}/v1/embeddings", {"model": embed_gguf.name, "input": texts}
        )
    return [d["embedding"] for d in body["data"]]


def main() -> None:
    text = DOC.read_text(errors="ignore")
    chunks = chunk_text(text)
    print(f"Chunked {DOC.name} into {len(chunks)} pieces")
    query = "What is this primer about?"
    score, best = retrieve(chunks, query, embed_texts)
    print(f"Best chunk (score {score:.3f}):\n{chunks[best][:300]}...\n")

    prompt = f"Answer using only this context:\n{chunks[best]}\n\nQuestion: {query}"
    chat_gguf = find_gguf()
    if chat_gguf is not None and require_binary("llama-server") is not None:
        server = llama_server(chat_gguf, CHAT_PORT, "--ctx-size", "1024", "-ngl", "99")
        label, model = "LLM answer", chat_gguf.name
    else:
        print("(no chat model: answering from the mock server)")
        server = mock_server()
        label, model = "Mock answer", "mock-model"
    with server as base:
        body = http_json(
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
            },
        )
    print(f"{label}:", body["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
