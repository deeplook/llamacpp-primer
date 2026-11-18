#!/usr/bin/env python3
"""Lesson 20: Capstone, server, tools, grammar, and RAG in one flow.

Retrieves a relevant chunk from the README (bag-of-words cosine), starts
llama-server with --jinja, lets the model call a weather tool, then asks for
a final answer grounded in the retrieved chunk and constrained by a GBNF
grammar to JSON with the keys "answer" and "weather". Skips without a GGUF
or llama-server binary.
"""

from __future__ import annotations

import json
import math
import re
import sys

from _common import ROOT, http_json, llama_server, require_binary, require_gguf

PORT = 19000

GRAMMAR = r"""root   ::= "{" ws "\"answer\"" ws ":" ws string "," ws "\"weather\"" ws ":" ws string ws "}"
string ::= "\"" ([^"\\\n] | "\\" ["\\/bfnrt])* "\""
ws     ::= [ \t\n]*
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    chunks = [text[i : i + size] for i in range(0, len(text), size - overlap)]
    return [c for c in chunks if c.strip()]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.hypot(*a), math.hypot(*b)
    return dot / (na * nb) if na and nb else 0.0


def bow_embed(texts: list[str]) -> list[list[float]]:
    def tokenize(t: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_-]{3,}", t.lower())

    vocab = sorted({tok for t in texts for tok in tokenize(t)})
    return [[float(tokenize(t).count(v)) for v in vocab] for t in texts]


def main() -> None:
    if require_binary("llama-server") is None:
        sys.exit(0)
    path = require_gguf()
    if path is None:
        sys.exit(0)

    # --- RAG: chunk README, embed with bag-of-words, retrieve best match ---
    chunks = chunk_text((ROOT / "README.md").read_text(errors="ignore"))
    query = "How do I get started with this primer?"
    vectors = bow_embed(chunks + [query])
    score, best = max((cosine(vectors[-1], v), i) for i, v in enumerate(vectors[:-1]))
    context = chunks[best]
    print(f"RAG: retrieved chunk {best} (score {score:.3f})")

    with llama_server(
        path, PORT, "--ctx-size", "2048", "--jinja", "-ngl", "99"
    ) as base:
        url = f"{base}/v1/chat/completions"
        print(f"Server up on port {PORT} with {path.name}")

        # --- Turn 1: tool call ---
        question = {"role": "user", "content": "What's the weather in Berlin?"}
        msg = http_json(
            url,
            {
                "model": path.name,
                "messages": [question],
                "tools": TOOLS,
                "max_tokens": 128,
            },
        )["choices"][0]["message"]
        messages = [question, msg]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            print(f"Tool call: {tc['function']['name']}({args})")
            weather = {"city": args["city"], "temp_c": 18, "condition": "cloudy"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", "call_0"),
                    "content": json.dumps(weather),
                }
            )
        else:
            print("No tool call returned (a small model may not support tools well).")

        # --- Turn 2: grammar-constrained answer grounded in the RAG context ---
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Answer using only this context:\n{context}\n\n"
                    "Question: how do I get started, and what's the weather? "
                    "Keep each field under 25 words."
                ),
            }
        )
        body = http_json(
            url,
            {
                "model": path.name,
                "messages": messages,
                "grammar": GRAMMAR,
                "max_tokens": 400,
            },
        )
    choice = body["choices"][0]
    content = choice["message"]["content"]
    assert choice["finish_reason"] == "stop", "output truncated by max_tokens"
    print("Grammar-constrained final answer:", content)
    parsed = json.loads(content)
    assert list(parsed) == ["answer", "weather"], parsed
    print("CAPSTONE PASSED")


if __name__ == "__main__":
    main()
