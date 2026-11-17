#!/usr/bin/env python3
"""Example 20: Capstone — server, tools, grammar, and RAG in one flow.

Starts llama-server on a quantized GGUF, registers a weather tool,
enables a GBNF grammar forcing JSON output, retrieves a relevant chunk
from the README via cosine similarity, and asks the model to answer
using the chunk while calling the tool and emitting grammar-constrained
JSON. Skips without a GGUF or llama-server binary.
"""
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 19000

GRAMMAR = """root   ::= object
object ::= "{" ws pair ("," ws pair)* "}" ws
pair   ::= string ":" ws value
value  ::= object | array | string | number | ("true" | "false" | "null") ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" [""\\bfnrt] | "\\u" [0-9a-fA-F]{4})* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= ([ \t\n\r])*"""

TOOLS = [{
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
}]


def find_gguf():
    models = ROOT / "models"
    if not models.exists():
        return None
    ggufs = sorted(models.glob("*.gguf"))
    return next((g for g in ggufs if "embed" not in g.name.lower()), None)


def chunk_text(text, size=400, overlap=80):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def bow_embed(texts):
    vocab = sorted(set(re.findall(r"[a-zA-Z0-9_-]{3,}", " ".join(texts).lower())))
    def vec(t):
        toks = re.findall(r"[a-zA-Z0-9_-]{3,}", t.lower())
        return [toks.count(v) for v in vocab]
    return [vec(t) for t in texts]


def post(url, payload, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main():
    if shutil.which("llama-server") is None:
        print("SKIP: llama-server not on PATH.")
        return 0
    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found. Run examples 09/10 first.")
        return 0

    # --- RAG: chunk README, embed with bag-of-words, retrieve best match ---
    doc = ROOT / "README.md"
    if not doc.exists():
        print("SKIP: README.md not found for RAG step")
        return 0
    chunks = chunk_text(doc.read_text(errors="ignore"))
    query = "How do I get started with this primer?"
    vectors = bow_embed(chunks + [query])
    best_i = max(range(len(chunks)), key=lambda i: cosine(vectors[-1], vectors[i]))
    context = chunks[best_i]
    print(f"RAG: retrieved chunk {best_i} (score {cosine(vectors[-1], vectors[best_i]):.3f})")

    # --- Start server ---
    proc = subprocess.Popen(
        ["llama-server", "--model", str(path), "--port", str(PORT),
         "--ctx-size", "2048"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        print(f"Server up on port {PORT} with {path.name}")

        # --- Turn 1: tool call ---
        body = post(f"http://127.0.0.1:{PORT}/v1/chat/completions", {
            "model": path.name,
            "messages": [{"role": "user", "content": "What's the weather in Berlin?"}],
            "tools": TOOLS,
            "max_tokens": 128,
        })
        msg = body["choices"][0]["message"]
        messages = [{"role": "user", "content": "What's the weather in Berlin?"}, msg]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            print(f"Tool call: {tc['function']['name']}({args})")
            weather = {"city": args["city"], "temp_c": 18, "condition": "cloudy"}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(weather)})

        # --- Turn 2: grammar-constrained answer grounded in RAG context ---
        prompt = (f"Answer using only this context:\n{context}\n\n"
                  f"Question: How do I get started, and what's the weather?")
        messages.append({"role": "user", "content": prompt})
        body = post(f"http://127.0.0.1:{PORT}/v1/chat/completions", {
            "model": path.name,
            "messages": messages,
            "grammar": GRAMMAR,
            "max_tokens": 256,
        })
        content = body["choices"][0]["message"]["content"]
        print("Grammar-constrained final answer:", content)
        parsed = json.loads(content)
        print("Validated JSON keys:", list(parsed.keys()))
        print("CAPSTONE PASSED")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
