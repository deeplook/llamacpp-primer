#!/usr/bin/env python3
"""Example 15: Minimal local RAG over a text file.

Chunks a document, embeds each chunk, retrieves the best match for a
query by cosine similarity, and asks the LLM to answer using it.
Uses the mock server if no embedding model is available, falling back
to a trivial keyword scorer so the pipeline is still demonstrable.
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
DOC = ROOT / "README.md"


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


def embed_texts(texts, port=18084):
    """Try llama-server embeddings; fall back to bag-of-words vectors."""
    if shutil.which("llama-server") is not None:
        ggufs = sorted((ROOT / "models").glob("*.gguf")) if (ROOT / "models").exists() else []
        embed_gguf = next((g for g in ggufs if "embed" in g.name.lower()), None)
        if embed_gguf is not None:
            proc = subprocess.Popen(
                ["llama-server", "--model", str(embed_gguf), "--port", str(port),
                 "--ctx-size", "512", "--embedding"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(60):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise RuntimeError("embed server did not start")
                out = []
                for t in texts:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/embeddings",
                        data=json.dumps({"model": embed_gguf.name, "input": t}).encode(),
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        out.append(json.load(resp)["data"][0]["embedding"])
                return out
            finally:
                proc.terminate()
                proc.wait(timeout=10)
    # Fallback: simple token-frequency vectors over a shared vocab
    vocab = sorted(set(re.findall(r"[a-zA-Z0-9_-]{3,}", " ".join(texts).lower())))
    def vec(t):
        toks = re.findall(r"[a-zA-Z0-9_-]{3,}", t.lower())
        return [toks.count(v) for v in vocab]
    return [vec(t) for t in texts]


def main():
    if not DOC.exists():
        print(f"SKIP: {DOC} not found")
        return 0
    text = DOC.read_text(errors="ignore")
    chunks = chunk_text(text)
    print(f"Chunked {DOC.name} into {len(chunks)} pieces")
    query = "What is this primer about?"
    vectors = embed_texts(chunks + [query])
    chunk_vecs, qvec = vectors[:-1], vectors[-1]
    scored = sorted(((cosine(qvec, v), i) for i, v in enumerate(chunk_vecs)), reverse=True)
    best_i = scored[0][1]
    print(f"Best chunk (score {scored[0][0]:.3f}):\n{chunks[best_i][:300]}...\n")
    # Ask the mock or real server to answer from the chunk
    answer_prompt = (f"Answer using only this context:\n{chunks[best_i]}\n\nQuestion: {query}")
    if shutil.which("llama-server") is not None:
        ggufs = sorted((ROOT / "models").glob("*.gguf")) if (ROOT / "models").exists() else []
        chat_gguf = next((g for g in ggufs if "embed" not in g.name.lower()), None)
        if chat_gguf is not None:
            proc = subprocess.Popen(
                ["llama-server", "--model", str(chat_gguf), "--port", "18085", "--ctx-size", "1024"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                for _ in range(60):
                    try:
                        urllib.request.urlopen("http://127.0.0.1:18085/v1/models", timeout=1)
                        break
                    except Exception:
                        time.sleep(1)
                req = urllib.request.Request(
                    "http://127.0.0.1:18085/v1/chat/completions",
                    data=json.dumps({"model": chat_gguf.name,
                                     "messages": [{"role": "user", "content": answer_prompt}],
                                     "max_tokens": 128}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    print("LLM answer:", json.load(resp)["choices"][0]["message"]["content"])
            finally:
                proc.terminate()
                proc.wait(timeout=10)
            return 0
    # Mock fallback
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "mock_server.py")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps({"model": "mock-model",
                             "messages": [{"role": "user", "content": answer_prompt}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("Mock answer:", json.load(resp)["choices"][0]["message"]["content"])
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
