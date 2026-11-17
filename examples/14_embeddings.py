#!/usr/bin/env python3
"""Example 14: Generate embeddings with llama-server's /v1/embeddings.

Requires a model built with embedding support (e.g. all-MiniLM or a
nomic embed model in GGUF). Skips without a GGUF or server binary.
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_gguf():
    models = ROOT / "models"
    if not models.exists():
        return None
    # Prefer anything with 'embed' in the name, else first GGUF
    ggufs = sorted(models.glob("*.gguf"))
    for g in ggufs:
        if "embed" in g.name.lower():
            return g
    return ggufs[0] if ggufs else None


def main():
    if shutil.which("llama-server") is None:
        print("SKIP: llama-server not on PATH.")
        return 0
    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found in models/. "
              "Download an embedding model, e.g. "
              "nomic-embed-text-v1.5.Q4_K_M.gguf")
        return 0

    proc = subprocess.Popen(
        ["llama-server", "--model", str(path), "--port", "18083", "--ctx-size", "512", "--embedding"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:18083/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")

        req = urllib.request.Request(
            "http://127.0.0.1:18083/v1/embeddings",
            data=json.dumps({"model": path.name, "input": "Hello world"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        emb = body["data"][0]["embedding"]
        print(f"Embedding dim: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
