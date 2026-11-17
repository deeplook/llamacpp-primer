#!/usr/bin/env python3
"""Lesson 05: Stream tokens as they are generated.

Uses llama-cpp-python's create_completion with stream=True.
Skips cleanly when no GGUF is found via Ollama.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"


def find_first_gguf() -> Path | None:
    if not OLLAMA_BLOBS.exists():
        return None
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            model = json.loads(line)
        except json.JSONDecodeError:
            continue
        digest = model.get("digest", "")
        if digest.startswith("sha256:"):
            blob = OLLAMA_BLOBS / ("sha256-" + digest.split(":", 1)[1])
            if blob.exists():
                return blob
    return None


def main() -> None:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Run: uv add llama-cpp-python")
        sys.exit(0)

    model_path = find_first_gguf()
    if model_path is None:
        print("No GGUF found via Ollama. Skipping.")
        sys.exit(0)

    llm = Llama(model_path=str(model_path), n_ctx=512, verbose=False)

    print("Streaming: ", end="", flush=True)
    for chunk in llm(
        "Count from one to five: ",
        max_tokens=32,
        stream=True,
        echo=False,
    ):
        token = chunk["choices"][0].get("text", "")
        print(token, end="", flush=True)
    print()  # newline after stream ends


if __name__ == "__main__":
    main()
