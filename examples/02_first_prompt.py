#!/usr/bin/env python3
"""Lesson 02: Load a GGUF model and generate one completion.

Uses llama-cpp-python. Skips cleanly when no GGUF is found via Ollama.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"


def find_first_gguf() -> Path | None:
    """Return the first GGUF blob Ollama reports, or None."""
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
        sys.exit(0)  # skip, don't fail CI

    model_path = find_first_gguf()
    if model_path is None:
        print("No GGUF found via Ollama. Skipping.")
        print(f"Checked: {OLLAMA_BLOBS}")
        sys.exit(0)

    print(f"Loading: {model_path.name}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        verbose=False,
    )

    prompt = "The capital of France is"
    output = llm(prompt, max_tokens=16, stop=["\n"], echo=True)
    print(output["choices"][0]["text"])


if __name__ == "__main__":
    main()
