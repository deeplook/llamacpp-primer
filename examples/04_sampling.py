#!/usr/bin/env python3
"""Lesson 04: Temperature, top-p, top-k, and repeat penalty.

Shows how sampling parameters change the same prompt's output.
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
    prompt = "The meaning of life is"

    configs = [
        {"temperature": 0.0, "label": "greedy (temp=0)"},
        {"temperature": 0.8, "top_p": 0.9, "label": "temp=0.8, top_p=0.9"},
        {"temperature": 1.2, "top_k": 50, "label": "temp=1.2, top_k=50"},
        {"temperature": 0.7, "repeat_penalty": 1.5, "label": "temp=0.7, repeat_penalty=1.5"},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        output = llm(prompt, max_tokens=24, stop=["\n"], echo=False, **cfg)
        text = output["choices"][0]["text"].strip().replace("\n", " ")
        print(f"{label:<28} -> {text[:80]}")
        cfg["label"] = label  # restore for cleanliness


if __name__ == "__main__":
    main()
