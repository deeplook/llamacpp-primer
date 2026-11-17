#!/usr/bin/env python3
"""Example 16: llama-cpp-python — direct model access without a server.

Loads a GGUF with Llama(), generates with n_predict, and shows batching
plus a token callback. Skips without llama-cpp-python or a GGUF.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_gguf():
    models = ROOT / "models"
    if not models.exists():
        return None
    ggufs = sorted(models.glob("*.gguf"))
    return next((g for g in ggufs if "embed" not in g.name.lower()), None)


def main():
    try:
        from llama_cpp import Llama
    except ImportError:
        print("SKIP: llama-cpp-python not installed. uv sync --all-groups")
        return 0

    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found. Run examples 09/10 first.")
        return 0

    llm = Llama(model_path=str(path), n_ctx=512, verbose=False)
    out = llm.create_completion(
        "The capital of France is",
        max_tokens=16,
        temperature=0.2,
    )
    print("Completion:", out["choices"][0]["text"].strip())

    # Batching: multiple prompts in one call
    batch = llm.create_completion(
        ["Say hi in German.", "Say hi in French."],
        max_tokens=8,
        temperature=0.0,
    )
    for i, choice in enumerate(batch["choices"]):
        print(f"Batch {i}:", choice["text"].strip())

    # Token callback for streaming-like visibility
    tokens = []
    llm.create_completion(
        "Count to five:",
        max_tokens=16,
        temperature=0.0,
        callback=lambda t: tokens.append(t),
    )
    print("Callback tokens:", tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
