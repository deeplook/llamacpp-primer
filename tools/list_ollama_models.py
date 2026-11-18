#!/usr/bin/env python3
"""List GGUF models available via Ollama's local blob store."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

OLLAMA_DIR = Path.home() / ".ollama" / "models"
BLOBS_DIR = OLLAMA_DIR / "blobs"


def run_ollama_list() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("ollama not found on PATH", file=sys.stderr)
        return []
    except subprocess.CalledProcessError as exc:
        print(f"ollama list failed: {exc.stderr.strip()}", file=sys.stderr)
        return []

    models = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            models.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return models


def digest_to_blob(digest: str) -> Path | None:
    # digests look like "sha256:<hex>"; blobs are named "sha256-<hex>"
    if digest.startswith("sha256:"):
        name = "sha256-" + digest.split(":", 1)[1]
        path = BLOBS_DIR / name
        return path if path.exists() else None
    return None


def main() -> None:
    if not BLOBS_DIR.exists():
        print(f"No Ollama blob store at {BLOBS_DIR}")
        sys.exit(0)

    models = run_ollama_list()
    if not models:
        print("No models reported by ollama list.")
        sys.exit(0)

    print(f"{'MODEL':<30} {'SIZE':>10} {'BLOB':<20} STATUS")
    for model in models:
        name = model.get("name", "?")
        size = model.get("size", 0)
        digest = model.get("digest", "")
        blob = digest_to_blob(digest)
        status = "OK" if blob else "blob missing"
        size_mb = size / (1024 * 1024) if size else 0
        print(
            f"{name:<30} {size_mb:>9.0f}MB {str(blob.name) if blob else '-':<20} {status}"
        )


if __name__ == "__main__":
    main()
