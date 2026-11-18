#!/usr/bin/env python3
"""Verify the default model against the Hugging Face Hub (needs network).

Confirms the exact GGUF that `make model-download` fetches still exists and
has the expected size, and checks a local copy against it if present.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
FILE = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
SIZE = 397_808_192
LOCAL = Path(__file__).resolve().parents[1] / "models" / FILE


def main() -> int:
    url = f"https://huggingface.co/api/models/{REPO}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as resp:
        items = json.load(resp)
    remote = next(i for i in items if i.get("path") == FILE)
    if remote["size"] != SIZE:
        print(f"FAIL: Hub reports {remote['size']} bytes, expected {SIZE}")
        return 1
    print(f"OK: {REPO}/{FILE} ({SIZE} bytes)")
    if LOCAL.exists():
        ok = LOCAL.stat().st_size == SIZE
        print(f"{'OK' if ok else 'FAIL'}: local copy {LOCAL.stat().st_size} bytes")
        return 0 if ok else 1
    print("Local copy missing; run: make model-download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
