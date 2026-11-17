#!/usr/bin/env python3
"""Example 09: Convert a Hugging Face model to GGUF.

Uses llama.cpp's convert_hf_to_gguf.py script. Requires the llama.cpp
source tree (cloned during CI or present locally). Skips if no source tree.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP = ROOT / "llama.cpp"
CONVERTER = LLAMA_CPP / "convert_hf_to_gguf.py"
OUT_DIR = ROOT / "models"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_GGUF = OUT_DIR / "tinyllama-1.1b-chat-f16.gguf"


def main():
    if not CONVERTER.exists():
        print(f"SKIP: {CONVERTER} not found. Clone llama.cpp into the repo root.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_GGUF.exists():
        print(f"SKIP: {OUT_GGUF} already exists")
        return 0

    cmd = [
        sys.executable, str(CONVERTER),
        MODEL_ID,
        "--outdir", str(OUT_DIR),
        "--outtype", "f16",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {OUT_GGUF} ({OUT_GGUF.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
