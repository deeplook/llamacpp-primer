#!/usr/bin/env python3
"""Example 10: Quantize a GGUF file with llama-quantize.

Takes an F16 GGUF and produces a 4-bit Q4_K_M version. Skips if the
input GGUF or llama-quantize binary is missing.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "tinyllama-1.1b-chat-f16.gguf"
DST = ROOT / "models" / "tinyllama-1.1b-chat-q4_k_m.gguf"
QUANTIZE = shutil.which("llama-quantize")


def main():
    if not SRC.exists():
        print(f"SKIP: {SRC} not found. Run example 09 first.")
        return 0
    if QUANTIZE is None:
        print("SKIP: llama-quantize not on PATH. Install llama.cpp.")
        return 0
    if DST.exists():
        print(f"SKIP: {DST} already exists")
        return 0

    cmd = [QUANTIZE, str(SRC), str(DST), "Q4_K_M"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {DST} ({DST.stat().st_size / 1e6:.0f} MB, "
          f"{100 * DST.stat().st_size / SRC.stat().st_size:.0f}% of original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
