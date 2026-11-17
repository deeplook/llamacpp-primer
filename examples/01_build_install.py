#!/usr/bin/env python3
"""Lesson 01: Build from source or install a binary; check Metal/CUDA support."""

from __future__ import annotations

import shutil
import subprocess
import sys


def check_binary(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return None
    try:
        result = subprocess.run(
            [name, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except (subprocess.SubprocessError, IndexError):
        return path


def main() -> None:
    print("Checking for llama.cpp binaries...\n")

    for binary in ("llama-server", "llama-cli"):
        info = check_binary(binary)
        if info:
            print(f"  [OK] {binary}: {info}")
        else:
            print(f"  [MISSING] {binary}")
            print(f"           Install with: brew install llama.cpp")

    print()
    # Probe compile-time backend support via llama-server --help text
    server = shutil.which("llama-server")
    if server:
        try:
            result = subprocess.run(
                [server, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            help_text = (result.stdout + result.stderr).lower()
            for backend in ("metal", "cuda", "vulkan", "blas"):
                if backend in help_text:
                    print(f"  Backend hint: {backend} appears in --help output")
        except subprocess.SubprocessError:
            pass

    print("\nTip: on macOS, Metal is used automatically on Apple Silicon.")
    print("     On Linux with an NVIDIA GPU, build with CUDA support.")


if __name__ == "__main__":
    main()
