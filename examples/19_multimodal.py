#!/usr/bin/env python3
"""Example 19: Image input with a vision-capable GGUF.

llama.cpp supports multimodal models (e.g. LLaVA, Qwen-VL) where an
image is passed alongside text. This example sends a tiny generated PNG
plus a question and prints the model's description. Skips without a
vision GGUF, llama-server, or the mmproj file.
"""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from pathlib import Path

from _common import MODELS_DIR, http_json, llama_server, require_binary

PORT = 18087


def find_vision_gguf() -> tuple[Path, Path] | tuple[None, None]:
    models = MODELS_DIR
    if not models.exists():
        return None, None
    # Look for a mmproj alongside any GGUF
    for g in sorted(models.glob("*.gguf")):
        if "embed" in g.name.lower() or "mmproj" in g.name.lower():
            continue
        mmproj = g.with_name(g.stem + "-mmproj.gguf")
        if not mmproj.exists():
            # common naming: model-name-mmproj.gguf next to model-name.gguf
            mmproj = models / (
                g.stem.replace("-q4_k_m", "").replace("-f16", "") + "-mmproj.gguf"
            )
        if mmproj.exists():
            return g, mmproj
    return None, None


def make_tiny_png(path: Path, color: tuple[int, int, int] = (200, 40, 40)) -> None:
    """Write a minimal 8x8 PNG of a solid color."""
    w = h = 8
    raw = b"".join(b"\x00" + bytes(color) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> None:
    if require_binary("llama-server") is None:
        sys.exit(0)
    model, mmproj = find_vision_gguf()
    if model is None or mmproj is None:
        print(
            "SKIP: no vision GGUF with mmproj found in models/. "
            "Download e.g. llava-v1.5-7b-q4_k_m.gguf and its mmproj."
        )
        sys.exit(0)

    img_path = MODELS_DIR / "tiny_red.png"
    make_tiny_png(img_path)
    b64 = base64.b64encode(img_path.read_bytes()).decode()

    with llama_server(
        model, PORT, "--mmproj", str(mmproj), "--ctx-size", "2048", startup_timeout=120
    ) as base:
        body = http_json(
            f"{base}/v1/chat/completions",
            {
                "model": model.name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 64,
            },
        )
    print("Vision answer:", body["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
