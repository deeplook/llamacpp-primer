#!/usr/bin/env python3
"""Example 19: Image input with a vision-capable GGUF.

llama.cpp supports multimodal models (e.g. LLaVA, Qwen-VL) where an
image is passed alongside text. This example sends a tiny generated PNG
plus a question and prints the model's description. Skips without a
vision GGUF, llama-server, or the mmproj file.
"""
import base64
import json
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 18087


def find_vision_gguf():
    models = ROOT / "models"
    if not models.exists():
        return None, None
    # Look for a mmproj alongside any GGUF
    for g in sorted(models.glob("*.gguf")):
        if "embed" in g.name.lower():
            continue
        mmproj = g.with_name(g.stem + "-mmproj.gguf")
        if not mmproj.exists():
            # common naming: model-name-mmproj.gguf next to model-name.gguf
            mmproj = models / (g.stem.replace("-q4_k_m", "").replace("-f16", "") + "-mmproj.gguf")
        if mmproj.exists():
            return g, mmproj
    return None, None


def make_tiny_png(path, color=(200, 40, 40)):
    """Write a minimal 8x8 PNG of a solid color."""
    w = h = 8
    raw = b"".join(b"\x00" + bytes(color) * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main():
    if shutil.which("llama-server") is None:
        print("SKIP: llama-server not on PATH.")
        return 0
    model, mmproj = find_vision_gguf()
    if model is None:
        print("SKIP: no vision GGUF with mmproj found in models/. "
              "Download e.g. llava-v1.5-7b-q4_k_m.gguf and its mmproj.")
        return 0

    img_path = ROOT / "models" / "tiny_red.png"
    make_tiny_png(img_path)
    b64 = base64.b64encode(img_path.read_bytes()).decode()

    cmd = ["llama-server", "--model", str(model), "--mmproj", str(mmproj),
           "--port", str(PORT), "--ctx-size", "2048"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")

        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({
                "model": model.name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }},
                    ],
                }],
                "max_tokens": 64,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.load(resp)
        print("Vision answer:", body["choices"][0]["message"]["content"])
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
