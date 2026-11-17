#!/usr/bin/env python3
"""Example 12: Constrain output with a JSON Schema.

llama-server accepts a response_format with json_schema, which it
compiles to a grammar internally. Skips without a GGUF or server binary.
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = {
    "name": "color_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "color": {"type": "string"},
            "hex": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
        },
        "required": ["color", "hex"],
        "additionalProperties": False,
    },
}


def find_gguf():
    for name in ("tinyllama-1.1b-chat-q4_k_m.gguf", "tinyllama-1.1b-chat-f16.gguf"):
        p = ROOT / "models" / name
        if p.exists():
            return p
    return None


def main():
    if shutil.which("llama-server") is None:
        print("SKIP: llama-server not on PATH.")
        return 0
    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found. Run examples 09/10 first.")
        return 0

    proc = subprocess.Popen(
        ["llama-server", "--model", str(path), "--port", "18081", "--ctx-size", "512"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:18081/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        req = urllib.request.Request(
            "http://127.0.0.1:18081/v1/chat/completions",
            data=json.dumps({
                "model": path.name,
                "messages": [{"role": "user", "content": "What color is grass?"}],
                "response_format": {"type": "json_schema", "json_schema": SCHEMA},
                "max_tokens": 64,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        content = body["choices"][0]["message"]["content"]
        print("Schema-constrained output:", content)
        parsed = json.loads(content)
        assert "color" in parsed and "hex" in parsed
        print("Validated against schema: color =", parsed["color"], ", hex =", parsed["hex"])
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
