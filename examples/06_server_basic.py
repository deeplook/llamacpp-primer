#!/usr/bin/env python3
"""Lesson 06: Start llama-server and call its OpenAI-compatible endpoint.

If llama-server is not running, this script starts it against the first
GGUF found via Ollama (or skips if none). It then hits /v1/chat/completions.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"
PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"


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


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def main() -> None:
    proc = None
    if not server_up():
        model_path = find_first_gguf()
        if model_path is None:
            print("No GGUF found via Ollama and no server running. Skipping.")
            sys.exit(0)
        print(f"Starting llama-server with {model_path.name}...")
        proc = subprocess.Popen(
            ["llama-server", "--model", str(model_path), "--port", str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(60):
            if server_up():
                break
            time.sleep(1)
        else:
            print("llama-server did not come up in time.")
            if proc:
                proc.terminate()
            sys.exit(1)
    else:
        print("llama-server already running.")

    try:
        req = urllib.request.Request(
            f"{BASE}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "local",
                    "messages": [{"role": "user", "content": "Say hi in one word."}],
                    "max_tokens": 16,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        print(data["choices"][0]["message"]["content"])
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
