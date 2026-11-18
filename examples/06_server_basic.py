#!/usr/bin/env python3
"""Lesson 06: Start llama-server and call its OpenAI-compatible endpoint.

If llama-server is already running on the port it is used as is. Otherwise
it is started against the first GGUF found (or the lesson skips if none).
"""

from __future__ import annotations

import sys
import urllib.request
from contextlib import AbstractContextManager, nullcontext

from _common import http_json, llama_server, require_binary, require_gguf

PORT = 8080
BASE = f"http://127.0.0.1:{PORT}"


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def main() -> None:
    ctx: AbstractContextManager[str]
    if server_up():
        print("llama-server already running.")
        ctx = nullcontext(BASE)
    else:
        if require_binary("llama-server") is None:
            sys.exit(0)
        model_path = require_gguf()
        if model_path is None:
            sys.exit(0)
        print(f"Starting llama-server with {model_path.name}...")
        ctx = llama_server(model_path, PORT, "-ngl", "99")

    with ctx as base:
        data = http_json(
            f"{base}/v1/chat/completions",
            {
                "model": "local",
                "messages": [{"role": "user", "content": "Say hi in one word."}],
                "max_tokens": 16,
            },
        )
    reply = data["choices"][0]["message"]["content"]
    assert reply.strip(), "empty reply"
    print(reply)


if __name__ == "__main__":
    main()
