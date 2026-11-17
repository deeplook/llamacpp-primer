#!/usr/bin/env python3
"""Lesson 07: Talk to llama-server from Python with the OpenAI SDK.

Expects llama-server on localhost:8080 (see example 06). Falls back to the
mock server on the same port if a real one isn't up. Skips if nothing responds.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8080/v1"


def server_up() -> bool:
    try:
        urllib.request.urlopen(f"{BASE_URL}/models", timeout=2)
        return True
    except Exception:
        return False


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai not installed. Run: uv add openai")
        sys.exit(0)

    mock = None
    if not server_up():
        print("No server on :8080. Starting mock server...")
        mock = subprocess.Popen(
            [sys.executable, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tools" / "mock_server.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(10):
            if server_up():
                break
            time.sleep(0.5)
        else:
            print("Mock server did not come up.")
            if mock:
                mock.terminate()
            sys.exit(0)

    try:
        client = OpenAI(base_url=BASE_URL, api_key="not-needed")
        models = client.models.list()
        print("Models:", [m.id for m in models.data])
        resp = client.chat.completions.create(
            model=models.data[0].id,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=16,
        )
        print(resp.choices[0].message.content)
    finally:
        if mock:
            mock.terminate()
            mock.wait(timeout=5)


if __name__ == "__main__":
    main()
