#!/usr/bin/env python3
"""Example 18: Reuse a stable prompt prefix across generations.

llama-server's -fa (flash attention) and the default KV cache keep a
processed prefix hot. This example sends the same long prefix twice with
different suffixes and shows the second call is faster because the prefix
was already in cache. Skips without llama-server or a GGUF.
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 18086

PREFIX = (
    "You are a helpful assistant. Always answer concisely. "
    "Here is a long document for reference: "
    + "The history of computing begins with mechanical calculators. " * 40
)


def find_gguf():
    models = ROOT / "models"
    if not models.exists():
        return None
    ggufs = sorted(models.glob("*.gguf"))
    return next((g for g in ggufs if "embed" not in g.name.lower()), None)


def timed_call(suffix):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({
            "model": "x",
            "messages": [
                {"role": "user", "content": PREFIX + suffix},
            ],
            "max_tokens": 32,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.load(resp)
    return time.time() - t0, body["choices"][0]["message"]["content"]


def main():
    if shutil.which("llama-server") is None:
        print("SKIP: llama-server not on PATH.")
        return 0
    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found. Run examples 09/10 first.")
        return 0

    proc = subprocess.Popen(
        ["llama-server", "--model", str(path), "--port", str(PORT),
         "--ctx-size", "2048"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")

        t1, a1 = timed_call("Question one: what is 2+2?")
        print(f"Call 1 (cold prefix): {t1:.2f}s -> {a1.strip()[:60]}")
        t2, a2 = timed_call("Question two: what is 3+3?")
        print(f"Call 2 (warm prefix):  {t2:.2f}s -> {a2.strip()[:60]}")
        if t2 < t1:
            print(f"Speedup: {t1/t2:.1f}x — prefix was cached.")
        else:
            print("No speedup observed; cache may not have hit "
                  "(short runs or small models can mask it).")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
