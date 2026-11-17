#!/usr/bin/env python3
"""Example 11: Constrain output with a GBNF grammar.

Forces the model to emit a JSON object with fixed keys. Works with
llama-server's grammar parameter or llama-cli's --grammar flag.
Skips if no GGUF is available.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "models" / "tinyllama-1.1b-chat-q4_k_m.gguf"
if not GGUF.exists():
    GGUF = ROOT / "models" / "tinyllama-1.1b-chat-f16.gguf"

GRAMMAR = """root   ::= object
object ::= "{" ws pair ("," ws pair)* "}" ws
pair   ::= string ":" ws value
value  ::= object | array | string | number | ("true" | "false" | "null") ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" [""\\bfnrt] | "\\u" [0-9a-fA-F]{4})* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= ([ \t\n\r])*"""


def find_gguf():
    for p in [GGUF, ROOT / "models" / "tinyllama-1.1b-chat-f16.gguf"]:
        if p.exists():
            return p
    return None


def via_server(path):
    proc = subprocess.Popen(
        ["llama-server", "--model", str(path), "--port", "18080", "--ctx-size", "512"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:18080/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        req = urllib.request.Request(
            "http://127.0.0.1:18080/v1/chat/completions",
            data=json.dumps({
                "model": path.name,
                "messages": [{"role": "user", "content": "Name one color."}],
                "grammar": GRAMMAR,
                "max_tokens": 64,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        content = body["choices"][0]["message"]["content"]
        print("Grammar-constrained output:", content)
        parsed = json.loads(content)
        print("Valid JSON with keys:", list(parsed.keys()))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main():
    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found. Run examples 09/10 or point Ollama at one.")
        return 0
    if shutil_which := __import__("shutil").which("llama-server") is None:
        print("SKIP: llama-server not on PATH.")
        return 0
    via_server(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
