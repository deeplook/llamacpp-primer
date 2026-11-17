#!/usr/bin/env python3
"""Example 13: Function calling with the tools API.

llama-server supports OpenAI-style tool definitions. The model returns
a tool call instead of text; you execute it and feed the result back.
Skips without a GGUF or llama-server binary.
"""
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]


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
        ["llama-server", "--model", str(path), "--port", "18082", "--ctx-size", "1024"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:18082/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")

        req = urllib.request.Request(
            "http://127.0.0.1:18082/v1/chat/completions",
            data=json.dumps({
                "model": path.name,
                "messages": [{"role": "user", "content": "What's the weather in Berlin?"}],
                "tools": TOOLS,
                "max_tokens": 128,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        msg = body["choices"][0]["message"]
        print("Assistant message:", json.dumps(msg, indent=2))
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            print(f"Tool call: {tc['function']['name']}({args})")
            # Simulate tool execution and send result back
            result = {"city": args["city"], "temp_c": 18, "condition": "cloudy"}
            req2 = urllib.request.Request(
                "http://127.0.0.1:18082/v1/chat/completions",
                data=json.dumps({
                    "model": path.name,
                    "messages": [
                        {"role": "user", "content": "What's the weather in Berlin?"},
                        msg,
                        {"role": "tool", "tool_call_id": tc["id"],
                         "content": json.dumps(result)},
                    ],
                    "max_tokens": 128,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req2, timeout=60) as resp:
                body2 = json.load(resp)
            print("Final answer:", body2["choices"][0]["message"]["content"])
        else:
            print("No tool call returned (model may not support tools well).")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
