#!/usr/bin/env python3
"""Lesson 13: Function calling with the tools API.

llama-server supports OpenAI-style tool definitions when started with
--jinja. The model returns a tool call instead of text; you execute it and
feed the result back. Skips without a GGUF or llama-server binary.
"""

from __future__ import annotations

import json
import sys

from _common import http_json, llama_server, require_binary, require_gguf

PORT = 18082

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def main() -> None:
    if require_binary("llama-server") is None:
        sys.exit(0)
    path = require_gguf()
    if path is None:
        sys.exit(0)

    question = {"role": "user", "content": "What's the weather in Berlin?"}
    with llama_server(
        path, PORT, "--ctx-size", "1024", "--jinja", "-ngl", "99"
    ) as base:
        url = f"{base}/v1/chat/completions"
        body = http_json(
            url,
            {
                "model": path.name,
                "messages": [question],
                "tools": TOOLS,
                "max_tokens": 128,
            },
        )
        msg = body["choices"][0]["message"]
        print("Assistant message:", json.dumps(msg, indent=2))
        if not msg.get("tool_calls"):
            print("No tool call returned (a small model may not support tools well).")
            return

        tc = msg["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        print(f"Tool call: {tc['function']['name']}({args})")
        # Simulate tool execution and send the result back
        result = {"city": args["city"], "temp_c": 18, "condition": "cloudy"}
        body = http_json(
            url,
            {
                "model": path.name,
                "messages": [
                    question,
                    msg,
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_0"),
                        "content": json.dumps(result),
                    },
                ],
                "max_tokens": 128,
            },
        )
        print("Final answer:", body["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
