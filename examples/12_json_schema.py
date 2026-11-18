#!/usr/bin/env python3
"""Lesson 12: Constrain output with a JSON Schema.

llama-server accepts a response_format with json_schema, which it
compiles to a grammar internally. Skips without a GGUF or server binary.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from _common import http_json, llama_server, require_binary, require_gguf

PORT = 18081

SCHEMA: dict[str, Any] = {
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


def main() -> None:
    if require_binary("llama-server") is None:
        sys.exit(0)
    path = require_gguf()
    if path is None:
        sys.exit(0)

    with llama_server(path, PORT, "--ctx-size", "512", "-ngl", "99") as base:
        body = http_json(
            f"{base}/v1/chat/completions",
            {
                "model": path.name,
                "messages": [{"role": "user", "content": "What color is grass?"}],
                "response_format": {"type": "json_schema", "json_schema": SCHEMA},
                "max_tokens": 64,
            },
        )
    content = body["choices"][0]["message"]["content"]
    print("Schema-constrained output:", content)
    parsed = json.loads(content)
    assert set(parsed) == {"color", "hex"}, parsed
    pattern = SCHEMA["schema"]["properties"]["hex"]["pattern"]
    assert re.fullmatch(pattern, parsed["hex"]), f"hex violates schema: {parsed}"
    print(f"Validated against schema: color = {parsed['color']}, hex = {parsed['hex']}")


if __name__ == "__main__":
    main()
