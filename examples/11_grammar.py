#!/usr/bin/env python3
"""Lesson 11: Constrain output with a GBNF grammar.

Forces the model to emit a JSON object with exactly the keys "color" and
"reason". Works with llama-server's `grammar` parameter or llama-cli's
--grammar flag. Skips if no GGUF or llama-server is available.
"""

from __future__ import annotations

import json
import sys

from _common import http_json, llama_server, require_binary, require_gguf

PORT = 18080

# Raw string: GBNF escapes (\" and \\) must reach llama-server untouched.
GRAMMAR = r"""root   ::= "{" ws "\"color\"" ws ":" ws string "," ws "\"reason\"" ws ":" ws string ws "}"
string ::= "\"" ([^"\\\n] | "\\" ["\\/bfnrt])* "\""
ws     ::= [ \t\n]*
"""


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
                "messages": [
                    {
                        "role": "user",
                        "content": "Name one color and give a reason in under 12 words.",
                    }
                ],
                "grammar": GRAMMAR,
                "max_tokens": 128,
            },
        )
    choice = body["choices"][0]
    content = choice["message"]["content"]
    # A grammar guarantees syntax only if generation is not cut off.
    assert choice["finish_reason"] == "stop", "output truncated by max_tokens"
    print("Grammar-constrained output:", content)
    parsed = json.loads(content)
    assert list(parsed) == ["color", "reason"], parsed
    print("Valid JSON with keys:", list(parsed))


if __name__ == "__main__":
    main()
