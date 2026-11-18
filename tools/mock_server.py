#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock server for CI.

Speaks the subset llama-server exposes: GET /v1/models and
POST /v1/chat/completions. Returns canned JSON, no model needed.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

PORT = 8080

MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": "mock-model",
            "object": "model",
            "owned_by": "mock",
        }
    ],
}

CHAT_RESPONSE = {
    "id": "chatmock-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from the mock server.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass  # quiet

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            self._send(200, MODELS_RESPONSE)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.startswith("/v1/chat/completions"):
            self._send(200, CHAT_RESPONSE)
        else:
            self._send(404, {"error": "not found"})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Mock server on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
