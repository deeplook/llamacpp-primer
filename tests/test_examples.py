"""PyTest suite for Python examples.

Model-dependent tests skip when no GGUF is present via Ollama.
Mock-server tests always run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"


def _has_gguf() -> bool:
    if not OLLAMA_BLOBS.exists():
        return False
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
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
                return True
    return False


HAS_GGUF = _has_gguf()


requires_gguf = pytest.mark.skipif(not HAS_GGUF, reason="No GGUF found via Ollama")


@requires_gguf
def test_first_prompt_runs():
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "02_first_prompt.py")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "France" in result.stdout or len(result.stdout.strip()) > 0


def test_mock_server_models():
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "mock_server.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        import time
        time.sleep(1)
        result = subprocess.run(
            ["curl", "-sf", "http://127.0.0.1:8080/v1/models"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["data"][0]["id"] == "mock-model"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mock_server_chat():
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "mock_server.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        import time
        time.sleep(1)
        result = subprocess.run(
            [
                "curl", "-sf", "http://127.0.0.1:8080/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(
                    {
                        "model": "mock-model",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "Hello from the mock server" in data["choices"][0]["message"]["content"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)
