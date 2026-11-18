"""PyTest suite for the examples.

Model-dependent tests skip when no GGUF is available (see examples/_common.py:
$MODEL_PATH, models/*.gguf, or Ollama). Mock-server tests always run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))

from _common import (  # noqa: E402
    DEFAULT_MODEL,
    context_length,
    find_gguf,
    read_gguf_metadata,
)

GGUF = find_gguf()
requires_gguf = pytest.mark.skipif(GGUF is None, reason="No GGUF available")
requires_server = pytest.mark.skipif(
    shutil.which("llama-server") is None, reason="llama-server not on PATH"
)


def run_example(name: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
    )


@pytest.fixture
def mock_url() -> Iterator[str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "mock_server.py"), str(port)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        proc.stdout.readline()  # "Mock server on ..."
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mock_server_models(mock_url: str) -> None:
    data = json.load(urllib.request.urlopen(f"{mock_url}/v1/models", timeout=10))
    assert data["data"][0]["id"] == "mock-model"


def test_mock_server_chat(mock_url: str) -> None:
    req = urllib.request.Request(
        f"{mock_url}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    data = json.load(urllib.request.urlopen(req, timeout=10))
    assert "Hello from the mock server" in data["choices"][0]["message"]["content"]


def test_all_examples_are_valid_python() -> None:
    """Every example file compiles."""
    for path in sorted(EXAMPLES.glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


@pytest.mark.parametrize("name", ["01_build_install.py", "19_multimodal.py"])
def test_environment_examples_never_fail(name: str) -> None:
    """These skip or report; neither may crash without a model."""
    result = run_example(name, timeout=60)
    assert result.returncode == 0, result.stderr


def test_rag_falls_back_to_mock() -> None:
    """With no model at all, lesson 15 still runs end to end via the mock."""
    env = {**os.environ, "MODEL_PATH": "", "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "15_local_rag.py")],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Mock answer: Hello from the mock server" in result.stdout


def test_rag_retrieval_picks_the_relevant_chunk() -> None:
    spec = importlib.util.spec_from_file_location("rag", EXAMPLES / "15_local_rag.py")
    assert spec and spec.loader
    rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag)
    chunks = [
        "cats purr loudly and sleep on warm sofas",
        "rockets burn fuel to reach orbit around the earth",
        "bread dough needs flour water and yeast",
    ]
    score, best = rag.retrieve(chunks, "how do rockets reach orbit", rag.bow_embed)
    assert best == 1
    assert 0 < score <= 1


@requires_gguf
def test_gguf_header_reader() -> None:
    assert GGUF is not None
    meta = read_gguf_metadata(GGUF)
    assert meta["version"] >= 2
    assert meta["n_tensors"] > 0
    if GGUF.name != DEFAULT_MODEL:
        pytest.skip("exact values are only known for the default model")
    assert meta["general.architecture"] == "qwen2"
    assert context_length(meta) == 32768
    assert meta["general.file_type"] == 15  # Q4_K_M
    assert meta["n_tensors"] == 290
    assert GGUF.stat().st_size == 397_808_192


@requires_gguf
@pytest.mark.parametrize(
    "name,expected",
    [
        ("02_first_prompt.py", "Loading"),
        ("04_sampling.py", "greedy decoding repeats exactly"),
        ("03_chat_template.py", ""),
        ("08_gguf_inspect.py", "quantization: Q4_K_M"),
        ("16_python_bindings.py", "Detokenized"),
    ],
)
def test_in_process_examples(name: str, expected: str) -> None:
    result = run_example(name)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert result.stdout.strip()


@requires_gguf
@requires_server
@pytest.mark.parametrize(
    "name,expected",
    [
        ("06_server_basic.py", ""),
        ("11_grammar.py", "Valid JSON with keys: ['color', 'reason']"),
        ("12_json_schema.py", "Validated against schema"),
        ("13_function_calling.py", "Final answer"),
        ("17_context_window.py", "rejected"),
        ("18_prompt_cache.py", "Prefix reuse confirmed"),
        ("20_capstone.py", "CAPSTONE PASSED"),
    ],
)
def test_server_examples(name: str, expected: str) -> None:
    result = run_example(name)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
