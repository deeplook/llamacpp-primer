"""Shared helpers for the lessons: model lookup, GGUF header reader, server."""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"

# The model fetched by `make model-download` (397,808,192 bytes).
DEFAULT_MODEL = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

SKIP_HINT = "Run `make model-download` or set MODEL_PATH to a GGUF file."


def _ollama_ggufs() -> list[Path]:
    """GGUF blobs Ollama knows about (empty if Ollama is absent)."""
    if not OLLAMA_BLOBS.exists():
        return []
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    blobs: list[Path] = []
    for line in result.stdout.splitlines():
        try:
            digest = json.loads(line).get("digest", "")
        except json.JSONDecodeError:
            continue
        if digest.startswith("sha256:"):
            blob = OLLAMA_BLOBS / ("sha256-" + digest.split(":", 1)[1])
            if blob.exists():
                blobs.append(blob)
    return blobs


def find_gguf(embedding: bool = False) -> Path | None:
    """Locate a GGUF: $MODEL_PATH, then models/*.gguf, then Ollama (chat only).

    Embedding models are told apart from chat models by "embed" in the name.
    """
    env = os.environ.get("MODEL_PATH")
    if env and not embedding:
        return Path(env) if Path(env).exists() else None
    local = [
        g for g in sorted(MODELS_DIR.glob("*.gguf")) if "mmproj" not in g.name.lower()
    ]
    wanted = [g for g in local if ("embed" in g.name.lower()) == embedding]
    if wanted:
        # Prefer the default model so every lesson uses the same weights.
        return next((g for g in wanted if g.name == DEFAULT_MODEL), wanted[0])
    if not embedding:
        return next(iter(_ollama_ggufs()), None)
    return None


def require_gguf(embedding: bool = False) -> Path | None:
    """find_gguf() that prints a SKIP line when nothing is found."""
    path = find_gguf(embedding)
    if path is None:
        kind = "embedding GGUF" if embedding else "GGUF"
        print(f"SKIP: no {kind} found. {SKIP_HINT}")
    return path


def require_binary(name: str) -> str | None:
    """shutil.which() that prints a SKIP line when the binary is missing."""
    path = shutil.which(name)
    if path is None:
        print(f"SKIP: {name} not on PATH (brew install llama.cpp).")
    return path


# --- GGUF header ---------------------------------------------------------

# value type -> struct format (GGUF spec); strings and arrays are special.
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}  # fmt: skip
_STRING, _ARRAY = 8, 9


def _read_string(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    """Read one value; arrays are skipped (tokenizer vocabularies are huge)."""
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        (val,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        return val
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if etype in _SCALARS:
            f.seek(n * struct.calcsize(_SCALARS[etype]), os.SEEK_CUR)
        else:
            for _ in range(n):
                _read_value(f, etype)
        return f"<array of {n}>"
    raise ValueError(f"unsupported GGUF value type {vtype}")


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read the key/value header of a GGUF file without loading the model."""
    meta: dict[str, Any] = {"size_mb": round(path.stat().st_size / 2**20, 1)}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        (meta["version"],) = struct.unpack("<I", f.read(4))
        meta["n_tensors"], n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_value(f, vtype)
    return meta


def context_length(meta: dict[str, Any]) -> int | None:
    """The trained context length, whatever the architecture prefix is."""
    arch = meta.get("general.architecture", "")
    val = meta.get(f"{arch}.context_length")
    return val if isinstance(val, int) else None


# --- llama-server --------------------------------------------------------


def http_json(
    url: str, payload: dict[str, Any] | None = None, timeout: int = 120
) -> Any:
    """GET (no payload) or POST JSON and return the decoded response."""
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


@contextmanager
def llama_server(
    model: Path, port: int, *extra: str, startup_timeout: int = 90
) -> Iterator[str]:
    """Run llama-server on `model` and yield its base URL; stop it on exit."""
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["llama-server", "--model", str(model), "--port", str(port), *extra],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(startup_timeout):
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become ready in time")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@contextmanager
def mock_server() -> Iterator[str]:
    """Run tools/mock_server.py on a free port and yield its base URL.

    Stands in for llama-server when no model is available (CI).
    """
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
