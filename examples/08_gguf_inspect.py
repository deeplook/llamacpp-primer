#!/usr/bin/env python3
"""Lesson 08: Read GGUF metadata: architecture, context length, quantization.

Parses the GGUF header directly from the file — no model load needed.
Uses the first GGUF found via Ollama, or a path given as argv[1].
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

OLLAMA_BLOBS = Path.home() / ".ollama" / "models" / "blobs"

# GGUF value types: 0=uint8 ... 12=array (simplified mapping)
GGUF_VALUE_NAMES = {
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}


def find_first_gguf() -> Path | None:
    if not OLLAMA_BLOBS.exists():
        return None
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
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
                return blob
    return None


def read_gguf_metadata(path: Path) -> dict:
    """Read key metadata fields from a GGUF file header."""
    meta = {"path": str(path), "size_mb": path.stat().st_size / (1024 * 1024)}
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            meta["error"] = f"Not a GGUF file (magic={magic!r})"
            return meta
        version = struct.unpack("<I", f.read(4))[0]
        meta["version"] = version
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        meta["n_tensors"] = n_tensors
        meta["n_kv"] = n_kv

        def read_value(vtype: int):
            if vtype == 8:  # string
                (slen,) = struct.unpack("<Q", f.read(8))
                return f.read(slen).decode("utf-8", errors="replace")
            if vtype in (4, 5):  # uint32 / int32
                return struct.unpack("<I" if vtype == 4 else "<i", f.read(4))[0]
            if vtype == 6:  # float32
                return struct.unpack("<f", f.read(4))[0]
            if vtype == 7:  # bool
                return bool(f.read(1)[0])
            if vtype in (10, 11):  # uint64 / int64
                return struct.unpack("<Q" if vtype == 10 else "<q", f.read(8))[0]
            if vtype == 12:  # float64
                return struct.unpack("<d", f.read(8))[0]
            if vtype == 9:  # array
                at = f.read(1)[0]
                (alen,) = struct.unpack("<Q", f.read(8))
                return [read_value(at) for _ in range(alen)]
            raise ValueError(f"unsupported value type {vtype}")

        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", f.read(8))
            key = f.read(klen).decode("utf-8", errors="replace")
            vtype = f.read(1)[0]
            try:
                val = read_value(vtype)
            except (ValueError, struct.error):
                break
            if key in {
                "general.architecture",
                "general.name",
                "general.quantization_version",
            } or key.startswith(f"{val if isinstance(val, str) else ''}.context_length".replace(str(val), "")):
                meta[key] = val
            # capture architecture-scoped keys like llama.context_length
            if ".context_length" in key or ".quantization" in key or key.startswith("general."):
                meta[key] = val
    return meta


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_first_gguf()
    if path is None:
        print("No GGUF found. Pass a path: python examples/08_gguf_inspect.py model.gguf")
        sys.exit(0)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    meta = read_gguf_metadata(path)
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
