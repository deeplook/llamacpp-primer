#!/usr/bin/env python3
"""Lesson 08: Read GGUF metadata: architecture, context length, quantization.

Parses the GGUF header directly from the file; no model load needed.
Uses the GGUF found by the shared lookup, or a path given as argv[1].
"""

from __future__ import annotations

import sys
from pathlib import Path

from _common import context_length, read_gguf_metadata, require_gguf

# GGUF general.file_type -> quantization name (the common ones)
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 15: "Q4_K_M"}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else require_gguf()
    if path is None:
        sys.exit(0)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    meta = read_gguf_metadata(path)
    arch = meta.get("general.architecture")
    ftype = meta.get("general.file_type")
    print(f"file:         {path.name} ({meta['size_mb']} MB)")
    print(f"gguf version: {meta['version']}")
    print(f"name:         {meta.get('general.name')}")
    print(f"architecture: {arch}")
    print(f"context:      {context_length(meta)} tokens")
    print(
        f"quantization: {FILE_TYPES.get(ftype, ftype) if isinstance(ftype, int) else ftype}"
    )
    print(f"tensors:      {meta['n_tensors']}")


if __name__ == "__main__":
    main()
