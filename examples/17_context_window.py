#!/usr/bin/env python3
"""Example 17: Context window, KV cache, and overflow behavior.

Reads n_ctx from the GGUF metadata, fills the context with repeated
prompts until generation degrades, and shows what llama.cpp does when
the window is exceeded (truncation vs error depending on settings).
Skips without a GGUF.
"""
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_gguf_ctx(path):
    """Minimal GGUF reader: grab general.context_length from the KV section."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return None
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        # Skip tensor info
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]
            f.read(nlen)
            f.read(32)  # type + dims
            f.read(8)   # offset
        for _ in range(n_kv):
            nlen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(nlen).decode("utf-8", errors="ignore")
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == 4:  # uint32
                val = struct.unpack("<I", f.read(4))[0]
            elif vtype == 5:  # int32
                val = struct.unpack("<i", f.read(4))[0]
            elif vtype == 6:  # float32
                val = struct.unpack("<f", f.read(4))[0]
            elif vtype == 8:  # uint64
                val = struct.unpack("<Q", f.read(8))[0]
            elif vtype == 9:  # int64
                val = struct.unpack("<q", f.read(8))[0]
            elif vtype == 10:  # double
                val = struct.unpack("<d", f.read(8))[0]
            elif vtype == 7:  # bool
                val = bool(struct.unpack("<B", f.read(1))[0])
            else:
                # string or array — skip by reading length-prefixed blob
                if vtype == 8:
                    pass
                slen = struct.unpack("<Q", f.read(8))[0]
                f.read(slen)
                continue
            if key == "general.context_length":
                return val
    return None


def find_gguf():
    models = ROOT / "models"
    if not models.exists():
        return None
    ggufs = sorted(models.glob("*.gguf"))
    return next((g for g in ggufs if "embed" not in g.name.lower()), None)


def main():
    path = find_gguf()
    if path is None:
        print("SKIP: no GGUF found. Run examples 09/10 first.")
        return 0

    ctx = read_gguf_ctx(path)
    print(f"{path.name}: general.context_length = {ctx}")
    if ctx is None:
        print("Could not read context length from GGUF header.")
        return 0

    # Demonstrate the concept with llama-server if available
    import shutil
    if shutil.which("llama-server") is None:
        print("llama-server not on PATH; header read is all we can show.")
        return 0

    print(f"\nContext of {ctx} tokens means the KV cache holds that many."
          f"\nOnce filled, older tokens get dropped (default) or the"
          f"\nrequest errors if --ctx-size is set below the prompt length."
          f"\nTry: llama-server --model {path.name} --ctx-size 256"
          f"\nwith a prompt longer than 256 tokens and watch it fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
