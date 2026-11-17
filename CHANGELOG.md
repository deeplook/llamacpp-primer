# Changelog

## 0.1.0 — 2026-11-17

Initial scaffold:

- README with 20 planned modules across eight themes
- Makefile with setup target and binary checks
- Ollama model discovery helper (`tools/list_ollama_models.py`)
- Example 01: build/install check
- Example 02: first prompt with GGUF auto-discovery
- Mock OpenAI-compatible server for CI (`tools/mock_server.py`)
- Bash smoke script (`tools/test.sh`)
- PyTest suite with GGUF-gated and mock tests
- GitHub Actions CI workflow with GGUF caching
- pyproject.toml for uv
