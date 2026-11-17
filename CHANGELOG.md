# Changelog

## [Unreleased]

### Added
- Examples 03–08: chat template, sampling, streaming, server basics, server
  from Python, GGUF inspection.
- Examples 09–15: GGUF conversion, quantization, grammar-constrained JSON,
  JSON Schema output, function calling, embeddings, local RAG.
- Examples 16–20: Python bindings, context window, prompt caching,
  multimodal image input, capstone composing server/tools/grammar/RAG.
- Expanded README intro (purpose, bottom-up approach, mock-server CI strategy,
  Ollama discovery), fixed module table filenames, and added appendix notes
  on speculative decoding, KV cache quantization, flash attention, batch sizing.
- Not-covered section documenting scope boundaries (distributed serving,
  fine-tuning, audio).
- Extended PyTest coverage: GGUF-gated tests for examples 08, 16, 17; syntax
  check for the capstone; compile-all guard across every example file.
- pyproject.toml: MIT license field, mypy strict and ruff configs, uv
  `package = false`, mypy and ruff added to dev dependencies.

### Fixed
- CI install step: replaced broken prebuilt tarball URL with a shallow
  source build of llama.cpp (later removed entirely — CI now runs
  mock-backed smoke tests only, no real binary needed).
- Makefile `check-binaries` now warns instead of hard-failing when binaries
  are missing; `tools/test.sh` skips the check so smoke stands on the mock
  server alone.

## [0.1.0] - 2026-11-17

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
