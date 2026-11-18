# Changelog

## [Unreleased]

### Added
- Examples 03-08: chat template, sampling, streaming, server basics, server
  from Python, GGUF inspection.
- Examples 09-15: GGUF conversion, quantization, grammar-constrained JSON,
  JSON Schema output, function calling, embeddings, local RAG.
- Examples 16-20: Python bindings, context window, prompt caching,
  multimodal image input, capstone composing server/tools/grammar/RAG.
- Shared `examples/_common.py`: one model lookup (`$MODEL_PATH`, `models/`,
  Ollama), a correct GGUF header reader, a `llama_server()` context manager
  and a `mock_server()` context manager, replacing per-lesson copies.
- Real-model workflow: `make model-download` (pinned Qwen2.5-0.5B-Instruct
  Q4_K_M, 397,808,192 bytes), `check-model` (verifies it against the Hugging
  Face Hub via `tools/check_model.py`), `live-core` (now runs every
  model-backed lesson), plus `format`, `lint`, `test` and `check-all`.
- `typecheck` and `install` Makefile targets; `check-all` is now lint, strict
  mypy and tests. All examples, tests and tools pass `mypy --strict`.
- Optional pre-commit hook (`make hooks`) running the generic file checks plus
  `make lint` and `make typecheck`.
- `.python-version`, a committed `uv.lock`, and `openai` as a dev dependency.
- README: setup flow, Metal CLI invocation (`-ngl 99 --single-turn
  --no-warmup`), `--jinja` for tool calling, model-lookup order, expanded
  intro, appendix on performance flags and a not-covered section.
- Stronger assertions: exact metadata for the pinned model (architecture,
  context length, quantization, tensor count, size); tokenizer round trip in
  16; hex pattern check in 12; repeatable greedy decoding in 04; non-empty
  reply in 06; RAG retrieval test on a synthetic corpus (15).
- Tests for 04, 06, 11-13, 15, 18 and 20 (17 already had one); lesson 15 falls back to the mock
  server when no chat model is available, and a test covers that path.
- pyproject.toml: MIT license field, mypy strict and ruff configs, uv
  `package = false`, mypy and ruff in dev dependencies.

### Fixed
- 08/17: the GGUF reader read value types as one byte (the spec uses uint32),
  and 17 parsed tensor info before the key/value section. Both now share one
  reader and report `<arch>.context_length`.
- 11/20: the grammar was malformed and did not fix its keys. It now enforces
  `color`/`reason` and `answer`/`weather`, and truncation by `max_tokens` is an
  error rather than invalid JSON.
- 11-13, 18, 20: looked for a TinyLlama file nothing downloads. 13 and 20 now
  start the server with `--jinja` so tool calls work; 18 reports cached tokens.
- CI now installs the locked environment and runs `make check-all` (ruff,
  mypy, pytest) on current action versions; the Buy Me a Coffee badge from
  `primer-standards` was added to the README.
- Makefile: `.PHONY` was lowercase and so ineffective; `check-all`, which the
  README documents, did not exist.
- Tests no longer depend on Ollama or collide with `llama-server` on port 8080
  (the mock server takes a port argument). Ruff selects an explicit rule set.
- CI install step: replaced a broken prebuilt tarball URL with a shallow
  source build of llama.cpp, later removed entirely; CI now runs mock-backed
  smoke tests only and needs no real binary.
- Makefile `check-binaries` warns instead of failing when binaries are
  missing; `tools/test.sh` skips the check so the smoke test stands on the
  mock server alone.

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
