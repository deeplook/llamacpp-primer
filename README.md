# llama.cpp Primer

[![CI](https://github.com/deeplook/primer-llamacpp/actions/workflows/ci.yml/badge.svg)](https://github.com/deeplook/primer-llamacpp/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/deeplook/primer-llamacpp?logo=github)](LICENSE)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/deeplook)

Small executable lessons for running and serving LLMs locally with llama.cpp.
It complements `openai-primer` and `agents-primer`: those talk to a hosted API;
this one runs models on your own machine — CLI, server, GGUF files,
quantization, constrained output, tools, and embeddings.

llama.cpp is the C/C++ inference engine behind Ollama and LM Studio. On macOS
it uses Metal for Apple Silicon GPUs; on Linux and Windows it targets CUDA or
CPU. The lessons are mostly Python, with a few shell examples for the CLI and
server binaries. A mock OpenAI-compatible server in `tools/` lets the networked
examples run in CI without any model file or GPU.

The approach is bottom-up: start with binaries and a first prompt, then layer
on chat templates, sampling, streaming, and server mode. From there it moves
into the GGUF toolchain — inspection, conversion, quantization — then
constrained output via grammars and JSON schemas, function calling, embeddings,
and a local RAG pipeline. The capstone composes server, tools, grammar, and
RAG into one workflow. An appendix covers performance flags that are config
knowledge rather than teachable spikes.

Examples that need a GGUF skip cleanly when none is found, so the test suite
stays offline and free. Every lesson resolves its model the same way
(`examples/_common.py`): `$MODEL_PATH`, then `models/*.gguf`, then an installed
Ollama (`make list-models` shows what it has).

## Setup

```bash
brew install llama.cpp            # llama-cli, llama-server, llama-quantize
uv sync --all-groups
make model-download               # Qwen2.5-0.5B-Instruct Q4_K_M, 398 MB, git-ignored
make live-core                    # run the model-backed lessons for real
make check-all                    # ruff + strict mypy + pytest (model tests skip without a GGUF)
make hooks                        # optional: install the pre-commit hook
```

`make check-model` verifies the pinned model (`bartowski/Qwen2.5-0.5B-Instruct-GGUF`,
397,808,192 bytes) against the Hugging Face Hub. `make setup` checks for
`llama-server` and `llama-cli` on PATH and only warns if they're absent, which
is fine for CI.

To run the CLI directly on Apple Silicon: `-ngl 99` offloads all layers to
Metal, and `--single-turn` stops `llama-cli` from dropping into interactive
mode after the reply:

```bash
llama-cli -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf -ngl 99 \
  -p "Reply with one word: primer" -n 8 --no-warmup --single-turn
```

Tool calling (lessons 13 and 20) needs `llama-server --jinja`; the lessons pass
it for you. Embedding lessons (14, 15) additionally use a model with "embed" in
its file name in `models/`; lesson 15 falls back to bag-of-words vectors
without one, and lesson 19 needs a vision model plus its `-mmproj.gguf`.

## Testing

`make check-all` (`lint`, `typecheck`, `test`) needs no model and never modifies
files; `make format` is the mutating counterpart. Without a GGUF the
model-backed tests skip; what still runs is the mock server, a compile check of
every lesson, the environment checks for 01 and 19, the RAG retrieval test, and
lesson 15 end to end against the mock. CI runs `make check-all` on the locked environment, plus
`tools/test.sh` (curl against the mock server).

With a model in `models/`, pytest also runs the lessons for real. Generated
text varies, so those tests assert structure, not wording: grammar and schema
output must parse and have the right keys, greedy decoding must repeat, the
tokenizer must round-trip, and for the pinned Qwen model the GGUF metadata must
match exactly. `make live-core` runs the same lessons outside pytest.

The optional pre-commit hook (`make hooks`) runs the generic file checks plus
`make lint` and `make typecheck`, the same commands CI uses.

Not covered by tests beyond compiling: 05, 07, 09, 10 and 14 (they need extra
models, source trees, or port 8080).

## Modules

| File | Topic |
|---|---|
| `01_build_install.py` | Build from source or install a binary; check Metal/CUDA support |
| `02_first_prompt.py` | Load a GGUF model and generate one completion |
| `03_chat_template.py` | Apply a model's chat template to multi-turn messages |
| `04_sampling.py` | Temperature, top-p, top-k, and repeat penalty |
| `05_streaming.py` | Stream tokens as they are generated |
| `06_server_basic.py` | Start `llama-server` and call its OpenAI-compatible endpoint |
| `07_server_python.py` | Talk to `llama-server` from Python with the OpenAI SDK |
| `08_gguf_inspect.py` | Read GGUF metadata: architecture, context length, quantization |
| `09_gguf_conversion.py` | Convert a Hugging Face model to GGUF |
| `10_quantization.py` | Quantize a GGUF from F16 to Q4_K_M and compare size/quality |
| `11_grammar.py` | Constrain output to valid JSON with a grammar |
| `12_json_schema.py` | Grammar from a JSON Schema for structured output |
| `13_function_calling.py` | Define tools, parse model-emitted calls, return results |
| `14_embeddings.py` | Embed texts and compute cosine similarity |
| `15_local_rag.py` | Local RAG: embed a corpus, retrieve top-k, generate grounded |
| `16_python_bindings.py` | llama-cpp-python: model options, batching, callbacks |
| `17_context_window.py` | Context length, KV cache, and what happens when it fills |
| `18_prompt_cache.py` | Reuse a stable prompt prefix across generations |
| `19_multimodal.py` | Image input with a vision-capable GGUF |
| `20_capstone.py` | Compose server, tools, grammar, and RAG in one workflow |

## Appendix: performance flags

Short notes, not full lessons:

- **Speculative decoding** — run a small draft model alongside the main one;
  accepted tokens skip the big forward pass. Flag: `--draft` / draft model path.
  Biggest win on long generations; negligible on short ones.
- **KV cache quantization** — store the cache at Q8_0 instead of F16. Roughly
  halves cache memory with minimal quality loss. Flag: `--cache-type-k q8_0
  --cache-type-v q8_0`.
- **Flash attention** — `-fa` enables memory-efficient attention. On Apple
  Silicon with Metal it's usually on by default for supported models; on CUDA
  it's a clear win for long contexts.
- **Batch sizing** — `--batch` and `--ubatch` control how many tokens process
  per step. Larger batches improve throughput on GPU but raise peak memory.
  For interactive chat, small batches with low latency matter more.

## Not covered

This primer focuses on single-node, single-model inference with llama.cpp.
Things outside that scope:

- **Distributed serving** — tensor parallelism, pipeline parallelism, or
  multi-node setups. Those belong to vLLM or TensorRT-LLM territory.
- **Fine-tuning** — llama.cpp has limited LoRA support; for training see
  Axolotl or Unsloth.
- **Audio and speech** — no STT/TTS pipelines here; audio I/O is orthogonal
  to the inference patterns this primer teaches.

## Further reading

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)
- [GGUF format spec](https://github.com/ggml/ggml/blob/master/docs/gguf.md)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
