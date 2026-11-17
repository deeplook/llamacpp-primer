# llama.cpp Primer

[![CI](https://github.com/deeplook/primer-llamacpp/actions/workflows/ci.yml/badge.svg)](https://github.com/deeplook/primer-llamacpp/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/deeplook/primer-llamacpp?logo=github)](LICENSE)

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

Examples that need a GGUF skip cleanly when the file is missing, so the test
suite stays offline and free. On a Mac with Ollama installed, `make list-models`
discovers existing blobs automatically.

## Setup

```bash
uv sync --all-groups
# optional: download a small GGUF, e.g. TinyLlama Q4_K_M from Hugging Face
llama-server --model models/tinyllama.Q4_K_M.gguf --port 8080
uv run python examples/06_server_basic.py
```

`make setup` checks for `llama-server` and `llama-cli` on PATH and warns if
they're absent — fine for CI, required for the live examples. `make check-all`
runs formatting, the mock-backed smoke tests, and PyTest.

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
