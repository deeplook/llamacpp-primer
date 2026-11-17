# llama.cpp Primer

Small executable lessons for running and serving LLMs locally with llama.cpp.
It complements `openai-primer` and `agents-primer`: those talk to a hosted API;
this one runs models on your own machine — CLI, server, GGUF files,
quantization, constrained output, tools, and embeddings.

llama.cpp is the C/C++ inference engine behind Ollama and LM Studio. On macOS
it uses Metal for Apple Silicon GPUs; on Linux and Windows it targets CUDA or
CPU. The lessons are mostly Python via `llama-cpp-python`, with a few shell
examples for the CLI and server binaries.

## Planned modules

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
| `09_convert_hf.py` | Convert a Hugging Face model to GGUF |
| `10_quantize.py` | Quantize a GGUF from F16 to Q4_K_M and compare size/quality |
| `11_grammar_json.py` | Constrain output to valid JSON with a grammar |
| `12_json_schema.py` | Grammar from a JSON Schema for structured output |
| `13_function_calling.py` | Define tools, parse model-emitted calls, return results |
| `14_embeddings.py` | Embed texts and compute cosine similarity |
| `15_rag_local.py` | Local RAG: embed a corpus, retrieve top-k, generate grounded |
| `16_python_bindings.py` | llama-cpp-python: model options, batching, callbacks |
| `17_context_window.py` | Context length, KV cache, and what happens when it fills |
| `18_prompt_cache.py` | Reuse a stable prompt prefix across generations |
| `19_multimodal.py` | Image input with a vision-capable GGUF |
| `20_capstone.py` | Compose server, tools, grammar, and RAG in one workflow |

## Appendix: performance flags

Not full lessons — short notes on speculative decoding, KV cache quantization,
flash attention, and batch sizing. Config knowledge, not teachable spikes.

## Setup (planned)

```bash
uv sync --all-groups
# download a small GGUF, e.g. TinyLlama Q4_K_M from Hugging Face
llama-server --model models/tinyllama.Q4_K_M.gguf --port 8080
uv run python examples/06_server_basic.py
```

Examples that need a model file will skip cleanly when it is missing, so the
test suite stays offline and free.

## Further reading

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama-server docs](https://github.com/ggml-org/llama.cpp/tree/master/examples/server)
- [GGUF format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
