.DEFAULT_GOAL := help

.phony: help setup check-binaries list-models live-core

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: check-binaries ## Verify llama.cpp binaries are on PATH
	@echo "Setup complete."

check-binaries: ## Check for llama-server and llama-cli (warns only)
	@command -v llama-server >/dev/null 2>&1 && echo "llama-server: $$(llama-server --version 2>&1 | head -1)" || echo "llama-server not found (ok in CI, run: brew install llama.cpp)"
	@command -v llama-cli >/dev/null 2>&1 && echo "llama-cli:    $$(llama-cli --version 2>&1 | head -1)" || echo "llama-cli not found (ok in CI, run: brew install llama.cpp)"

list-models: ## List GGUF models found via Ollama
	@python3 tools/list_ollama_models.py

live-core: ## Run GGUF-dependent examples against a real llama-server (Mac only)
	@echo "Requires llama-server on PATH and a GGUF (see: make list-models)"
	@command -v llama-server >/dev/null 2>&1 || (echo "llama-server not found"; exit 1)
	python3 examples/02_first_prompt.py
	python3 examples/03_chat_template.py
	python3 examples/04_sampling.py
	python3 examples/05_streaming.py
	python3 examples/08_gguf_inspect.py
	python3 examples/16_python_bindings.py
	python3 examples/17_context_window.py
	python3 examples/18_prompt_cache.py
	@echo "live-core: all real-model examples passed"
