.DEFAULT_GOAL := help

.phony: help setup check-binaries

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: check-binaries ## Verify llama.cpp binaries are on PATH
	@echo "Setup complete."

check-binaries: ## Check for llama-server and llama-cli (warns only)
	@command -v llama-server >/dev/null 2>&1 && echo "llama-server: $$(llama-server --version 2>&1 | head -1)" || echo "llama-server not found (ok in CI, run: brew install llama.cpp)"
	@command -v llama-cli >/dev/null 2>&1 && echo "llama-cli:    $$(llama-cli --version 2>&1 | head -1)" || echo "llama-cli not found (ok in CI, run: brew install llama.cpp)"

list-models: ## List GGUF models found via Ollama
	@python3 tools/list_ollama_models.py
