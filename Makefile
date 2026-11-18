.DEFAULT_GOAL := help

EXAMPLE ?= examples/01_build_install.py
MODEL_FILE := Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
MODEL_URL := https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/$(MODEL_FILE)

.PHONY: help setup check-binaries list-models model-download check-model live-core run install format lint typecheck test check-all hooks

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: check-binaries ## Verify llama.cpp binaries are on PATH
	@echo "Setup complete."

check-binaries: ## Check for llama-server and llama-cli (warns only)
	@command -v llama-server >/dev/null 2>&1 && echo "llama-server: $$(llama-server --version 2>&1 | head -1)" || echo "llama-server not found (ok in CI, run: brew install llama.cpp)"
	@command -v llama-cli >/dev/null 2>&1 && echo "llama-cli:    $$(llama-cli --version 2>&1 | head -1)" || echo "llama-cli not found (ok in CI, run: brew install llama.cpp)"

list-models: ## List GGUF models found via Ollama
	@python3 tools/list_ollama_models.py

run: ## Run one example (EXAMPLE=examples/07_server_python.py); Python via uv, Bash directly
	@case "$(EXAMPLE)" in \
		*.py) uv run python $(EXAMPLE) ;; \
		*.sh) bash $(EXAMPLE) ;; \
		*) echo "Unknown extension in $(EXAMPLE)"; exit 1 ;; \
	esac

model-download: ## Download the default 0.5B Q4_K_M model (398 MB) into models/
	mkdir -p models
	curl -L --fail -C - -o models/$(MODEL_FILE) $(MODEL_URL)

check-model: ## Verify the default model's name and size against the Hugging Face Hub
	@python3 tools/check_model.py

live-core: ## Run GGUF-dependent examples against a real model (see: make model-download)
	@command -v llama-server >/dev/null 2>&1 || (echo "llama-server not found (brew install llama.cpp)"; exit 1)
	@test -n "$$(ls models/*.gguf 2>/dev/null)" || (echo "No GGUF in models/ (run: make model-download)"; exit 1)
	@for ex in 02_first_prompt 03_chat_template 04_sampling 05_streaming 06_server_basic \
	    08_gguf_inspect 11_grammar 12_json_schema 13_function_calling 15_local_rag \
	    16_python_bindings 17_context_window 18_prompt_cache 20_capstone; do \
	  echo "==> $$ex"; uv run python examples/$$ex.py || exit 1; \
	done
	@echo "live-core: all real-model examples passed"

install: ## Install locked dependencies
	uv sync --all-groups --locked

format: ## Format and fix Python code
	uv run ruff format examples tests tools
	uv run ruff check --fix examples tests tools

lint: ## Check formatting and linting
	uv run ruff format --check examples tests tools
	uv run ruff check examples tests tools

typecheck: ## Strict mypy over examples, tests and tools
	uv run mypy examples tests tools

test: ## Run the PyTest suite (model-backed tests skip without a GGUF)
	uv run pytest

check-all: lint typecheck test ## Safe quality gate: lint, typecheck, test (never modifies files)

hooks: ## Install the git pre-commit hook (needs uv)
	uvx pre-commit install
