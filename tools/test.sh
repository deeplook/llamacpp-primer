#!/usr/bin/env bash
# Smoke tests for CLI and server examples. No GGUF required.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Running example 01 (build/install check)"
python3 examples/01_build_install.py

echo "==> Starting mock server"
python3 tools/mock_server.py &
MOCK_PID=$!
trap 'kill $MOCK_PID' EXIT
sleep 1

echo "==> Hitting mock /v1/models"
curl -sf http://127.0.0.1:8080/v1/models | grep -q 'mock-model'

echo "==> Hitting mock /v1/chat/completions"
curl -sf http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}' \
  | grep -q 'Hello from the mock server'

echo "==> All smoke tests passed"
