#!/usr/bin/env bash
# Dev mode: starts FastAPI backend + Vite + Tauri window with hot reload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    pkill -P "$BACKEND_PID" 2>/dev/null || true
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  # Last-resort sweep in case an orphan is still bound to the backend port.
  lsof -ti:8765 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Install frontend deps if missing.
if [[ ! -d "$SCRIPT_DIR/frontend/node_modules" ]]; then
  (cd "$SCRIPT_DIR/frontend" && pnpm install)
fi

# Backend. Pre-clear the port so restarts after a crashed run still work.
lsof -ti:8765 2>/dev/null | xargs -r kill -9 2>/dev/null || true
(cd "$REPO_ROOT" && exec uv run python -m desktop.backend.server) &
BACKEND_PID=$!

# Tauri (launches Vite via beforeDevCommand).
cd "$SCRIPT_DIR/src-tauri"
cargo tauri dev
