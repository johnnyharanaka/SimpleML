#!/usr/bin/env bash
# Build mode: produces a distributable Tauri app bundle.
# Output: desktop/src-tauri/target/release/bundle/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install frontend deps if missing.
if [[ ! -d "$SCRIPT_DIR/frontend/node_modules" ]]; then
  (cd "$SCRIPT_DIR/frontend" && pnpm install)
fi

cd "$SCRIPT_DIR/src-tauri"
cargo tauri build

echo
echo "Build artifacts: $SCRIPT_DIR/src-tauri/target/release/bundle/"
