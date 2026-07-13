#!/usr/bin/env bash
# Auto-format the codebase and fix auto-fixable lint issues with ruff.
#
# Usage:
#   scripts/format.sh          # format in place
#   scripts/format.sh --check  # fail if anything would change, don't write (for CI)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "--check" ]]; then
    ruff format --check .
    ruff check .
else
    ruff format .
    ruff check --fix .
fi
