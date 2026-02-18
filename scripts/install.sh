#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install this repo in editable mode with optional feature sets.

Usage:
  ./scripts/install.sh [--frontend] [--ml] [--no-base]

Defaults:
  Installs base dependencies + editable package.

Examples:
  ./scripts/install.sh
  ./scripts/install.sh --frontend
  ./scripts/install.sh --ml
  ./scripts/install.sh --frontend --ml
EOF
}

WITH_BASE=1
WITH_FRONTEND=0
WITH_ML=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --no-base) WITH_BASE=0; shift ;;
    --frontend) WITH_FRONTEND=1; shift ;;
    --ml) WITH_ML=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

EXTRAS=()
if [[ $WITH_BASE -eq 1 ]]; then EXTRAS+=("base"); fi
if [[ $WITH_FRONTEND -eq 1 ]]; then EXTRAS+=("frontend"); fi
if [[ $WITH_ML -eq 1 ]]; then EXTRAS+=("ml"); fi

if [[ ${#EXTRAS[@]} -eq 0 ]]; then
  python -m pip install -e .
else
  python -m pip install -e ".[${EXTRAS[*]}]"
fi

echo ""
echo "[OK] Installed editable package."

