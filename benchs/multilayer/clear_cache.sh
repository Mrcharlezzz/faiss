#!/usr/bin/env bash
# Usage: clear_cache.sh [--all]. Default removes dedupe markers; --all also removes indexes and ground truth.
set -euo pipefail
cache="$(cd "$(dirname "$0")" && pwd)/cache"
[ -d "$cache" ] || exit 0
if [ "${1:-}" = "--all" ]; then
    rm -rf "$cache"
else
    rm -f "$cache"/run-*.txt
fi
