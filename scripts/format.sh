#!/usr/bin/env bash
set -euo pipefail

LLVM_MODULE="${LLVM_MODULE:-llvm/21.1.4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash -lc '
set -euo pipefail
echo "[format] Starting formatting run..."
module load "$1" >/dev/null 2>&1
command -v clang-format >/dev/null || {
  echo "clang-format not found after loading module: $1" >&2
  exit 1
}
file_count="$(find "$2/src" "$2/include" -type f | wc -l)"
echo "[format] Found ${file_count} file(s) under src/ and include/"
if [ "$file_count" -eq 0 ]; then
  echo "[format] Nothing to format."
  exit 0
fi
find "$2/src" "$2/include" -type f -print0 | xargs -0r clang-format -i
echo "[format] Done."
' _ "$LLVM_MODULE" "$REPO_ROOT"
