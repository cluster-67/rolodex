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

echo "[format] Skipping third_party directories"
mapfile -d "" files < <(find "$2/src" "$2/include" -type f -not -path "*/third_party/*" -print0)

file_count="${#files[@]}"
echo "[format] Found ${file_count} file(s) under src/ and include/"
if [ "$file_count" -eq 0 ]; then
  echo "[format] Nothing to format."
  exit 0
fi

for file in "${files[@]}"; do
  clang-format -i "$file"
done

echo "[format] Done."
' _ "$LLVM_MODULE" "$REPO_ROOT"
