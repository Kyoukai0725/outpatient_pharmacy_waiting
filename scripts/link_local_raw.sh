#!/usr/bin/env bash
# Optional: symlink raw extracts from the local study folder without copying ~1GB.
set -euo pipefail
SRC="${1:-/home/pharmacy-department/research/pharmacy_lining/raw}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi
mkdir -p "$ROOT/raw"
for name in wait_times dispense layout schedules machine_changes; do
  if [[ -e "$SRC/$name" && ! -e "$ROOT/raw/$name" ]]; then
    ln -s "$SRC/$name" "$ROOT/raw/$name"
    echo "linked raw/$name"
  fi
done
echo "Done. Remember: raw extracts must not be committed."
