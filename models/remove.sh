#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MATCH_ID="${1:-}"

if [[ -z "$MATCH_ID" ]]; then
  echo "which model id? see: models/fetch.sh list"
  exit 1
fi

MANIFEST="models/manifest.json"
IN_MANIFEST=$(python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
print(1 if any(x['id'] == '$MATCH_ID' for x in m['models']) else 0)
")

if [[ "$IN_MANIFEST" != "1" ]]; then
  echo "'$MATCH_ID' not in manifest."
  exit 1
fi

TARGET="models/weights/$MATCH_ID"

if [[ ! -d "$TARGET" ]]; then
  echo "nothing downloaded for '$MATCH_ID'"
  exit 0
fi

SIZE=$(du -sh "$TARGET" | cut -f1)
read -r -p "delete $TARGET ($SIZE)? [y/N] " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
  echo "aborted"
  exit 0
fi

rm -rf "$TARGET"
echo "removed $TARGET"