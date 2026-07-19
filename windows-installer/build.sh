#!/usr/bin/env bash
# Builds LLM-Coder-Setup.exe from the current working tree.
#
# Stages exactly the files `git ls-files` tracks (so none of the gitignored
# personal data — config.json, conversations.json, account credentials,
# release.keystore, etc. — can ever end up in a package handed to someone
# else) plus the Windows-only launcher/installer scripts, which are new and
# not yet tracked. Then compiles windows-installer/installer.nsi with
# makensis (install: `brew install makensis` — it's bottled/prebuilt, no
# Windows or Wine needed to build the installer itself, only to run it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

cd "$REPO_DIR"

echo "Staging tracked files..."
git ls-files | while IFS= read -r f; do
    mkdir -p "$STAGE_DIR/$(dirname "$f")"
    cp "$f" "$STAGE_DIR/$f"
done

echo "Adding Windows launcher/installer scripts..."
for f in install-windows.ps1 install-windows.bat launch-windows.ps1 launch-windows.bat; do
    if [[ ! -f "$REPO_DIR/$f" ]]; then
        echo "ERROR: expected $f in repo root but it's missing." >&2
        exit 1
    fi
    cp "$REPO_DIR/$f" "$STAGE_DIR/$f"
done

APP_VERSION="$(grep -oP '(?<=APP_VERSION = ")[^"]+' "$REPO_DIR/backend/main.py" || echo "1.1.0")"
echo "Building installer for version $APP_VERSION..."

mkdir -p "$SCRIPT_DIR/dist"
makensis -DSTAGE_DIR="$STAGE_DIR" -DAPP_VERSION="$APP_VERSION" "$SCRIPT_DIR/installer.nsi"

echo ""
echo "Built: $SCRIPT_DIR/dist/LLM-Coder-Setup.exe"
