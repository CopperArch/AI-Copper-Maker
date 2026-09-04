#!/bin/bash
# Headless LM Studio (llmster) setup for macOS.
#
# UNVERIFIED: written and reviewed against LM Studio's official docs
# (https://lmstudio.ai/docs/developer/core/headless) but never actually run
# on a Mac — this session only had a Linux (Bazzite) machine to test against.
# The equivalent Linux setup (same install command, a systemd --user service
# instead of the LaunchAgent below) is confirmed working; treat this as a
# best-effort port until it's been run for real on macOS.
set -euo pipefail

echo "Installing llmster (LM Studio headless) via the official installer..."
curl -fsSL https://lmstudio.ai/install.sh | bash

# The installer places `lms` in ~/.lmstudio/bin and should have added that to
# PATH already; source the shell profile it modified, or fall back to adding
# it for this script's own run.
export PATH="$HOME/.lmstudio/bin:$PATH"

if ! command -v lms &>/dev/null; then
    echo "lms not found on PATH after install — check ~/.lmstudio/bin exists and try opening a new shell." >&2
    exit 1
fi

echo "Starting the llmster daemon and API server once, to confirm it works..."
lms daemon up
lms server start
echo "Check http://localhost:1234/v1/models — it should list at least one model."

# ── Persistence: LaunchAgent (macOS's equivalent of the systemd --user unit
# used on Linux) ─────────────────────────────────────────────────────────────
PLIST="$HOME/Library/LaunchAgents/ai.llmcoder.lmstudio.plist"
LMS_BIN="$HOME/.lmstudio/bin/lms"

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.llmcoder.lmstudio</string>
    <key>ProgramArguments</key>
    <array>
        <string>$LMS_BIN</string>
        <string>daemon</string>
        <string>up</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/llmstudio-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/llmstudio-daemon.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "LaunchAgent installed and loaded — llmster will now start automatically at login."
echo "NOTE: this only brings the daemon up; if 'lms server start' doesn't persist its own"
echo "enabled-state across daemon restarts, add a second LaunchAgent (or an ExecStartPost-"
echo "style follow-up call) that runs 'lms server start' a few seconds after login too —"
echo "this wasn't verified either way since it couldn't be tested here."
